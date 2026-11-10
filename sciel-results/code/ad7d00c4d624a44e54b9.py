# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_S = 630.0
_SAFETY_S = 30.0  # return this far before the hard kill

# Sofa grid points (shared, computed once)
def _sofa_points():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return pts

_SOFAS = None

def _get_sofas():
    global _SOFAS
    if _SOFAS is None:
        _SOFAS = _sofa_points()
    return _SOFAS


def _densify(path):
    """Insert the evaluator's endpoints and densify to <=1-degree angle steps."""
    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pN = np.array([-2.0, 0.0, 90.0], dtype=np.float64)
    full = np.vstack([p0, path, pN])
    # Deduplicate consecutive identical rows (in case user already has endpoints)
    keep = [0]
    for i in range(1, len(full)):
        if not np.allclose(full[i], full[i - 1], atol=1e-12):
            keep.append(i)
    full = full[keep]

    out = [full[0]]
    for i in range(1, len(full)):
        a1, a2 = full[i - 1, 2], full[i, 2]
        da = a2 - a1
        m = max(1, int(np.ceil(abs(da))))
        for k in range(1, m + 1):
            t = k / m
            out.append(full[i - 1] + t * (full[i] - full[i - 1]))
    return np.array(out, dtype=np.float64)


def _evaluate(dense):
    """Return raw score (area) for a densified path of shape (P,3)."""
    sofas = _get_sofas()
    n = len(sofas)
    mask = np.ones(n, dtype=bool)
    P = len(dense)
    for i in range(P):
        x, y, a = dense[i]
        rad = np.deg2rad(-a)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        local = (sofas - np.array([x, y])) @ R.T  # R(-a)(w - (x,y))
        u = local[:, 0]
        v = local[:, 1]
        # base = {0<=u<=1, -3<=v<=1} union {-3<=u<=1, 0<=v<=1}
        in1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask &= (in1 | in2)
        if not mask.any():
            return 0.0
    return float(mask.sum()) * 0.0001


def _evaluate_path(path):
    return _evaluate(_densify(path))


def _initial_path():
    # Start from the seed's hand-tuned path (a gentle arc) and add a couple
    # of interior waypoints for more freedom.
    base = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    return base


def moving_sofa_2d() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + _DEADLINE_S - _SAFETY_S

    best_path = _initial_path()
    best_score = _evaluate_path(best_path)

    # Nelder–Mead local search on the flattened (n*3) parameter vector.
    # We allow x, y, and a to move. Small step sizes to stay in a sensible region.
    def _obj(vec):
        p = vec.reshape(-1, 3)
        # Keep positions in a reasonable box to avoid absurd placements
        # (the evaluator will still densify and evaluate; we just want a
        # well-behaved search).
        p[:, 0] = np.clip(p[:, 0], -3.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], -3.0, 1.0)
        p[:, 2] = np.clip(p[:, 2], 0.0, 90.0)
        return -_evaluate_path(p)

    n_wp = len(best_path)
    x0 = best_path.ravel().astype(np.float64)

    from scipy.optimize import minimize

    n_iter = 0
    max_iter_per_run = 400  # cap per run to bound worst case

    # First, a single NM run from the initial path.
    try:
        res = minimize(
            _obj, x0, method="Nelder-Mead",
            options={"maxiter": max_iter_per_run, "xatol": 1e-4, "fatol": 1e-7,
                     "adaptive": True},
        )
        n_iter += res.nfev
        if res.fun < -best_score:
            cand = res.x.reshape(-1, 3)
            cand[:, 0] = np.clip(cand[:, 0], -3.0, 1.0)
            cand[:, 1] = np.clip(cand[:, 1], -3.0, 1.0)
            cand[:, 2] = np.clip(cand[:, 2], 0.0, 90.0)
            s = _evaluate_path(cand)
            if s > best_score:
                best_score = s
                best_path = cand
    except Exception:
        pass

    # Restart loop: perturb the current best and run a shorter NM.
    rng = np.random.default_rng(12345)
    restart = 0
    while time.time() < deadline and restart < 30:
        # Perturb best_path slightly
        noise = rng.normal(0.0, 0.15, size=best_path.shape)
        noise[:, 2] *= 2.0  # larger angle noise
        x_start = (best_path + noise).ravel().astype(np.float64)
        x_start[:, 2] = np.clip(x_start[:, 2], 0.0, 90.0)
        x_start[:, 0] = np.clip(x_start[:, 0], -3.0, 1.0)
        x_start[:, 1] = np.clip(x_start[:, 1], -3.0, 1.0)

        try:
            res = minimize(
                _obj, x_start, method="Nelder-Mead",
                options={"maxiter": max(50, max_iter_per_run // 3),
                         "xatol": 1e-4, "fatol": 1e-7, "adaptive": True},
            )
            n_iter += res.nfev
            if res.fun < -best_score:
                cand = res.x.reshape(-1, 3)
                cand[:, 0] = np.clip(cand[:, 0], -3.0, 1.0)
                cand[:, 1] = np.clip(cand[:, 1], -3.0, 1.0)
                cand[:, 2] = np.clip(cand[:, 2], 0.0, 90.0)
                s = _evaluate_path(cand)
                if s > best_score:
                    best_score = s
                    best_path = cand
        except Exception:
            pass

        restart += 1
        # If we're getting close to the deadline, stop.
        if time.time() > deadline - 15.0:
            break

    # Final safety: make sure best_path is a valid finite array.
    best_path = np.asarray(best_path, dtype=np.float64)
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_path()
    return best_path


# EVOLVE-BLOCK-END
