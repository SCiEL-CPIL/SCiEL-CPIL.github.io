"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

DEADLINE = 140.0  # safety margin before 150s


def _penalty_objective(v, lam_b, lam_o):
    """v: [x(26), y(26), r(26)]. Minimize -sum(r) + penalties."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]

    obj = -r.sum()

    # Boundary penalty
    b = (np.maximum(0, r - x) + np.maximum(0, r - (1.0 - x)) +
         np.maximum(0, r - y) + np.maximum(0, r - (1.0 - y)))
    obj += lam_b * b.sum()

    # Overlap penalty (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(np.maximum(0, dx**2 + dy**2))
    rsum = r[:, None] + r[None, :]
    overlap = np.maximum(0, rsum - dist)
    np.fill_diagonal(overlap, 0.0)
    obj += lam_o * (overlap**2).sum()

    return obj


def _is_valid(arr, tol=1e-7):
    """Check if arrangement is valid (all constraints satisfied)."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _make_grid_initial():
    """Grid initial guess."""
    side = 6  # 6x6 = 36 points, enough for 26
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    pts = pts[:N]
    arr = np.array(pts)
    r0 = 0.5 / side * 0.9
    for i in range(N):
        r0 = min(r0, arr[i, 0], 1 - arr[i, 0], arr[i, 1], 1 - arr[i, 1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])


def _make_hex_initial():
    """Hexagonal packing initial guess."""
    pts = []
    for row in range(6):
        for col in range(5):
            x = 0.1 + col * 0.2 + (0.1 if row % 2 == 1 else 0.0)
            y = 0.1 + row * 0.16
            pts.append([x, y])
    pts = pts[:N]
    arr = np.array(pts)
    min_dist = np.inf
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            min_dist = min(min_dist, d)
    r0 = min_dist / 2 * 0.85
    for i in range(N):
        r0 = min(r0, pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])


def _make_random_initial(seed):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    min_dists = np.full(N, 0.3)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dists[i] = min(min_dists[i], d)
    r = 0.4 * min_dists
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.01)
    return np.column_stack([x, y, r])


def _optimize_lbfgsb(arr_init, lam_b, lam_o, maxiter=300):
    """Run L-BFGS-B optimization from initial guess."""
    v0 = arr_init.flatten()
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    try:
        res = minimize(
            _penalty_objective, v0,
            args=(lam_b, lam_o),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return arr_init


def _optimize_nm(arr_init, lam_b, lam_o, maxiter=200):
    """Run Nelder-Mead polish from initial guess."""
    v0 = arr_init.flatten()
    try:
        res = minimize(
            _penalty_objective, v0,
            args=(lam_b, lam_o),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        # Clamp to bounds
        x = np.clip(res.x[:N], 0, 1)
        y = np.clip(res.x[N:2*N], 0, 1)
        r = np.clip(res.x[2*N:3*N], 0, 0.5)
        return np.column_stack([x, y, r])
    except Exception:
        return arr_init


def _fix_validity(arr):
    """Iteratively reduce radii to fix overlaps and boundary violations."""
    x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    # Fix boundary
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.0)
    # Iteratively reduce to fix overlaps
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j]:
                    excess = r[i] + r[j] - d
                    r[i] = max(0, r[i] - excess / 2)
                    r[j] = max(0, r[j] - excess / 2)
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])


def construct():
    start_time = time.time()
    best_arr = None
    best_val = -np.inf

    # Generate initial guesses
    inits = []
    inits.append(_make_grid_initial())
    inits.append(_make_hex_initial())
    for seed in [42, 123, 456, 789, 1024, 2048, 3141, 2718]:
        inits.append(_make_random_initial(seed))

    # Phase 1: Differential Evolution for global search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 40:
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        try:
            de_res = differential_evolution(
                _penalty_objective, bounds,
                args=(100.0, 500.0),
                seed=42, maxiter=50, popsize=15,
                tol=1e-8, polish=True,
                callback=lambda x, convergence: (time.time() - start_time > DEADLINE - 20)
            )
            if de_res.x is not None:
                de_arr = de_res.x.reshape(N, 3)
                de_arr = _fix_validity(de_arr)
                if _is_valid(de_arr):
                    val = de_arr[:, 2].sum()
                    if val > best_val:
                        best_val = val
                        best_arr = de_arr.copy()
        except Exception:
            pass

    # Phase 2: Multi-start L-BFGS-B with different lambda values
    lambda_sets = [
        (10.0, 100.0),
        (50.0, 500.0),
        (100.0, 1000.0),
        (200.0, 2000.0),
    ]

    for lam_b, lam_o in lambda_sets:
        if time.time() - start_time > DEADLINE - 20:
            break
        for arr_init in inits:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                arr = _optimize_lbfgsb(arr_init, lam_b, lam_o, maxiter=200)
                arr = _fix_validity(arr)
                if _is_valid(arr):
                    val = arr[:, 2].sum()
                    if val > best_val:
                        best_val = val
                        best_arr = arr.copy()
            except Exception:
                continue

    # Phase 3: Nelder-Mead polish on best candidates
    if best_arr is not None and time.time() - start_time < DEADLINE - 10:
        for lam_b, lam_o in [(100.0, 500.0), (500.0, 2000.0)]:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                polished = _optimize_nm(best_arr, lam_b, lam_o, maxiter=300)
                polished = _fix_validity(polished)
                if _is_valid(polished):
                    val = polished[:, 2].sum()
                    if val > best_val:
                        best_val = val
                        best_arr = polished.copy()
            except Exception:
                continue

    # Phase 4: High-precision final polish
    if best_arr is not None and time.time() - start_time < DEADLINE - 3:
        try:
            v0 = best_arr.flatten()
            bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
            res = minimize(
                _penalty_objective, v0,
                args=(500.0, 2000.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            final_arr = res.x.reshape(N, 3)
            final_arr = _fix_validity(final_arr)
            if _is_valid(final_arr):
                val = final_arr[:, 2].sum()
                if val > best_val:
                    best_val = val
                    best_arr = final_arr.copy()
        except Exception:
            pass

    # Fallback
    if best_arr is None:
        best_arr = _make_grid_initial()

    return best_arr
# EVOLVE-BLOCK-END
