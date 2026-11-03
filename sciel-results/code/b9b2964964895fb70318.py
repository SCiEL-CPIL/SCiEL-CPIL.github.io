"""Pack 50 equal spheres in the unit cube [-1/2,1/2]^3 to maximise common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5
# Best known reference radius (Pfoertner)
REF_R = 0.1359545126417


def _score(centers):
    """Raw radius: min over pair distances/2 and wall distances."""
    # wall term
    wall = HALF - np.abs(centers)
    wall_min = wall.min()
    # pair distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    pair_min = d[iu].min()
    return 0.5 * pair_min, wall_min


def _radius(centers):
    pm, wm = _score(centers)
    return min(pm, wm)


def _objective_flat(x, beta):
    """Smooth surrogate to maximise: minimise -log r_smooth.
    r_smooth = min( logsumexp(-beta*pair/2), logsumexp(-beta*wall) )
    Minimising -log(min(a,b)) = max(-log a, -log b) = -logsumexp(log a, log b)...
    Use: we want large r. r_smooth = min(A,B) where A=pair term, B=wall term.
    log r_smooth = min(logA, logB) = -max(-logA,-logB).
    -log r_smooth = max(-logA, -logB) = logsumexp(-logA, -logB)  (soft max).
    """
    c = x.reshape(N, 3)
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    pd = d[iu] / 2.0
    wall = HALF - np.abs(c)
    w = wall.min(axis=1)
    # logA = log(mean exp(-beta*pd)) approx; use logsumexp for stability
    A = -np.logaddexp.reduce(-beta * pd)  # log of sum exp(-beta pd)
    B = -np.logaddexp.reduce(-beta * w)
    # -log r_smooth = max(-A, -B) = logsumexp(-A, -B)
    neglog = np.logaddexp(-A, -B)
    return neglog


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(float)


def _init_random(seed):
    rng = np.random.default_rng(seed)
    return (HALF * (1.0 - 2 * REF_R)) * (2.0 * rng.random((N, 3)) - 1.0)


def _optimize(x0, beta, deadline, maxiter):
    x0 = x0.reshape(-1).astype(float)
    best_x = x0.copy()
    best_val = _radius(x0.reshape(N, 3))
    t_end = time.time() + (deadline - time.time())
    for it in range(maxiter):
        if time.time() > t_end - 2.0:
            break
        try:
            res = minimize(_objective_flat, x0, args=(beta,), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            break
        x0 = res.x
        val = _radius(x0.reshape(N, 3))
        if val > best_val:
            best_val = val
            best_x = x0.copy()
        # stop if very close to reference and no room
        if best_val >= REF_R - 1e-9:
            break
    return best_x, best_val


def construct():
    deadline = time.time() + 140.0  # safety margin vs 150s kill
    best_x = _init_grid()
    best_val = _radius(best_x)

    rng = np.random.default_rng(0)

    # Multi-start: grid + several random starts, with annealing of beta.
    beta_schedule = [20.0, 40.0, 80.0, 160.0]
    starts = [best_x]
    for s in range(6):
        starts.append(_init_random(s * 1000 + 7))

    for start in starts:
        if time.time() > deadline - 5.0:
            break
        cur = start.copy()
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
            cur, val = _optimize(cur, beta, deadline, maxiter=1)
            if val > best_val:
                best_val = val
                best_x = cur.copy()

    # Final high-precision polish on best
    if time.time() < deadline - 5.0:
        best_x, val = _optimize(best_x, 300.0, deadline, maxiter=1)
        if val > best_val:
            best_val = val
            best_x = best_x

    # Safety: clip to be strictly inside
    out = best_x.reshape(N, 3)
    out = np.clip(out, -HALF + 1e-9, HALF - 1e-9)
    return out


if __name__ == "__main__":
    c = construct()
    print("radius:", _radius(c))
    print("shape:", c.shape)
