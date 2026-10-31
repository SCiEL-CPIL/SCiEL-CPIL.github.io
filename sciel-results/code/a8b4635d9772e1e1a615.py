"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2.0
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _valid(p):
    """Check if all points are strictly inside the triangle (0,0),(1,0),(0.5,H)."""
    x, y = p[:, 0], p[:, 1]
    if np.any(y < 0) or np.any(y > H):
        return False
    # y <= H * x  (left edge)
    if np.any(y > H * x + 1e-9):
        return False
    # y <= H * (1 - x)  (right edge)
    if np.any(y > H * (1.0 - x) + 1e-9):
        return False
    return True

def _min_area(p):
    """Smallest triangle area among all C(11,3)=165 triangles."""
    n = len(p)
    best = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ax, ay = p[i, 0], p[i, 1]
                bx, by = p[j, 0], p[j, 1]
                cx, cy = p[k, 0], p[k, 1]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < best:
                    best = area
    return best

def _obj(p):
    """Objective for scipy: negative min area (we maximize)."""
    if not _valid(p):
        return 1e10
    return -_min_area(p)

def _refine(p):
    """Nelder-Mead refinement of a candidate."""
    try:
        res = minimize(_obj, p, method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success and _valid(res.x):
            return res.x
    except Exception:
        pass
    return p

def construct():
    t0 = time.time()
    best_p = None
    best_val = -np.inf

    def _try_update(p):
        nonlocal best_p, best_val
        if not _valid(p):
            return
        v = _min_area(p)
        if v > best_val:
            best_val = v
            best_p = p.copy()

    # ---- Phase 1: Differential Evolution (global search) ----
    # Bounds: x in [0,1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    # Use a fixed seed for reproducibility but allow DE to explore
    de_kwargs = dict(
        bounds=bounds,
        seed=42,
        maxiter=60,
        popsize=18,
        tol=1e-6,
        polish=False,
        init='sobol',
        update_interval=10,
    )

    # Time check before DE
    if time.time() - t0 < DEADLINE - 30:
        try:
            de_res = differential_evolution(_obj, bounds, **de_kwargs)
            if de_res.success:
                _try_update(de_res.x)
        except Exception:
            pass

    # ---- Phase 2: Nelder-Mead refinement from multiple starts ----
    # Generate diverse starting points: structured + random
    rng = np.random.RandomState(123)
    starts = []

    # Structured: grid-like points in the triangle
    for gx in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for gy in [0.1, 0.3, 0.5]:
            # Map to triangle via affine transform
            u, v = gx, gy
            if u + v <= 1.0:
                starts.append([u + 0.5 * v, H * v])
    # Random starts
    for _ in range(8):
        u, v = rng.random(2)
        if u + v < 1.0:
            starts.append([u + 0.5 * v, H * v])

    # Add DE result as a start if available
    if best_p is not None:
        starts.append(best_p.copy())

    # Refine each start with time guard
    for s in starts:
        if time.time() - t0 > DEADLINE - 10:
            break
        if not _valid(s):
            continue
        refined = _refine(s)
        _try_update(refined)

    # ---- Phase 3: Final polish on the best point ----
    if best_p is not None and time.time() - t0 < DEADLINE - 5:
        final = _refine(best_p)
        _try_update(final)

    if best_p is None:
        # Fallback: return a valid configuration
        pts = []
        for i in range(N):
            u = 0.1 + 0.8 * (i % 4) / 3.0
            v = 0.1 + 0.6 * (i // 4) / 2.0
            if u + v <= 1.0:
                pts.append([u + 0.5 * v, H * v])
        best_p = np.array(pts[:N])

    return best_p
# EVOLVE-BLOCK-END
