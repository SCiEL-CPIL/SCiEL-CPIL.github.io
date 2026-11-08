"""Circle packing in perimeter-4 rectangle: Voronoi-derived radii + smooth center optimization."""
import numpy as np
import time

N = 21
DEADLINE = time.time() + 140  # safety margin before 150s kill


def _derived_radii_and_penalty(cx, cy, w, h):
    """
    For given centers and box (w,h) with w+h=2:
      r_i = min( min_j dist(i,j)/2 for j!=i, cx_i, cy_i, w-cx_i, h-cy_i )
    We want to maximize sum r_i.
    Use a smooth approximation: for each pair, soft-min of (dist/2) and boundary distances.
    Penalty term for any "violation" is not needed since r_i is capped by min;
    but to make gradient smooth, we use a soft-min over all constraints.
    """
    n = len(cx)
    # Boundary distances for each circle
    bx = np.minimum(cx, w - cx)
    by = np.minimum(cy, h - cy)
    bdist = np.minimum(bx, by)  # shape (n,)

    # Pairwise distances / 2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist, axis=1) / 2.0  # shape (n,)

    # r_i = min(bdist_i, pair_min_i)  -- use soft-min for smoothness
    # soft_min(a, b) = -logaddexp(-a/k, -b/k) * k  approximates min
    k = 0.01
    r = -k * np.logaddexp(-bdist / k, -pair_min / k)

    # Ensure non-negative
    r = np.maximum(r, 0.0)
    return r


def _objective(params, k):
    """
    params: [cx_0, cy_0, ..., cx_{n-1}, cy_{n-1}, w]
    h = 2 - w
    Returns negative sum of radii (to minimize).
    Add a large penalty if w or h <= 0 or any center outside box.
    """
    n = N
    cx = params[:n]
    cy = params[n:2*n]
    w = params[2*n]
    h = 2.0 - w

    # Penalty for invalid box
    if w <= 0.1 or h <= 0.1:
        return 1e6

    # Penalty for centers outside box (soft)
    outside = (cx < 0) | (cx > w) | (cy < 0) | (cy > h)
    if np.any(outside):
        # smooth penalty
        ox = np.maximum(0, -cx, cx - w)
        oy = np.maximum(0, -cy, cy - h)
        pen = np.sum(ox**2 + oy**2) * 100.0
    else:
        pen = 0.0

    r = _derived_radii_and_penalty(cx, cy, w, h)
    return -(np.sum(r)) + pen


def _bounds():
    """Bounds for params."""
    lb = np.zeros(2 * N + 1)
    ub = np.zeros(2 * N + 1)
    # cx, cy will be bounded by w,h dynamically, but L-BFGS-B needs static bounds.
    # We'll use loose bounds and rely on penalty.
    lb[:2*N] = 0.0
    ub[:2*N] = 2.0
    lb[2*N] = 0.1
    ub[2*N] = 1.9
    return lb, ub


def _gradient(params, k):
    """Finite-difference gradient (L-BFGS-B can do it, but explicit may be faster)."""
    # Use L-BFGS-B's internal numerical gradient instead.
    return None  # Let L-BFGS-B compute it.


def construct():
    start = time.time()
    lb, ub = _bounds()
    best_params = None
    best_val = -np.inf

    # Try several initializations
    rng = np.random.default_rng(42)

    # Initialization 1: grid-like
    side = int(np.ceil(np.sqrt(N)))
    init1 = np.zeros(2 * N + 1)
    init1[2*N] = 1.0  # w=1, h=1
    for i in range(N):
        ix = i % side
        iy = i // side
        init1[i] = (ix + 0.5) / side
        init1[N + i] = (iy + 0.5) / side

    # Initialization 2: random in square
    init2 = np.zeros(2 * N + 1)
    init2[2*N] = 1.0
    init2[:N] = rng.uniform(0.1, 0.9, N)
    init2[N:2*N] = rng.uniform(0.1, 0.9, N)

    # Initialization 3: random in rectangle w=1.5, h=0.5
    init3 = np.zeros(2 * N + 1)
    init3[2*N] = 1.5
    init3[:N] = rng.uniform(0.05, 1.45, N)
    init3[N:2*N] = rng.uniform(0.05, 0.45, N)

    # Initialization 4: random in rectangle w=0.5, h=1.5
    init4 = np.zeros(2 * N + 1)
    init4[2*N] = 0.5
    init4[:N] = rng.uniform(0.05, 0.45, N)
    init4[N:2*N] = rng.uniform(0.05, 1.45, N)

    inits = [init1, init2, init3, init4]

    # Add more random inits if time permits
    for attempt in range(10):
        w_try = rng.uniform(0.3, 1.7)
        h_try = 2.0 - w_try
        init_r = np.zeros(2 * N + 1)
        init_r[2*N] = w_try
        init_r[:N] = rng.uniform(0.02, max(0.02, w_try - 0.02), N)
        init_r[N:2*N] = rng.uniform(0.02, max(0.02, h_try - 0.02), N)
        inits.append(init_r)

    for idx, x0 in enumerate(inits):
        if time.time() > DEADLINE - 5:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(
                _objective, x0, args=(0,),
                method='L-BFGS-B',
                bounds=list(zip(lb, ub)),
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            continue

    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    cx = best_params[:N]
    cy = best_params[N:2*N]
    w = best_params[2*N]
    h = 2.0 - w

    # Compute final radii (hard min, no soft)
    n = N
    bx = np.minimum(cx, w - cx)
    by = np.minimum(cy, h - cy)
    bdist = np.minimum(bx, by)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist, axis=1) / 2.0
    r = np.minimum(bdist, pair_min)
    r = np.maximum(r, 1e-6)

    # Slightly shrink radii to ensure disjoint interiors (strict inequality)
    r = r * (1.0 - 1e-6)

    result = np.column_stack([cx, cy, r])
    return result
