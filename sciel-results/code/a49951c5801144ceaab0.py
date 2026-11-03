"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])
TRI_AREA = SQRT3 / 4.0

# Precompute all triples
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]
M_TRIPLES = len(_IDX)


def _bary_to_cart(bary):
    """bary (n,2) -> cartesian (n,2). point = u*V1 + v*V2."""
    x = bary[:, 0] + 0.5 * bary[:, 1]
    y = H * bary[:, 1]
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """cartesian (n,2) -> barycentric (n,2)."""
    v = pts[:, 1] / H
    u = pts[:, 0] - 0.5 * v
    return np.column_stack([u, v])


def _clamp_bary(b):
    """Clamp barycentric coords to triangle: u>=0, v>=0, u+v<=1."""
    b = b.copy()
    b[:, 0] = np.clip(b[:, 0], 0.0, 1.0)
    b[:, 1] = np.clip(b[:, 1], 0.0, 1.0)
    s = b[:, 0] + b[:, 1]
    mask = s > 1.0
    if np.any(mask):
        b[mask] = b[mask] / s[mask, None]
    # Tiny margin for safety
    eps = 1e-10
    b[:, 0] = np.clip(b[:, 0], eps, 1.0 - eps)
    b[:, 1] = np.clip(b[:, 1], eps, 1.0 - eps)
    s = b[:, 0] + b[:, 1]
    mask = s > 1.0 - 2 * eps
    if np.any(mask):
        b[mask] = b[mask] * (1.0 - 2 * eps) / s[mask, None]
    return b


def _min_area_cart(pts):
    """Exact min triangle area for cartesian points (n,2)."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _make_softmin_obj(temperature):
    """Create a smooth soft-min objective in barycentric coordinates.
    
    Returns a function f(x) where x is (2N,) barycentric coords.
    Uses -logsumexp(areas/tau)/tau as smooth approximation of -min(areas).
    """
    def obj(x):
        bary = x.reshape(N, 2)
        # Clamp to valid region (for safety, though bounds should handle it)
        bary = _clamp_bary(bary)
        pts = _bary_to_cart(bary)
        pi = pts[_I]
        pj = pts[_J]
        pk = pts[_K]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
                (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        # Soft-min: -logsumexp(areas/tau) / tau
        a = areas / temperature
        a_max = np.max(a)
        logsumexp = a_max + np.log(np.sum(np.exp(a - a_max)))
        return -logsumexp / temperature
    
    return obj


def _make_exact_obj(x):
    """Exact (non-smooth) objective: negative min area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    pts = _bary_to_cart(bary)
    return -_min_area_cart(pts)


def _lbfgs_optimize(bary0, temperature, maxiter=300):
    """Run L-BFGS-B with soft-min objective."""
    x0 = bary0.flatten()
    bounds = [(0.0, 1.0)] * (2 * N)
    obj = _make_softmin_obj(temperature)
    try:
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()


def _nm_optimize(bary0, maxiter=1000):
    """Nelder-Mead with exact objective (for final polish)."""
    x0 = bary0.flatten()
    try:
        res = minimize(
            _make_exact_obj, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()


def _random_bary(seed):
    """Random points in barycentric triangle."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 2))
    for i in range(N):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1 - u)
        bary[i] = [u, v]
    return bary


def _grid_bary(level, seed=42):
    """Grid initialization in barycentric space with jitter."""
    rng = np.random.RandomState(seed)
    pts = []
    for i in range(level + 1):
        for j in range(level + 1 - i):
            u = i / level
            v = j / level
            if u + v <= 1.0:
                ju = rng.uniform(-0.02, 0.02)
                jv = rng.uniform(-0.02, 0.02)
                bu, bv = u + ju, v + jv
                bu = max(0.0, min(bu, 1.0))
                bv = max(0.0, min(bv, 1.0))
                if bu + bv > 1.0:
                    s = 1.0 / (bu + bv)
                    bu *= s
                    bv *= s
                pts.append([bu, bv])
    if len(pts) > N:
        idx = rng.choice(len(pts), size=N, replace=False)
        pts = [pts[i] for i in idx]
    elif len(pts) < N:
        while len(pts) < N:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1 - u)
            pts.append([u, v])
    return np.array(pts)


def _symmetric_init(seed=42):
    """Symmetric initialization: vertices, midpoints, centroid, and symmetric interior."""
    rng = np.random.RandomState(seed)
    # Start with a symmetric structure
    bary = np.array([
        [0.0, 0.0],   # V0
        [1.0, 0.0],   # V1
        [0.0, 1.0],   # V2
        [0.5, 0.0],   # midpoint V0-V1
        [0.0, 0.5],   # midpoint V0-V2
        [0.5, 0.5],   # midpoint V1-V2
        [1.0/3, 1.0/3],  # centroid
    ])
    # Add 4 more points with 3-fold symmetry around centroid
    # In barycentric, 3-fold symmetry: (u,v) -> (v, u) -> (1-u-v, u)
    for _ in range(2):  # 2 orbits of 2 (one fixed point + one orbit of 3 would be 4, but let's do 2 orbits of 2)
        # Random point in interior
        u = rng.uniform(0.1, 0.6)
        v = rng.uniform(0.1, 0.6 - u)
        # Orbit under 3-fold: (u,v), (v, 1-u-v), (1-v, u)
        # Wait, 3-fold in barycentric: (u,v,w) -> (v,w,u) where w=1-u-v
        # So (u,v) -> (v, 1-u-v) -> (1-u-v, u) -> (u,v)
        p1 = [u, v]
        p2 = [v, 1.0 - u - v]
        p3 = [1.0 - u - v, u]
        # Pick 2 of 3 (or all 3 if we have room)
        bary = np.vstack([bary, [p1, p2]])
    
    # We now have 7 + 4 = 11 points
    bary = _clamp_bary(bary)
    return bary


def _perturb_best(bary, seed, scale=0.05):
    """Perturb a barycentric configuration."""
    rng = np.random.RandomState(seed)
    pert = rng.uniform(-scale, scale, size=bary.shape)
    return _clamp_bary(bary + pert)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    def score(bary):
        pts = _bary_to_cart(bary)
        return _min_area_cart(pts)
    
    best_bary = None
    best_score = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_score
        bary = _clamp_bary(bary)
        s = score(bary)
        if s > best_score:
            best_score = s
            best_bary = bary.copy()
    
    # Generate diverse initializations
    inits = []
    
    # Symmetric inits
    for s in [42, 123, 7, 99, 2024, 31415]:
        inits.append(_symmetric_init(seed=s))
    
    # Grid inits
    for level in [2, 3, 4]:
        for s in [42, 7, 999]:
            inits.append(_grid_bary(level, seed=s))
    
    # Random inits
    for s in [42, 123, 7, 999, 2024, 31415, 27182, 16180]:
        inits.append(_random_bary(seed=s))
    
    # Score and sort initializations
    scored = []
    for init in inits:
        s = score(init)
        scored.append((s, init))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Multi-start optimization with annealing temperature
    # Phase 1: Coarse (high temperature, smooth landscape)
    # Phase 2: Fine (low temperature, sharper)
    # Phase 3: Exact Nelder-Mead polish
    
    temperatures_coarse = [0.02, 0.01]
    temperatures_fine = [0.005, 0.002, 0.001]
    
    # How many starts to do in each phase
    n_coarse = min(8, len(scored))
    n_fine = min(12, len(scored))
    
    # Phase 1: Coarse L-BFGS-B
    for idx in range(n_coarse):
        if time_left() < 15.0:
            break
        init_bary = scored[idx][1]
        for temp in temperatures_coarse:
            bary_opt = _lbfgs_optimize(init_bary, temperature=temp, maxiter=200)
            update_best(bary_opt)
            init_bary = bary_opt  # Chain: use result as next init
    
    # Phase 2: Fine L-BFGS-B (from best results of phase 1 + more inits)
    # Use the current best as the primary start, plus some from the scored list
    fine_starts = []
    if best_bary is not None:
        fine_starts.append(best_bary)
    for idx in range(n_fine):
        fine_starts.append(scored[idx][1])
    
    for idx, init_bary in enumerate(fine_starts):
        if time_left() < 15.0:
            break
        for temp in temperatures_fine:
            bary_opt = _lbfgs_optimize(init_bary, temperature=temp, maxiter=300)
            update_best(bary_opt)
            init_bary = bary_opt
    
    # Phase 3: Nelder-Mead polish on the best
    if best_bary is not None:
        # Polish the best
        bary_polished = _nm_optimize(best_bary, maxiter=2000)
        update_best(bary_polished)
        
        # Polish a few perturbations of the best
        if time_left() > 20.0:
            for s in [1, 2, 3, 4, 5]:
                if time_left() < 10.0:
                    break
                pert = _perturb_best(best_bary, seed=s, scale=0.03)
                bary_p = _nm_optimize(pert, maxiter=1000)
                update_best(bary_p)
    
    # Phase 4: If time remains, try more diverse starts with full pipeline
    if best_bary is not None and time_left() > 30.0:
        rng = np.random.RandomState(77777)
        for _ in range(3):
            if time_left() < 15.0:
                break
            # Random perturbation of best
            pert = _perturb_best(best_bary, seed=int(rng.randint(100000)), scale=0.1)
            # Coarse -> Fine -> Polish
            for temp in [0.01, 0.003]:
                bary_opt = _lbfgs_optimize(pert, temperature=temp, maxiter=200)
                pert = bary_opt
            bary_final = _nm_optimize(pert, maxiter=1000)
            update_best(bary_final)
    
    if best_bary is None:
        best_bary = _random_bary(0)
    
    return _bary_to_cart(best_bary)
