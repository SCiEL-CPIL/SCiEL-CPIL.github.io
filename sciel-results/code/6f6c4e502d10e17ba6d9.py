"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

def _cart_to_bary(p):
    """Convert Cartesian to barycentric (u,v): p = u*V1 + v*V2."""
    v = p[1] / H
    u = p[0] - v * 0.5
    return np.array([u, v])

def _bary_to_cart(bary):
    """Vectorized barycentric to Cartesian."""
    return bary[:, 0:1] * V1 + bary[:, 1:2] * V2

def _clamp_bary(bary):
    """Clamp barycentric coords to valid region."""
    b = np.clip(bary, 0, 1)
    s = b[:, 0] + b[:, 1]
    mask = s > 1
    if np.any(mask):
        b[mask] /= s[mask, np.newaxis]
    return b

def _min_area(points):
    """Compute min triangle area among all triples."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                a = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if a < min_a:
                    min_a = a
    return min_a

def _neg_min_area_cart(x):
    """Objective in Cartesian: negative min area. x is flat array of 2*N coords."""
    pts = x.reshape(N, 2)
    # Check all points are inside triangle (with small tolerance)
    for i in range(N):
        px, py = pts[i]
        if px < -1e-9 or py < -1e-9 or px > 1 + 1e-9 or py > H + 1e-9:
            return 10.0  # penalty
        # Check inside: barycentric coords must be >= 0
        v = py / H
        u = px - v * 0.5
        if u < -1e-9 or v < -1e-9 or u + v > 1 + 1e-9:
            return 10.0
    return -_min_area(pts)

def _nm_cartesian(x0, maxiter=2000):
    """Nelder-Mead in Cartesian coordinates."""
    try:
        res = minimize(
            _neg_min_area_cart, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _grid_init_cart(level, seed=42):
    """Deterministic grid with structured jitter in Cartesian."""
    rng = np.random.RandomState(seed)
    pts = []
    for i in range(level + 1):
        for j in range(level + 1 - i):
            u = i / level
            v = j / level
            if u + v <= 1.0:
                # Jitter in barycentric space
                ju = rng.uniform(-0.03, 0.03)
                jv = rng.uniform(-0.03, 0.03)
                bu, bv = u + ju, v + jv
                bu = max(0.0, min(bu, 1.0))
                bv = max(0.0, min(bv, 1.0))
                if bu + bv > 1.0:
                    scale = 1.0 / (bu + bv)
                    bu *= scale
                    bv *= scale
                px = bu * 1.0 + bv * 0.5
                py = bv * H
                pts.append([px, py])
    
    if len(pts) > N:
        idx = rng.choice(len(pts), size=N, replace=False)
        pts = [pts[i] for i in idx]
    elif len(pts) < N:
        while len(pts) < N:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1 - u)
            pts.append([u + 0.5 * v, v * H])
    return np.array(pts)

def _random_init_cart(seed):
    """Random points in Cartesian inside triangle."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1 - u)
        pts.append([u + 0.5 * v, v * H])
    return np.array(pts)

def _de_global(bounds, maxiter=100, popsize=15, seed=42):
    """Differential evolution for global search in Cartesian."""
    # Bounds: each point (x, y) with x in [0,1], y in [0,H]
    # But we need y <= H*(1-x) for x<=0.5, and y <= H*x for x>0.5... 
    # Actually simpler: use barycentric for DE, convert to cart
    def de_obj(x_flat):
        bary = x_flat.reshape(N, 2)
        bary = _clamp_bary(bary)
        pts = _bary_to_cart(bary)
        return -_min_area(pts)
    
    # Bounds in barycentric: u in [0,1], v in [0,1] (clamped later)
    b = [(0, 1)] * (2 * N)
    try:
        res = differential_evolution(
            de_obj, b, maxiter=maxiter, popsize=popsize,
            seed=seed, tol=1e-8, mutation=(0.5, 1.5), recombination=0.8,
            polish=True, init='sobol'
        )
        return res.x.reshape(N, 2)
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -float('inf')
    
    def update_best(pts):
        nonlocal best_points, best_score
        # Verify all points are inside
        for i in range(N):
            px, py = pts[i]
            if px < -1e-8 or py < -1e-8:
                return
            v = py / H
            u = px - v * 0.5
            if u < -1e-8 or v < -1e-8 or u + v > 1 + 1e-8:
                return
        s = _min_area(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Phase 1: Differential Evolution for global search (barycentric space)
    if time.time() < deadline - 60.0:
        de_result = _de_global(None, maxiter=80, popsize=12, seed=42)
        if de_result is not None:
            de_pts = _bary_to_cart(_clamp_bary(de_result))
            update_best(de_pts)
    
    # Phase 2: Multi-start Nelder-Mead in Cartesian from structured initializations
    inits = []
    # Grid-based initializations with various levels and jitter seeds
    for level in [2, 3, 4, 5, 6]:
        for seed in [42, 123, 7, 999, 2024]:
            inits.append(_grid_init_cart(level, seed))
    # Random initializations
    for seed in [42, 123, 7, 999, 2024, 31415, 27182, 16180]:
        inits.append(_random_init_cart(seed))
    
    # If DE found a good solution, add perturbed versions
    if best_points is not None:
        rng = np.random.RandomState(777)
        for scale in [0.05, 0.1, 0.15]:
            pert = best_points + rng.uniform(-scale, scale, size=(N, 2))
            # Project back inside triangle
            for i in range(N):
                px, py = pert[i]
                px = max(0.0, min(px, 1.0))
                py = max(0.0, min(py, H))
                v = py / H
                u = px - v * 0.5
                if u < 0:
                    u = 0
                    px = 0.5 * v
                if v < 0:
                    v = 0
                    py = 0
                if u + v > 1:
                    s = 1.0 / (u + v)
                    u *= s
                    v *= s
                    px = u + 0.5 * v
                    py = v * H
                pert[i] = [px, py]
            inits.append(pert)
    
    # Run Nelder-Mead from each initialization
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 8.0:
            break
        
        x0 = init_pts.flatten()
        x_opt = _nm_cartesian(x0, maxiter=1500)
        opt_pts = x_opt.reshape(N, 2)
        update_best(opt_pts)
        
        # If time remains, do a second NM round with tighter tolerance
        if time.time() < deadline - 5.0 and idx < len(inits) // 2:
            x_opt2 = _nm_cartesian(x_opt, maxiter=1000)
            opt_pts2 = x_opt2.reshape(N, 2)
            update_best(opt_pts2)
    
    # Phase 3: Final refinement on best solution if time remains
    if best_points is not None and time.time() < deadline - 3.0:
        x_best = best_points.flatten()
        x_final = _nm_cartesian(x_best, maxiter=3000)
        final_pts = x_final.reshape(N, 2)
        update_best(final_pts)
    
    if best_points is None:
        # Fallback: simple grid
        bary = np.array([[0,0],[1,0],[0.5,H/H],[0.25,H*0.25/H],[0.75,H*0.25/H],
                         [0.5,H*0.5/H],[0.25,H*0.5/H],[0.75,H*0.5/H],
                         [0.5,H*0.75/H],[0.25,H*0.75/H],[0.75,H*0.75/H]])
        # That's wrong, let me just use random
        rng = np.random.RandomState(0)
        bary = np.array([[rng.uniform(0,1), rng.uniform(0,1)] for _ in range(N)])
        bary = _clamp_bary(bary)
        best_points = _bary_to_cart(bary)
    
    return best_points
# EVOLVE-BLOCK-END
