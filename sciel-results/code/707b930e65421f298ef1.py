import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

# Vertices in Cartesian: (0,0), (1,0), (0.5, H)
# Barycentric to Cartesian:
# P = u*V1 + v*V2 + w*V3
# V1=(0,0), V2=(1,0), V3=(0.5, H)
# x = u*0 + v*1 + w*0.5 = v + 0.5*w
# y = u*0 + v*0 + w*H = H*w
# Given u,v,w >= 0, u+v+w=1
# We can represent a point by (u, v) where w = 1 - u - v
# Constraints: u >= 0, v >= 0, u + v <= 1

def bary_to_cart(bary):
    """
    bary: (N, 2) array of (u, v). w is implicit as 1-u-v.
    Returns (N, 2) Cartesian coordinates.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return np.column_stack((x, y))

def cart_to_bary(cart):
    """
    cart: (N, 2) array of (x, y).
    Returns (N, 2) array of (u, v).
    """
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack((u, v))

def _min_area_bary(bary_pts):
    """
    Calculate min triangle area given points in barycentric coords (u, v).
    Converts to Cartesian internally for area calculation using vectorized ops.
    """
    pts = bary_to_cart(bary_pts)
    Pi = pts[I]
    Pj = pts[J]
    Pk = pts[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_area_bary(bary_pts, beta=100.0):
    """
    Smooth approximation of min area.
    """
    pts = bary_to_cart(bary_pts)
    Pi = pts[I]
    Pj = pts[J]
    Pk = pts[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp approximation of min
    min_approx = -np.log(np.sum(np.exp(-beta * areas))) / beta
    
    return min_approx

def _project_bary(bary):
    """
    Project a point (u, v) into the valid barycentric simplex (u>=0, v>=0, u+v<=1).
    Simple clipping and normalization might not be perfect, but for local search
    we can use a simple projection:
    1. Clip u, v to [0, 1].
    2. If u+v > 1, scale u, v such that u+v=1.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    return np.column_stack((u, v))

def _generate_hex_init():
    """
    Generate a hexagonal/triangular lattice initialization in barycentric coordinates.
    """
    # We want 11 points.
    # A common good start is a small triangular grid.
    # Let's place points on a grid in the (u,v) plane.
    # The domain is a triangle in (u,v) space: (0,0), (1,0), (0,1).
    
    # 4 rows:
    # Row 0 (bottom, v=0): 4 points at u = 0.1, 0.3, 0.5, 0.7
    # Row 1 (v=0.25): 3 points at u = 0.15, 0.35, 0.55
    # Row 2 (v=0.5): 2 points at u = 0.2, 0.4
    # Row 3 (v=0.75): 1 point at u = 0.25
    
    # Wait, the triangle in (u,v) is u>=0, v>=0, u+v<=1.
    # Let's use a standard grid.
    
    pts = []
    # Row 0: v=0.05
    for u in [0.1, 0.3, 0.5, 0.7]:
        if u + 0.05 <= 1.0:
            pts.append([u, 0.05])
            
    # Row 1: v=0.25
    for u in [0.15, 0.35, 0.55]:
        if u + 0.25 <= 1.0:
            pts.append([u, 0.25])
            
    # Row 2: v=0.45
    for u in [0.2, 0.4]:
        if u + 0.45 <= 1.0:
            pts.append([u, 0.45])
            
    # Row 3: v=0.65
    for u in [0.25]:
        if u + 0.65 <= 1.0:
            pts.append([u, 0.65])
            
    # We need 11 points. We have 4+3+2+1 = 10.
    # Add one more in the center?
    pts.append([0.33, 0.33])
    
    pts = np.array(pts)
    return _project_bary(pts)

def _generate_random_init(rng):
    """
    Generate random valid barycentric points.
    """
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            # Scale back
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        pts.append([u, v])
    return np.array(pts)

def _simulated_annealing_bary(initial_bary, time_limit, rng):
    """
    Simulated Annealing in barycentric coordinates.
    """
    current_bary = initial_bary.copy()
    current_score = _min_area_bary(current_bary)
    
    best_bary = current_bary.copy()
    best_score = current_score
    
    T = 0.05
    cooling_rate = 0.9995
    min_T = 1e-8
    step_scale = 0.1
    
    start_time = time.time()
    no_improve_count = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move: pick a point, add Gaussian noise in (u,v) space
        idx = rng.integers(N)
        noise = rng.normal(0, step_scale, size=2)
        
        new_bary = current_bary.copy()
        new_bary[idx] += noise
        
        # Project to valid simplex
        new_bary = _project_bary(new_bary)
        
        new_score = _min_area_bary(new_bary)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
            current_bary = new_bary
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > 100:
            step_scale = max(0.005, step_scale * 0.9)
            no_improve_count = 0
        elif no_improve_count == 0:
            step_scale = min(0.2, step_scale * 1.02)
            
        T *= cooling_rate
        
        # Occasional large jump
        if rng.random() < 0.005:
            idx = rng.integers(N)
            new_bary = current_bary.copy()
            # Random valid point
            u = rng.random()
            v = rng.random()
            if u + v > 1.0:
                scale = 1.0 / (u + v)
                u *= scale
                v *= scale
            new_bary[idx] = [u, v]
            new_bary = _project_bary(new_bary)
            
            new_score = _min_area_bary(new_bary)
            delta = new_score - current_score
            if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                current_bary = new_bary
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_bary = current_bary.copy()
    
    return best_bary, best_score

def _nlp_refine_bary(bary_pts, time_limit, rng):
    """
    Refine using SLSQP/L-BFGS-B in barycentric coordinates.
    """
    best_bary = bary_pts.copy()
    best_score = _min_area_bary(best_bary)
    
    start_time = time.time()
    
    # Objective to minimize: -soft_min_area
    def obj(flat_bary):
        b = flat_bary.reshape(-1, 2)
        # Ensure validity? SLSQP bounds handle u>=0, v>=0.
        # We need u+v <= 1. We can add a constraint or just rely on penalty if we use soft min.
        # SLSQP supports constraints.
        # Let's use a penalty in the objective for u+v > 1 to be safe with L-BFGS-B later.
        # Or just use SLSQP with constraints.
        u = b[:, 0]
        v = b[:, 1]
        viol = np.sum(np.maximum(0, u + v - 1.0)**2)
        area = _soft_min_area_bary(b, beta=200.0)
        return -area + 100.0 * viol
    
    bounds = [(0, 1)] * (2 * N)
    
    # SLSQP
    try:
        res = minimize(
            obj,
            best_bary.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        if res.success:
            new_bary = res.x.reshape(-1, 2)
            new_bary = _project_bary(new_bary)
            new_score = _min_area_bary(new_bary)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary.copy()
    except Exception:
        pass
    
    # L-BFGS-B
    if time.time() - start_time < time_limit - 5:
        try:
            res2 = minimize(
                obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'gtol': 1e-12}
            )
            new_bary2 = res2.x.reshape(-1, 2)
            new_bary2 = _project_bary(new_bary2)
            new_score2 = _min_area_bary(new_bary2)
            if new_score2 > best_score:
                best_score = new_score2
                best_bary = new_bary2.copy()
        except Exception:
            pass
            
    return best_bary, best_score

def _differential_evolution_global(rng, time_limit):
    """
    Use differential evolution for global search.
    """
    from scipy.optimize import differential_evolution
    
    def de_obj(x):
        bary = x.reshape(N, 2)
        bary = _project_bary(bary)
        return -_min_area_bary(bary)
    
    bounds = [(0, 1)] * (2 * N)
    
    try:
        res = differential_evolution(
            de_obj,
            bounds,
            seed=rng.integers(10000),
            maxiter=30,
            popsize=10,
            tol=1e-6,
            polish=False
        )
        de_bary = _project_bary(res.x.reshape(N, 2))
        return de_bary, -de_obj(res.x)
    except Exception:
        return None, -np.inf

def _refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using two stages:
    1. L-BFGS-B on the smooth soft-min surrogate.
    2. Nelder-Mead on the hard minimum.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            lambda x: -_soft_min_area_bary(x.reshape(N, 2), beta=80.0) + 100.0 * np.sum(np.maximum(0, x.reshape(N, 2)[:, 0] + x.reshape(N, 2)[:, 1] - 1.0)**2),
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
    except Exception:
        return bary_init
    
    if time.time() - start_time > time_limit:
        return bary_init
        
    bary_mid = _project_bary(result1.x.reshape(N, 2))
    
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            lambda x: -_min_area_bary(x.reshape(N, 2)),
            bary_mid.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-16}
        )
    except Exception:
        return bary_mid
        
    return _project_bary(result2.x.reshape(N, 2))

def construct():
    start_time = time.time()
    timeout = 145.0
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -np.inf
    
    # Phase 1: Multi-start SA
    num_starts = 5
    time_per_start = (timeout - 40) / num_starts
    
    init_strategies = [
        _generate_hex_init,
        lambda: _generate_random_init(rng),
        lambda: _generate_random_init(rng),
        lambda: _generate_random_init(rng),
        lambda: _generate_random_init(rng),
    ]
    
    for i in range(num_starts):
        if time.time() - start_time > timeout - 40:
            break
            
        init_pts = init_strategies[i]()
        sa_pts, sa_score = _simulated_annealing_bary(init_pts, time_per_start, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_bary = sa_pts
    
    # Phase 2: Differential Evolution Global Search
    if time.time() - start_time < timeout - 30:
        de_bary, de_score = _differential_evolution_global(rng, timeout - 30 - (time.time() - start_time))
        if de_bary is not None and de_score > best_score:
            best_score = de_score
            best_bary = de_bary
    
    # Phase 3: NLP Refinement
    if best_bary is not None and time.time() - start_time < timeout - 15:
        nlp_time = timeout - 15 - (time.time() - start_time)
        nlp_pts, nlp_score = _nlp_refine_bary(best_bary, nlp_time, rng)
        
        if nlp_score > best_score:
            best_score = nlp_score
            best_bary = nlp_pts
    
    # Phase 4: Multi-start local refinement with random and perturbed starts
    if best_bary is not None:
        restart_count = 0
        while time.time() - start_time < timeout - 10:
            restart_count += 1
            if restart_count % 3 == 0:
                # Perturbation of best solution
                sigma = 0.06
                noise = rng.normal(0, sigma, best_bary.shape)
                init = _project_bary(best_bary + noise)
            else:
                # Fresh random start
                init = _generate_random_init(rng)
            
            refined = _refine_local(init, time_limit=min(4.0, timeout - 10 - (time.time() - start_time)))
            refined_score = _min_area_bary(refined)
            
            if refined_score > best_score:
                best_score = refined_score
                best_bary = refined
    
    # Final Conversion and Validation
    if best_bary is None:
        best_bary = _generate_hex_init()
        
    # Convert to Cartesian
    cart_pts = bary_to_cart(best_bary)
    
    # Double check validity (just in case)
    x = cart_pts[:, 0]
    y = cart_pts[:, 1]
    
    # Clip to triangle
    y = np.maximum(y, 0)
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
        
    cart_pts[:, 0] = x
    cart_pts[:, 1] = y
    
    return cart_pts
