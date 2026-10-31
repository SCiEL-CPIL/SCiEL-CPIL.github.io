import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi
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

def _compute_voronoi_stats(cart_pts):
    """
    Compute Voronoi cell statistics: circumradii and areas of each cell.
    Returns (circumradii, cell_areas) arrays of length N.
    """
    try:
        vor = Voronoi(cart_pts)
    except Exception:
        # Fallback: return uniform values if Voronoi fails
        return np.ones(N) * 0.1, np.ones(N) * (1.0 / N)
    
    circumradii = np.zeros(N)
    cell_areas = np.zeros(N)
    
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded cell - estimate using distance to nearest other point
            dists = np.linalg.norm(cart_pts - cart_pts[i], axis=1)
            dists[i] = np.inf
            circumradii[i] = 0.5 * np.min(dists)
            cell_areas[i] = 1.0 / N
            continue
        
        vertices = vor.vertices[region]
        
        # Compute circumradius: max distance from point i to any vertex
        dists = np.linalg.norm(vertices - cart_pts[i], axis=1)
        circumradii[i] = np.max(dists) if len(dists) > 0 else 0.0
        
        # Compute area using shoelace formula
        if len(vertices) >= 3:
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            cell_areas[i] = area
        else:
            cell_areas[i] = 1.0 / N
    
    return circumradii, cell_areas

def _voronoi_surrogate(cart_pts, beta_circ=50.0, weight_var=0.5):
    """
    Smooth differentiable surrogate based on Voronoi cell statistics.
    Minimizes a combination of:
    1. Soft max of circumradii (proxy for worst-case spacing)
    2. Variance of cell areas (proxy for uniformity)
    
    Lower surrogate value = better configuration.
    """
    circumradii, cell_areas = _compute_voronoi_stats(cart_pts)
    
    # Soft max of circumradii (approximate max)
    max_circ = np.max(circumradii)
    # Use a smooth approximation: log-sum-exp
    scaled = -beta_circ * circumradii
    max_scaled = np.max(scaled)
    soft_max = -(max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta_circ
    
    # Variance of cell areas
    var_areas = np.var(cell_areas)
    
    # Combined objective (minimize this)
    return soft_max + weight_var * var_areas

def _voronoi_surrogate_grad(cart_pts, beta_circ=50.0, weight_var=0.5, eps=1e-6):
    """
    Numerical gradient of the Voronoi surrogate.
    """
    grad = np.zeros_like(cart_pts)
    base_val = _voronoi_surrogate(cart_pts, beta_circ, weight_var)
    
    for i in range(N):
        for j in range(2):
            cart_plus = cart_pts.copy()
            cart_plus[i, j] += eps
            val_plus = _voronoi_surrogate(cart_plus, beta_circ, weight_var)
            grad[i, j] = (val_plus - base_val) / eps
    
    return grad

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

def _voronoi_sa_bary(initial_bary, time_limit, rng):
    """
    Simulated Annealing using Voronoi-based surrogate in barycentric coordinates.
    The surrogate provides smoother landscape than hard min-area.
    """
    current_bary = initial_bary.copy()
    current_cart = bary_to_cart(current_bary)
    current_surrogate = _voronoi_surrogate(current_cart)
    current_score = _min_area_bary(current_bary)
    
    best_bary = current_bary.copy()
    best_score = current_score
    
    T = 0.02
    cooling_rate = 0.999
    min_T = 1e-8
    step_scale = 0.08
    
    start_time = time.time()
    no_improve_count = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move
        idx = rng.integers(N)
        noise = rng.normal(0, step_scale, size=2)
        
        new_bary = current_bary.copy()
        new_bary[idx] += noise
        new_bary = _project_bary(new_bary)
        
        new_cart = bary_to_cart(new_bary)
        new_surrogate = _voronoi_surrogate(new_cart)
        
        # Accept if surrogate improves, or with probability based on temperature
        delta_surr = new_surrogate - current_surrogate
        
        if delta_surr < 0 or (T > min_T and rng.random() < np.exp(-delta_surr / T)):
            current_bary = new_bary
            current_cart = new_cart
            current_surrogate = new_surrogate
            
            new_score = _min_area_bary(new_bary)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > 80:
            step_scale = max(0.003, step_scale * 0.92)
            no_improve_count = 0
        elif no_improve_count == 0:
            step_scale = min(0.15, step_scale * 1.03)
            
        T *= cooling_rate
        
        # Occasional large jump for exploration
        if rng.random() < 0.003:
            idx = rng.integers(N)
            u = rng.random()
            v = rng.random()
            if u + v > 1.0:
                scale = 1.0 / (u + v)
                u *= scale
                v *= scale
            new_bary = current_bary.copy()
            new_bary[idx] = [u, v]
            new_bary = _project_bary(new_bary)
            
            new_cart = bary_to_cart(new_bary)
            new_surrogate = _voronoi_surrogate(new_cart)
            delta_surr = new_surrogate - current_surrogate
            
            if delta_surr < 0 or (T > min_T and rng.random() < np.exp(-delta_surr / T)):
                current_bary = new_bary
                current_cart = new_cart
                current_surrogate = new_surrogate
                new_score = _min_area_bary(new_bary)
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary.copy()
    
    return best_bary, best_score

def _nlp_refine_bary(bary_pts, time_limit, rng):
    """
    Refine using L-BFGS-B on the Voronoi surrogate in Cartesian coordinates,
    then polish with Nelder-Mead on hard min-area.
    """
    best_bary = bary_pts.copy()
    best_score = _min_area_bary(best_bary)
    start_time = time.time()
    
    # Stage 1: Optimize Voronoi surrogate in Cartesian space
    def cart_obj(flat_cart):
        cart = flat_cart.reshape(N, 2)
        # Penalty for points outside triangle
        x = cart[:, 0]
        y = cart[:, 1]
        # Triangle constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
        viol = np.sum(np.maximum(0, -y)**2)
        viol += np.sum(np.maximum(0, y - S3 * x)**2)
        viol += np.sum(np.maximum(0, y - S3 * (1 - x))**2)
        surr = _voronoi_surrogate(cart)
        return surr + 50.0 * viol
    
    cart_init = bary_to_cart(best_bary)
    
    try:
        res1 = minimize(
            cart_obj,
            cart_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_cart = res1.x.reshape(N, 2)
        new_bary = _project_bary(cart_to_bary(new_cart))
        new_score = _min_area_bary(new_bary)
        if new_score > best_score:
            best_score = new_score
            best_bary = new_bary.copy()
    except Exception:
        pass
    
    # Stage 2: Polish with Nelder-Mead on hard min-area
    if time.time() - start_time < time_limit - 3:
        try:
            def hard_obj(flat_bary):
                b = flat_bary.reshape(N, 2)
                b = _project_bary(b)
                return -_min_area_bary(b)
            
            res2 = minimize(
                hard_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            new_bary2 = _project_bary(res2.x.reshape(N, 2))
            new_score2 = _min_area_bary(new_bary2)
            if new_score2 > best_score:
                best_score = new_score2
                best_bary = new_bary2.copy()
        except Exception:
            pass
    
    return best_bary, best_score

def _voronoi_global_search(rng, time_limit):
    """
    Global search using differential evolution on the Voronoi surrogate.
    """
    from scipy.optimize import differential_evolution
    
    def de_obj(flat_cart):
        cart = flat_cart.reshape(N, 2)
        # Penalty for points outside triangle
        x = cart[:, 0]
        y = cart[:, 1]
        viol = np.sum(np.maximum(0, -y)**2)
        viol += np.sum(np.maximum(0, y - S3 * x)**2)
        viol += np.sum(np.maximum(0, y - S3 * (1 - x))**2)
        surr = _voronoi_surrogate(cart)
        return surr + 50.0 * viol
    
    # Bounds in Cartesian: x in [0,1], y in [0, H]
    bounds = [(0, 1)] * N + [(0, H)] * N
    
    try:
        res = differential_evolution(
            de_obj,
            bounds,
            seed=rng.integers(10000),
            maxiter=40,
            popsize=12,
            tol=1e-6,
            polish=False
        )
        de_cart = res.x.reshape(N, 2)
        de_bary = _project_bary(cart_to_bary(de_cart))
        de_score = _min_area_bary(de_bary)
        return de_bary, de_score
    except Exception:
        return None, -np.inf

def _refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using:
    1. L-BFGS-B on Voronoi surrogate in Cartesian.
    2. Nelder-Mead on hard min-area in barycentric.
    """
    start_time = time.time()
    
    # Stage 1: Voronoi surrogate in Cartesian
    def cart_obj(flat_cart):
        cart = flat_cart.reshape(N, 2)
        x = cart[:, 0]
        y = cart[:, 1]
        viol = np.sum(np.maximum(0, -y)**2)
        viol += np.sum(np.maximum(0, y - S3 * x)**2)
        viol += np.sum(np.maximum(0, y - S3 * (1 - x))**2)
        surr = _voronoi_surrogate(cart)
        return surr + 50.0 * viol
    
    cart_init = bary_to_cart(bary_init)
    
    try:
        result1 = minimize(
            cart_obj,
            cart_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
    except Exception:
        return bary_init
    
    if time.time() - start_time > time_limit:
        return bary_init
    
    bary_mid = _project_bary(cart_to_bary(result1.x.reshape(N, 2)))
    
    # Stage 2: Hard min-area polish
    try:
        def hard_obj(flat_bary):
            b = flat_bary.reshape(N, 2)
            b = _project_bary(b)
            return -_min_area_bary(b)
        
        result2 = minimize(
            hard_obj,
            bary_mid.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
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
    
    # Phase 1: Multi-start Voronoi-guided SA
    num_starts = 4
    time_per_start = (timeout - 50) / num_starts
    
    init_strategies = [
        _generate_hex_init,
        lambda: _generate_random_init(rng),
        lambda: _generate_random_init(rng),
        lambda: _generate_random_init(rng),
    ]
    
    for i in range(num_starts):
        if time.time() - start_time > timeout - 50:
            break
            
        init_pts = init_strategies[i]()
        sa_pts, sa_score = _voronoi_sa_bary(init_pts, time_per_start, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_bary = sa_pts
    
    # Phase 2: Voronoi-guided global search (DE on surrogate)
    if time.time() - start_time < timeout - 35:
        de_time = timeout - 35 - (time.time() - start_time)
        de_bary, de_score = _voronoi_global_search(rng, de_time)
        if de_bary is not None and de_score > best_score:
            best_score = de_score
            best_bary = de_bary
    
    # Phase 3: NLP Refinement (Voronoi surrogate + hard polish)
    if best_bary is not None and time.time() - start_time < timeout - 15:
        nlp_time = timeout - 15 - (time.time() - start_time)
        nlp_pts, nlp_score = _nlp_refine_bary(best_bary, nlp_time, rng)
        
        if nlp_score > best_score:
            best_score = nlp_score
            best_bary = nlp_pts
    
    # Phase 4: Multi-start local refinement
    if best_bary is not None:
        restart_count = 0
        while time.time() - start_time < timeout - 10:
            restart_count += 1
            if restart_count % 3 == 0:
                # Perturbation of best solution
                sigma = 0.05
                noise = rng.normal(0, sigma, best_bary.shape)
                init = _project_bary(best_bary + noise)
            else:
                # Fresh random start
                init = _generate_random_init(rng)
            
            remaining = timeout - 10 - (time.time() - start_time)
            if remaining < 2:
                break
            refined = _refine_local(init, time_limit=min(4.0, remaining))
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
