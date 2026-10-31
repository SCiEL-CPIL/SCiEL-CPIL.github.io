import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

def _min_area_bary_vectorized(points_bary):
    """
    Calculate the minimum area of triangles formed by N points 
    given in barycentric coordinates (u, v) where w = 1 - u - v.
    Uses vectorization for speed.
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    # Create indices for all unique triples (i, j, k) with i < j < k
    # We can use np.triu_indices or just generate them
    # For N=11, the number of triples is 11*10*9/6 = 165. This is small enough to handle explicitly in a loop if needed, 
    # but vectorization over the triples is better.
    
    # Generate all triples
    idx_i = np.repeat(np.arange(n), n*(n-1)//2) # This is wrong for full triples, let's just use a simple loop for clarity and correctness first, 
    # actually for N=11, 165 triangles is tiny. A pure python loop over 165 items is fast.
    # But let's stick to a semi-vectorized approach for the inner product if possible, or just standard loops.
    # Standard loops are often faster for very small N due to overhead of array creation.
    
    min_a = float('inf')
    
    # Precompute w for speed
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Determinant of 3x3 matrix
                # | u_i v_i w_i |
                # | u_j v_j w_j |
                # | u_k v_k w_k |
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                       - v[i] * (u[j] * w[k] - w[j] * u[k]))
                # Wait, w[k] is 1-u[k]-v[k]
                # Let's just compute w values explicitly in the loop to avoid repeated 1-u-v
                # Actually, w is precomputed above.
                # Correct formula:
                # det = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
                
                # Let's use the precomputed w
                d = (u[i] * (v[j] * w[k] - w[j] * v[k]) 
                     - v[i] * (u[j] * w[k] - w[j] * u[k]) 
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                
                a = abs(d)
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_area_bary(points_bary, beta=20.0):
    """
    Soft minimum area for smoother optimization landscape.
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                       - v[i] * (u[j] * w[k] - w[j] * u[k])
                       + w[i] * (u[j] * v[k] - v[j] * u[k]))
                areas.append(abs(det))
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
        
    # Soft min: -1/beta * log(sum(exp(-beta * a)))
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    log_sum_exp = max_a + np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = -log_sum_exp / beta
    
    return soft_min

def _penalty(points_bary):
    """
    Penalty for points outside the valid simplex region.
    """
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    penalty = 0.0
    # u, v >= 0 is handled by bounds in optimizer, but w >= 0 is u+v <= 1
    if np.any(w < 0):
        penalty += np.sum(w[w < 0]**2) * 100
    if np.any(u < 0):
        penalty += np.sum(u[u < 0]**2) * 100
    if np.any(v < 0):
        penalty += np.sum(v[v < 0]**2) * 100
    return penalty

def _objective_soft(params, beta=20.0):
    """
    Objective for global search (DE). Minimizes soft min area + penalty.
    """
    uv = params.reshape(N, 2)
    soft_a = _soft_min_area_bary(uv, beta=beta)
    pen = _penalty(uv)
    return soft_a + pen

def _objective_hard(params):
    """
    Objective for local refinement (L-BFGS-B). Minimizes hard min area + penalty.
    """
    uv = params.reshape(N, 2)
    min_a = _min_area_bary_vectorized(uv)
    pen = _penalty(uv)
    return -min_a + pen # We want to maximize min_a, so minimize -min_a

def _clamp_to_triangle(bary):
    """
    Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1.
    """
    bary = bary.copy()
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Clip u, v to [0, 1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    # Ensure u + v <= 1
    sum_uv = u + v
    mask = sum_uv > 1
    if np.any(mask):
        factor = 1.0 / sum_uv[mask]
        u[mask] *= factor
        v[mask] *= factor
        
    bary[:, 0] = u
    bary[:, 1] = v
    return bary

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates to Cartesian coordinates in the unit area equilateral triangle.
    Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    """
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    pts = u[:, np.newaxis] * A + v[:, np.newaxis] * B + w[:, np.newaxis] * C
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = -float('inf')
    best_points = None
    
    # Define bounds for DE: u in [0, 1], v in [0, 1]
    # Note: DE will explore the square [0,1]x[0,1], but penalty will push it to the triangle.
    # It's better to define bounds that respect the triangle if possible, but DE doesn't support linear constraints easily.
    # So we use [0,1] for both and rely on penalty.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Initial guess for DE: Center of mass
    # A good symmetric starting point:
    # 3 corners, 3 edge midpoints, 3 points slightly inside, 1 center?
    # Or just random.
    
    # Strategy:
    # 1. Run Differential Evolution to find a good global candidate.
    # 2. Refine with L-BFGS-B.
    # 3. Repeat if time allows.
    
    rng = np.random.RandomState(42)
    
    # We will try a few different DE seeds/strategies
    # DE is expensive, so we limit maxiter.
    
    # Let's try one main DE run with reasonable parameters.
    # popsize=15, maxiter=100
    # Tolerance: 1e-6
    
    try:
        # Use a soft minimum for DE to make it smoother
        # Beta=20 is a good balance
        de_result = differential_evolution(
            _objective_soft,
            bounds=bounds,
            args=(20.0,),
            maxiter=100,
            popsize=20,
            tol=1e-5,
            seed=42,
            polish=False, # We will do our own polish
            update_interval=10,
            callback=None
        )
        
        # Check time
        if time.time() > deadline - 10:
            return _bary_to_cart(de_result.x.reshape(N, 2))
            
        # de_result.x is the best point found
        uv_de = de_result.x.reshape(N, 2)
        uv_de = _clamp_to_triangle(uv_de)
        
        # Refine with L-BFGS-B using hard minimum
        try:
            res_lbgfgs = minimize(
                _objective_hard,
                uv_de.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            uv_opt = res_lbgfgs.x.reshape(N, 2)
            uv_opt = _clamp_to_triangle(uv_opt)
            
            # Evaluate final score
            min_a = _min_area_bary_vectorized(uv_opt)
            if min_a > best_score:
                best_score = min_a
                best_points = uv_opt.copy()
                
        except Exception:
            # If L-BFGS fails, use DE result
            min_a = _min_area_bary_vectorized(uv_de)
            if min_a > best_score:
                best_score = min_a
                best_points = uv_de.copy()

    except Exception:
        # Fallback to random search if DE fails completely
        pass

    # If we have time, do a few random restarts + L-BFGS to see if we beat the DE result
    # DE is usually good, but random restarts can sometimes find better local minima if DE gets stuck.
    
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Do 5 quick random restarts
        for i in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            uv_rand = rng.uniform(0, 1, size=(N, 2))
            uv_rand = _clamp_to_triangle(uv_rand)
            
            try:
                res_lbgfgs = minimize(
                    _objective_hard,
                    uv_rand.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                uv_opt = res_lbgfgs.x.reshape(N, 2)
                uv_opt = _clamp_to_triangle(uv_opt)
                
                min_a = _min_area_bary_vectorized(uv_opt)
                if min_a > best_score:
                    best_score = min_a
                    best_points = uv_opt.copy()
                    
            except Exception:
                pass

    # Fallback if no valid points found
    if best_points is None:
        # Simple grid fallback
        # 11 points: 3 corners, 3 edge midpoints, 3 inner, 1 center?
        # Let's just use a random valid configuration
        best_points = rng.uniform(0, 1, size=(N, 2))
        best_points = _clamp_to_triangle(best_points)

    # Convert to Cartesian
    pts = _bary_to_cart(best_points)
    
    return pts
