import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial import ConvexHull
import itertools

N = 14

# Precompute triangle indices for vectorization
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Vectorized for speed.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas vectorized
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
        
    return min_area / hull_area

def neg_score(x):
    """
    Objective function for minimization (we maximize score).
    x is a flat array of length 2*N.
    """
    points = x.reshape((N, 2))
    score = compute_score(points)
    return -score

def neg_soft_score(x, tau=0.01):
    """
    Objective function using soft-min to provide a differentiable landscape.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e9
    except Exception:
        return 1e9
        
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
    
    # Soft min calculation
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def _four_ring_points(params):
    """
    Generate 14 points using 4 rings.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    n1+n2+n3+n4 = 14
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    
    # Ensure integer counts and sum to N
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Simple correction if sum != 14
    total = n1 + n2 + n3 + n4
    if total != N:
        diff = N - total
        # Add/subtract from the last ring for simplicity
        n4 += diff
        if n4 < 0:
            n4 = 0
            n3 += n4 # This logic is fragile, let's use a safer initialization
            # For DE, we should constrain bounds to ensure sum is roughly correct or handle in post
            # Actually, better to just ensure bounds allow valid configs.
            # Let's force n4 to make sum 14
            n4 = N - (n1 + n2 + n3)
            if n4 < 0 or n4 > 14:
                 # Fallback to a valid config if bad
                 return _default_ring_config()
    
    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2 * np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1 * np.cos(angles), r1 * np.sin(angles))))
    if n2 > 0:
        angles = th2 + np.linspace(0, 2 * np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2 * np.cos(angles), r2 * np.sin(angles))))
    if n3 > 0:
        angles = th3 + np.linspace(0, 2 * np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3 * np.cos(angles), r3 * np.sin(angles))))
    if n4 > 0:
        angles = th4 + np.linspace(0, 2 * np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4 * np.cos(angles), r4 * np.sin(angles))))
        
    if not points:
        return _default_ring_config()
        
    result = np.vstack(points)
    
    # Handle edge cases where rounding/adjustment leads to wrong count
    if len(result) != N:
        # Pad or trim
        if len(result) < N:
            extra = np.zeros((N - len(result), 2))
            result = np.vstack([result, extra])
        else:
            result = result[:N]
            
    return result

def _default_ring_config():
    # 7 outer, 7 inner
    a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    a_in = a_out + np.pi / 7
    p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
    p_in = np.column_stack((0.5 * np.cos(a_in), 0.5 * np.sin(a_in)))
    return np.vstack((p_out, p_in))

def _neg_four_ring_score(params):
    """
    Objective for 4-ring parameterization.
    """
    try:
        points = _four_ring_points(params)
        score = compute_score(points)
        return -score
    except Exception:
        return 1e9

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    
    def try_update(points):
        nonlocal best_points, best_score
        if points is None:
            return
        # Ensure points are valid (finite, etc)
        if not np.all(np.isfinite(points)):
            return
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Phase 1: Global Search in 4-Ring Parameter Space
    # Parameters: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    # We will fix the counts to specific valid partitions to avoid the integer optimization complexity
    # Partitions of 14 into 4 parts:
    # 7,4,2,1 | 6,4,3,1 | 5,5,3,1 | 4,4,3,3 | 7,3,2,2 | etc.
    # Let's just use a continuous search on radii and angles for a fixed good partition, 
    # or use DE on the full 12D space with integer rounding.
    
    # To be robust, we will run DE on a specific promising partition: 4,4,3,3
    # And another: 7,4,2,1
    # And 6,4,3,1
    
    partitions = [
        (4, 4, 3, 3),
        (7, 4, 2, 1),
        (6, 4, 3, 1),
        (5, 5, 3, 1)
    ]
    
    for n1, n2, n3, n4 in partitions:
        if time.time() - start_time > deadline - 40:
            break
            
        # Bounds for radii and angles
        # r1 (outer) should be large, r4 (inner) small
        # Angles in [0, 2pi]
        bounds = [
            (0.5, 1.5),  # r1
            (0.2, 1.0),  # r2
            (0.0, 0.8),  # r3
            (0.0, 0.5),  # r4
            (0, 2*np.pi),  # th1
            (0, 2*np.pi),  # th2
            (0, 2*np.pi),  # th3
            (0, 2*np.pi),  # th4
            (n1, n1),      # n1 fixed
            (n2, n2),      # n2 fixed
            (n3, n3),      # n3 fixed
            (n4, n4)       # n4 fixed
        ]
        
        try:
            res_de = differential_evolution(
                _neg_four_ring_score,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=True,
                init='sobol',
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.9
            )
            
            if res_de.fun < 0:
                points_ring = _four_ring_points(res_de.x)
                try_update(points_ring)
        except Exception:
            pass

    # Phase 2: Local Refinement in Full 28D Space
    # Take the best ring-based points and refine in full space
    if best_points is not None:
        x0 = best_points.flatten()
        
        # Step 2a: Soft-min optimization with L-BFGS-B to navigate the non-differentiable min
        # Use a sequence of decreasing tau
        taus = [0.1, 0.05, 0.01, 0.005]
        current_x = x0
        
        for tau in taus:
            if time.time() - start_time > deadline - 15:
                break
            try:
                res_soft = minimize(
                    lambda x: neg_soft_score(x, tau=tau),
                    current_x,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_soft.fun < neg_soft_score(current_x, tau=tau):
                    current_x = res_soft.x
            except Exception:
                pass
        
        # Step 2b: Final Nelder-Mead on exact score
        try:
            res_nm = minimize(
                neg_score,
                current_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            points_nm = res_nm.x.reshape((N, 2))
            try_update(points_nm)
        except Exception:
            pass
        
        # Step 2c: Powell as alternative local optimizer
        if time.time() - start_time < deadline - 10:
            try:
                res_pw = minimize(
                    neg_score,
                    x0, # Start from original ring config again
                    method='Powell',
                    options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                points_pw = res_pw.x.reshape((N, 2))
                try_update(points_pw)
            except Exception:
                pass

    # Phase 3: Multi-start Local Optimization from Diverse Initializations
    # If time permits, try other structured starts
    if time.time() - start_time < deadline - 20:
        rng = np.random.default_rng(123)
        
        # 1. Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # 2. 7+7 Ring (Optimized radii)
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        p_77 = np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
        ))
        
        # 3. Jittered Best Points
        if best_points is not None:
            p_jit = best_points + rng.normal(0, 0.05, (N, 2))
        else:
            p_jit = p_reg + rng.normal(0, 0.1, (N, 2))
            
        starts = [p_reg, p_77, p_jit]
        
        for start in starts:
            if time.time() - start_time > deadline - 5:
                break
            
            x0 = start.flatten()
            try:
                res = minimize(
                    neg_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                points_new = res.x.reshape((N, 2))
                try_update(points_new)
            except Exception:
                pass

    # Fallback: If nothing was found, return a default config
    if best_points is None:
        best_points = _default_ring_config()
        
    return best_points
