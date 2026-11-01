import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices for vectorized area calculation
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
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
    """Raw negative score for minimizers."""
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau):
    """
    Smooth surrogate for the minimum area.
    f(x) = - ( -tau * log(sum(exp(-A_i/tau))) ) / hull_area
    This is differentiable and approximates the raw score as tau -> 0.
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 1e10 # Large penalty for invalid hull
        
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Clip areas to avoid log(0) or exp overflow issues
    areas = np.clip(areas, 1e-12, None)
    
    # Log-sum-exp trick for stability: -tau * log(sum(exp(-A/tau)))
    # Let u = -A/tau. We want -tau * (max(u) + log(sum(exp(u - max(u)))))
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def generate_initial_points(rng):
    """Generate diverse initial points for global search."""
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    candidates.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Two rings 7+7
    a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    a_in = a_out + np.pi/7
    candidates.append(np.vstack((
        np.column_stack((np.cos(a_out), np.sin(a_out))),
        np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
    )))
    
    # 3. Three rings 5+5+4
    a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    candidates.append(np.vstack((
        np.column_stack((np.cos(a1), np.sin(a1))),
        np.column_stack((0.7*np.cos(a2), 0.7*np.sin(a2))),
        np.column_stack((0.4*np.cos(a3), 0.4*np.sin(a3)))
    )))
    
    # 4. Jittered regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 + rng.normal(0, 0.1, N)
    angles += rng.normal(0, 0.1, N)
    candidates.append(np.column_stack((radii*np.cos(angles), radii*np.sin(angles))))
    
    # 5. Random points in unit circle
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2*np.pi, N)
    candidates.append(np.column_stack((r*np.cos(theta), r*np.sin(theta))))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s
    
    # Phase 1: Global Search using Differential Evolution on full coordinates
    # We use a bounded box [-1.5, 1.5] to cover likely optimal regions
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    # Use multiple initial populations from generate_initial_points
    # DE handles the population internally, but we can seed it or run multiple instances
    # Here we run one main DE instance with a larger population and polish=True
    
    if time.time() - start_time < deadline - 60:
        try:
            # Use a custom initialization if possible, or just rely on DE's Latin Hypercube
            # We'll run DE with a reasonable maxiter
            res = differential_evolution(
                neg_score,
                bounds,
                seed=rng.integers(0, 2**31),
                maxiter=50,
                tol=1e-5,
                popsize=20,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False, # We will do our own polish
                init='latinhypercube',
                update_interval=1
            )
            if res.x is not None:
                points = res.x.reshape((N, 2))
                try_update(points)
        except Exception:
            pass
    
    # Phase 2: Local Refinement with Soft-Min Annealing
    # Start from the best point found so far, plus some diverse starts
    
    starts = []
    if best_points is not None:
        starts.append(best_points)
    
    # Add structured starts if time permits
    if time.time() - start_time < deadline - 40:
        for start_pts in generate_initial_points(rng):
            starts.append(start_pts)
            if len(starts) >= 5: # Limit number of starts to save time
                break
    
    # For each start, perform soft-min annealing
    taus = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    for start in starts:
        if time.time() - start_time > deadline - 20:
            break
            
        x0 = start.flatten()
        current_x = x0
        
        for tau in taus:
            if time.time() - start_time > deadline - 15:
                break
            
            try:
                # Use L-BFGS-B for smooth objective
                res = minimize(
                    soft_min_score,
                    current_x,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
                new_p = current_x.reshape((N, 2))
                try_update(new_p)
            except Exception:
                pass
    
    # Phase 3: Final High-Precision Polish on Raw Objective
    if best_points is not None and time.time() - start_time < deadline - 10:
        x0 = best_points.flatten()
        
        # Try Powell first as it handles non-smooth functions better than Nelder-Mead sometimes
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12})
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass
        
        # Try Nelder-Mead as a second opinion
        if time.time() - start_time < deadline - 5:
            try:
                res = minimize(neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                new_p = res.x.reshape((N, 2))
                try_update(new_p)
            except Exception:
                pass
    
    # Ensure we return something valid
    if best_points is None:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
