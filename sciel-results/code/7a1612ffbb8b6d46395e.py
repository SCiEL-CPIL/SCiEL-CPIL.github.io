import numpy as np
import scipy.spatial as spat
import time

N = 14

def calc_score_fast(points):
    """
    Fast calculation of min triangle area / hull area.
    Uses vectorization for triangle area calculation.
    """
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    n = len(points)
    # Generate all combinations of 3 indices
    # For N=14, C(14,3) = 364. 
    # Vectorizing this is much faster than a Python loop.
    
    # We can compute areas for all triples at once if we structure the arrays right,
    # but for N=14, a simple loop over i, and vectorized j,k might be overkill 
    # or we can just do the full vectorized cross product.
    
    # Let's do the fully vectorized approach for maximum speed.
    # Indices for i, j, k
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Number of triangles
    M = len(i_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Vectors
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product z-component (2D cross product is scalar)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points.copy()
    best_score = calc_score_fast(points)
    
    # Simulated Annealing / Stochastic Local Search
    # We perturb points and accept if better, or occasionally if worse (to escape local minima)
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Parameters
    T = 1.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    max_steps = 20000
    
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Generate perturbation
        # Gaussian noise scaled by current temperature
        noise = rng.normal(0, T, size=(N, 2))
        new_points = current_points + noise
        
        new_score = calc_score_fast(new_points)
        
        if new_score > current_score:
            # Accept
            current_points = new_points
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Since we are maximizing, delta = new - current < 0
            # Prob = exp(-|delta|/T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(-abs(delta) / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
            
        # Cool down
        T *= cooling_rate
        if T < min_T:
            # Reheat or stop? Let's just break if we've cooled down enough
            # and haven't found a better solution in a while.
            # For simplicity, we just continue with min_T or break.
            # Breaking is safer for time budget.
            pass
            
    return best_points
