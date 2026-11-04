import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14

# Pre-compute indices for all triangles i < j < k
# There are 364 triangles for N=14
TRI_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area."""
    if len(points) < 3:
        return 0.0
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _initial_points():
    """
    Generate initial points: Regular 14-gon with small jitter.
    This provides a symmetric starting point that is not degenerate.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add small random jitter to break perfect symmetry and allow SA to move
    jitter = np.random.normal(0, 0.05, (N, 2))
    points = np.column_stack([np.cos(angles), np.sin(angles)]) + jitter
    # Normalize to unit area roughly? No, score is invariant to scaling, 
    # but let's keep it in a reasonable range [0,1] or [-1,1]
    # The score function handles scaling via hull area.
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial state
    current_points = _initial_points()
    current_score = _score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature
    alpha = 0.9995  # Cooling rate
    min_T = 1e-6
    
    # We will run until time limit or temperature is too low
    step_count = 0
    
    while time.time() < deadline and T > min_T:
        # Generate a new candidate
        # Strategy: Random walk in 2D
        # Step size is proportional to temperature
        step_size = 0.05 * (T / 0.05) + 0.001
        noise = np.random.normal(0, step_size, (N, 2))
        
        # Occasionally, do a larger "jump" or swap to escape deep local minima
        if np.random.rand() < 0.05:
            # Swap two random points
            i, j = np.random.choice(N, 2, replace=False)
            noise[i] += 0.1 * np.random.normal(0, 1, 2)
            noise[j] -= 0.1 * np.random.normal(0, 1, 2)
            
        candidate_points = current_points + noise
        
        # Ensure points stay within a reasonable bounding box to avoid numerical issues
        # The convex hull area will shrink if points go far, but let's keep them compact
        # Actually, the score is scale invariant, so we don't strictly need to bound,
        # but large coordinates can cause precision issues in cross products.
        # Let's normalize the candidate to have unit area or just keep it raw?
        # Raw is fine as long as we don't overflow.
        
        candidate_score = _score(candidate_points)
        
        # Acceptance criterion
        if candidate_score > current_score:
            current_points = candidate_points
            current_score = candidate_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Metropolis criterion: accept with probability exp((delta)/T)
            delta = candidate_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_points = candidate_points
                    current_score = candidate_score
        
        # Cool down
        T *= alpha
        step_count += 1
        
        # Time check inside loop to be safe
        if time.time() > deadline - 2.0:
            break
            
    # Final local polish with Nelder-Mead on the best found configuration
    # This helps refine the solution after the coarse SA search
    if time.time() < deadline - 5.0:
        from scipy.optimize import minimize
        
        def obj(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
            
        try:
            res = minimize(
                obj, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() < deadline:
                res_score = -res.fun
                if res_score > best_score:
                    best_score = res_score
                    best_points = res.x.reshape(-1, 2).copy()
        except Exception:
            pass

    return best_points
