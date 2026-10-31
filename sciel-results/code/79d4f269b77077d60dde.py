import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    if n < 3:
        return 0.0
        
    # Vectorized calculation for all triples
    # For N=13, number of triples is 286.
    # We can use combinations for clarity and efficiency if N is small,
    # but meshgrid is faster for vectorization in NumPy.
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (i < j) & (j < k)
    i_idx = i[mask]
    j_idx = j[mask]
    k_idx = k[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def compute_hull_area(points):
    try:
        # Check for degenerate cases
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def objective_function(x):
    """
    Objective function for differential_evolution.
    We want to MAXIMIZE the minimum triangle area normalized by hull area.
    differential_evolution MINIMIZES, so we return the negative of the score.
    """
    points = x.reshape(N, 2)
    
    # Compute hull area
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0 # Invalid configuration, return 0 (which is -0 score)
    
    # Normalize to unit area
    scale = 1.0 / np.sqrt(hull_area)
    pts_scaled = points * scale
    
    # Compute min triangle area
    min_tri_area = compute_min_triangle_area(pts_scaled)
    
    return -min_tri_area

def construct():
    start_time = time.time()
    time_limit = 115.0 # 5s buffer
    
    # Define bounds for each coordinate
    # A regular polygon of radius 1 has area ~1.17.
    # To allow flexibility, we bound coordinates in [-10, 10].
    bounds = [(-10, 10)] * (2 * N)
    
    # Configure Differential Evolution
    # maxiter: number of generations. 
    # popsize: size of the population.
    # tol: relative tolerance for termination.
    # seed: for reproducibility
    
    # We need to be careful with time. 
    # DE can be slow. We will monitor time in a callback if possible, 
    # or just set limits that fit within 120s.
    # A typical DE run with popsize=20, maxiter=100 for 26 dims might take a few seconds to minutes.
    # Let's try a moderate setting.
    
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True, # Use L-BFGS-B at the end to refine
            init='sobol', # Latin hypercube sampling for better initial spread
            updating='deferred' # Faster updates
        )
        
        if result.success or result.fun < 0: # If it found a valid solution
            best_points = result.x.reshape(N, 2)
        else:
            # Fallback to regular polygon if DE fails completely
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_points = np.column_stack([np.cos(angles), np.sin(angles)])
            
    except Exception as e:
        # Fallback to regular polygon if DE throws an error
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Check time, if we are close to limit, just return what we have.
    # DE usually returns the best found so far.
    
    return best_points
