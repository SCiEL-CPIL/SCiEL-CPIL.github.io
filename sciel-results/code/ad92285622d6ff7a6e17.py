"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill

def _min_triangle_area_vec(pts):
    """
    Compute the minimum triangle area for a set of points.
    Vectorized for speed.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    min_area = float('inf')
    
    # Iterate over the first point of the triangle
    for i in range(n - 2):
        # Indices for the other two points
        rem_idx = np.arange(i + 1, n)
        m = len(rem_idx)
        if m < 2:
            continue
        
        # Get coordinates of remaining points
        x_rem = x[rem_idx]
        y_rem = y[rem_idx]
        
        # We need areas of triangles (i, p, q) where p, q are distinct indices in rem_idx.
        # Area = 0.5 * | cross( P_p - P_i, P_q - P_i ) |
        
        # Vectorize over all pairs (p, q) in rem_idx with p < q
        # To avoid creating huge meshgrids for every i (though N=14 is small, let's be efficient)
        # We can use broadcasting.
        # p_indices: [0, 1, ..., m-1]
        # q_indices: [0, 1, ..., m-1]
        # We want p < q.
        
        # Let's just compute all pairs and mask, or use triangular indices.
        # For N=14, max m=13. Pairs = 78. This is tiny.
        p_idx, q_idx = np.triu_indices(m, k=1)
        
        px = x_rem[p_idx]
        py = y_rem[p_idx]
        qx = x_rem[q_idx]
        qy = y_rem[q_idx]
        
        # Vectors from i
        v1x = px - x[i]
        v1y = py - y[i]
        v2x = qx - x[i]
        v2y = qy - y[i]
        
        # Cross product
        cross = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(cross)
        
        local_min = np.min(areas)
        if local_min < min_area:
            min_area = local_min
            if min_area < 1e-12:
                return 0.0
                
    if min_area == float('inf'):
        return 0.0
        
    return min_area

def _score(pts):
    """
    Compute min triangle area / hull area.
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
        min_area = _min_triangle_area_vec(pts)
        return min_area / hull_area
    except Exception:
        return 0.0

def _normalize_unit_area(pts):
    """
    Scale points so that their convex hull has area 1.
    """
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area > 1e-9:
            return pts / np.sqrt(area)
    except Exception:
        pass
    # Fallback: if hull calculation fails or area is 0, return as is or randomize
    return pts

def _initial_regular_polygon():
    angles = 2 * np.pi * np.arange(N) / N
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)

def _initial_mixed():
    # A mix of regular polygon and some random perturbation
    pts = _initial_regular_polygon()
    rng = np.random.RandomState(123)
    noise = rng.normal(0, 0.1, size=pts.shape)
    pts += noise
    return _normalize_unit_area(pts)

def _initial_random():
    rng = np.random.RandomState(42)
    pts = rng.uniform(-1, 1, size=(N, 2))
    return _normalize_unit_area(pts)

def construct():
    start_time = time.time()
    
    # Differential Evolution parameters
    # Population size: 40 is reasonable for 28 dimensions
    pop_size = 40
    # Bounds: points are normalized to unit area, so coordinates are roughly within [-2, 2]
    # We will re-normalize after each step, so bounds are effectively infinite, 
    # but DE works better with bounds. We'll use a large box and rely on normalization.
    low = -5.0
    high = 5.0
    
    # Initialize population
    # Use a mix of good inits and random ones
    population = []
    
    # Add some structured inits
    population.append(_initial_regular_polygon())
    population.append(_initial_mixed())
    population.append(_initial_random())
    
    rng = np.random.RandomState(999)
    while len(population) < pop_size:
        # Random points, then normalized
        pts = rng.uniform(low, high, size=(N, 2))
        population.append(_normalize_unit_area(pts))
        
    population = np.array(population)
    
    # Evaluate initial population
    scores = np.array([_score(p) for p in population])
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_pts = population[best_idx].copy()
    
    # DE parameters
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover rate
    
    # Main loop
    while True:
        if time.time() - start_time > DEADLINE - 2:
            break
            
        # Check if we have time for a full generation
        # Each generation involves pop_size evaluations.
        # _score is fast, but let's estimate.
        # If we are close to deadline, break.
        
        for i in range(pop_size):
            if time.time() - start_time > DEADLINE - 1:
                break
                
            # Select 3 distinct random individuals different from i
            # Using a simple random choice without replacement
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = rng.choice(candidates, size=3, replace=False)
            
            # Differential Evolution mutation
            # v = population[a] + F * (population[b] - population[c])
            v = population[a] + F * (population[b] - population[c])
            
            # Crossover
            # For each dimension, with probability CR, take from v, else from population[i]
            # N=14, dims=28
            mask = rng.rand(N, 2) < CR
            # Ensure at least one dimension is taken from v to maintain diversity
            # (Standard DE does this, but for small dims, it's less critical)
            if not np.any(mask):
                # Force one random bit to be True
                r_idx = rng.randint(0, N*2)
                mask.flat[r_idx] = True
                
            # Create trial vector
            trial = population[i].copy()
            trial[mask] = v[mask]
            
            # Normalize trial to unit area
            trial = _normalize_unit_area(trial)
            
            # Evaluate trial
            trial_score = _score(trial)
            
            # Selection
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial.copy()
                    
    return best_pts
# EVOLVE-BLOCK-END
