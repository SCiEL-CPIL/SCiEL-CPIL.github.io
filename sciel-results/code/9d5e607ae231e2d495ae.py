"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

# Precompute triplets for vectorized area calculation
def _generate_triplets(n):
    return np.array(list(itertools.combinations(range(n), 3)))

TRIPLETS = _generate_triplets(N)

def _fast_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Assumes points are in convex position for hull area calculation.
    """
    # 1. Convex Hull Area (Shoelace formula on angle-sorted points)
    centroid = np.mean(points, axis=0)
    # Handle case where all points are identical or very close
    if np.allclose(points, points[0]):
        return 0.0
        
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = points[sorted_idx]
    
    x = sorted_pts[:, 0]
    y = sorted_pts[:, 1]
    
    # Shoelace formula: 0.5 * | sum(x_i * y_{i+1} - x_{i+1} * y_i) |
    hull_area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    
    if hull_area < 1e-12:
        return 0.0
        
    # 2. Minimum Triangle Area
    # Vectorized calculation for all 364 triangles
    pi = points[TRIPLETS[:, 0]]
    pj = points[TRIPLETS[:, 1]]
    pk = points[TRIPLETS[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we stop at 110s
    deadline = start_time + 110.0
    
    # Initialization: Regular 14-gon
    # This is a strong local optimum for this problem
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    current_score = _fast_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # Initial temperature: needs to be large enough to accept worse moves
    # Score scale is ~0.07. Differences can be small.
    T = 0.01
    T_min = 1e-5
    cooling_rate = 0.9995
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Batch of steps? No, just loop until time out.
        # Pick a random point to perturb
        idx = rng.integers(N)
        
        # Gaussian perturbation
        delta = rng.normal(0, step_size * T, size=2)
        new_points = points.copy()
        new_points[idx] += delta
        
        new_score = _fast_score(new_points)
        
        # Acceptance Criteria
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Metropolis Acceptance
            # Avoid division by zero or log issues
            if T > T_min and new_score > 0:
                # Probability of accepting worse move
                # Delta E = current - new (positive if new is worse)
                delta_e = current_score - new_score
                if delta_e < 100 * T: # Overflow guard
                    prob = np.exp(-delta_e / T)
                    if rng.random() < prob:
                        points = new_points
                        current_score = new_score
            
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Reheat occasionally? 
            # For now, keep it simple. If stuck, we rely on the initial good guess.
            
    return best_points
# EVOLVE-BLOCK-END
