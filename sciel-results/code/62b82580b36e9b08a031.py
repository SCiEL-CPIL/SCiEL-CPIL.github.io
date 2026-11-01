"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _score(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
        
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j+1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    if min_area <= 0:
                        return 0.0
    return min_area / hull_area

def construct():
    # Start with regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    start_time = time.time()
    max_time = 110.0 # 10s buffer
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    temperature = 0.05
    step_size = 0.1
    min_temperature = 0.001
    
    # Track stagnation
    stuck_counter = 0
    
    while time.time() - start_time < max_time:
        # Pick a random point
        idx = rng.integers(0, N)
        
        # Perturb
        new_points = points.copy()
        noise = rng.normal(0, step_size, size=2)
        new_points[idx] += noise
        
        new_score = _score(new_points)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        accept = False
        if delta > 0:
            accept = True
        else:
            if temperature > min_temperature:
                prob = np.exp(delta / temperature)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            points = new_points
            current_score = new_score
            stuck_counter = 0
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            stuck_counter += 1
            
        # Cool down
        temperature *= 0.999
        
        # If stuck for a long time, do a kick
        if stuck_counter > 200:
            # Move a random point significantly
            kick_idx = rng.integers(0, N)
            # Reset to random position on circle or random in disk?
            # Let's try random on circle to maintain convexity tendency
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0.5, 1.0)
            points[kick_idx] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            current_score = _score(points)
            stuck_counter = 0
            # Increase temperature slightly to explore
            temperature = max(temperature, 0.01)
            
    return best_points
# EVOLVE-BLOCK-END
