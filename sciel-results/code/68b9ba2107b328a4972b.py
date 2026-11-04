"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _initial_construction_rings():
    """2 rings: 7 outer, 7 inner."""
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    r_outer = 1.0
    r_inner = 0.5
    
    outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Use the best known initialization: concentric heptagons
    current_pts = _initial_construction_rings()
    
    # Normalize initial points to unit RMS radius to ensure consistent scale
    current_pts = current_pts / np.sqrt(np.mean(np.sum(current_pts**2, axis=1)))
    
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.5  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.9995
    max_iter = 50000
    step_scale = 0.05
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a random perturbation
        # We perturb all points slightly
        noise = rng.normal(0, step_scale, size=(N, 2))
        new_pts = current_pts + noise
        
        # Normalize new points to maintain scale (prevents drift to origin or infinity)
        # RMS radius normalization
        rms = np.sqrt(np.mean(np.sum(new_pts**2, axis=1)))
        if rms > 1e-6:
            new_pts = new_pts / rms
        else:
            new_pts = current_pts # Fallback if degenerate
            
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(-delta/T)
            delta = current_score - new_score
            if delta < 1e-12:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            
            if rng.random() < prob:
                current_pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        if T < T_min:
            break

    return best_pts
# EVOLVE-BLOCK-END
