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
    # Vectorized calculation for all triples is O(N^3) but memory heavy for large N.
    # For N=14, N^3 is ~2744 triples. We can do it with loops or small vectorization.
    # Let's use a loop for clarity and to avoid large temporary arrays, 
    # but optimize the inner cross product.
    
    min_area = float('inf')
    n = len(pts)
    # Precompute differences? No, just iterate.
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            # Vectorized over k > j
            pk = pts[j+1:]
            # Cross product of (pj - pi) and (pk - pi)
            # | (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x) | / 2
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

def _initial_construction():
    """
    Construct initial points using concentric regular polygons.
    This provides a strong symmetric starting point.
    """
    pts = []
    # Strategy: 7 points on outer circle, 7 points on inner circle?
    # Or 1 point center, 6 inner, 7 outer?
    # Let's try: 1 center, 6 on radius r1, 7 on radius r2.
    # Or simply 14 points on a single circle? No, that's degenerate for min area if not careful, 
    # but actually regular polygon on circle has decent min area.
    # However, mixing radii often improves "packing" of triangles.
    
    # Let's try a 2-ring structure: 7 points on radius 1.0, 7 points on radius 0.5.
    # Offset the inner ring to maximize distance to outer ring points.
    
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7 # Offset by half step
    
    r_outer = 1.0
    r_inner = 0.6 # Tunable, 0.5 or 0.6 usually good
    
    for a in outer_angles:
        pts.append([r_outer * np.cos(a), r_outer * np.sin(a)])
    for a in inner_angles:
        pts.append([r_inner * np.cos(a), r_inner * np.sin(a)])
        
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # 1. Initial symmetric construction
    pts = _initial_construction()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # 2. Stochastic Local Search (Simulated Annealing / Hill Climbing)
    # We want to maximize the minimum triangle area.
    # Perturb a single point at a time.
    
    # Parameters
    T = 1.0 # Temperature
    T_min = 1e-5
    cooling_rate = 0.999
    max_iters = 100000 # Will be limited by time
    
    rng = np.random.default_rng(42)
    
    i = 0
    while time.time() < deadline:
        i += 1
        # Select a random point to move
        idx = rng.integers(0, N)
        
        # Generate a random perturbation
        # Gaussian noise, scale depends on temperature
        noise_scale = T * 0.1
        move = rng.normal(0, noise_scale, size=2)
        
        new_pts = pts.copy()
        new_pts[idx] += move
        
        # Optional: Keep points within a reasonable bounding box?
        # The score is normalized by hull area, so scaling doesn't matter, 
        # but extreme points might create tiny hull areas or numerical issues.
        # We can normalize the hull area to 1.0 after each step to keep numbers stable?
        # No, the score function handles normalization. 
        # However, if points drift far apart, float precision might degrade.
        # Let's re-center and scale occasionally? 
        # Actually, since score is scale-invariant, we can just normalize 
        # the point set to unit hull area every K steps to prevent drift.
        
        if i % 100 == 0:
            # Normalize to unit hull area
            try:
                hull = ConvexHull(new_pts)
                area = hull.volume
                if area > 1e-9:
                    new_pts = new_pts / np.sqrt(area)
            except:
                pass
            
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            # Always accept improvement
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            # Metropolis criterion: accept worse with probability exp((new - cur)/T)
            # Since we maximize, delta = new - cur < 0
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    pts = new_pts
                    current_score = new_score
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_pts
# EVOLVE-BLOCK-END
