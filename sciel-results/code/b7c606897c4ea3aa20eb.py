import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

# EVOLVE-BLOCK-START
def compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0 if points are degenerate or hull is invalid.
    """
    if len(points) < 3:
        return 0.0
    
    # Check for duplicate points which cause zero area
    # This is a quick check, full check is in triangle calc
    # We can skip explicit duplicate check for speed, as area will be 0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Min Triangle Area
    # There are N*(N-1)*(N-2)/6 triangles. For N=13, 286 triangles.
    # We can vectorize this or loop. Loop is fine for 286.
    
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
                        
    return min_area / hull_area

def normalize_points(points):
    """
    Normalize points to have centroid at (0,0) and max radius 1.
    This makes the problem scale-invariant and stabilizes optimization.
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    distances = np.linalg.norm(points_centered, axis=1)
    max_dist = np.max(distances)
    if max_dist < 1e-10:
        return points_centered
    return points_centered / max_dist

def objective_wrapper(x):
    """
    Objective function for the optimizer.
    x is a flat array of size 2*N.
    Returns negative score (to minimize).
    """
    points = x.reshape(-1, 2)
    # Normalize to keep scale fixed
    points = normalize_points(points)
    score = compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_score = 0.0
    best_points = None
    
    # Generate initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(reg_poly)
    
    # 2. Jittered Regular Polygon
    rng = np.random.RandomState(42)
    jittered = reg_poly + rng.normal(0, 0.1, (N, 2))
    candidates.append(jittered)
    
    # 3. Grid-like (4x4 minus 3)
    # Create a 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.array([[x, y] for x in xs for y in ys])
    # Remove 3 corners to make it more convex/round?
    # Indices: 0:(-1,-1), 3:(-1,1), 12:(1,-1), 15:(1,1)
    # Removing corners 0, 3, 15 leaves 13 points.
    grid_13 = np.delete(grid, [0, 3, 15], axis=0)
    candidates.append(grid_13)
    
    # 4. Two Rings
    # 5 inner, 8 outer?
    r1 = 0.5
    r2 = 1.0
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ring1 = r1 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    ring2 = r2 * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    two_rings = np.vstack([ring1, ring2])
    candidates.append(two_rings)
    
    # 5. Random points
    candidates.append(rng.rand(N, 2) - 0.5)

    # Evaluate initial candidates
    for cand in candidates:
        cand_norm = normalize_points(cand)
        score = compute_score(cand_norm)
        if score > best_score:
            best_score = score
            best_points = cand_norm.copy()
        
        if time.time() > deadline:
            break

    # Local Optimization using Nelder-Mead
    # We will try to optimize the best few candidates
    # Sort candidates by score
    scored_candidates = []
    for cand in candidates:
        cand_norm = normalize_points(cand)
        score = compute_score(cand_norm)
        scored_candidates.append((score, cand_norm))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Optimize top 3 candidates
    for i, (score, cand) in enumerate(scored_candidates[:3]):
        if time.time() > deadline - 10: # Leave 10s for final checks
            break
            
        x0 = cand.flatten()
        try:
            result = minimize(objective_wrapper, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.success or result.fun < -score:
                opt_points = normalize_points(result.x.reshape(-1, 2))
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass

    # If best_points is still None (shouldn't happen), fallback
    if best_points is None:
        best_points = normalize_points(candidates[0])
        
    return best_points
# EVOLVE-BLOCK-END
