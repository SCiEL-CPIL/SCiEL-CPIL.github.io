import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 140.0  # Safety margin

def _hull_area(pts):
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas.extend(0.5 * np.abs(cross))
    return np.array(areas)

def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 0.0
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    return min_a / hull_a

def _objective_de(x):
    """Objective for Differential Evolution. Minimizes negative score."""
    pts = x.reshape(-1, 2)
    score = _score(pts)
    if score == 0.0:
        return 1e10
    return -score

def _local_polish(pts, time_limit=10.0):
    """
    Greedy local search:
    1. Find min area triangle.
    2. Try to move its vertices to increase its area.
    3. Check if the new global min area is improved.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    start = time.time()
    
    # We will try to improve the minimum area.
    # Strategy: Identify the min triangle (i,j,k). 
    # Perturb i, j, k slightly. 
    # If the new min area over ALL triangles is > old min area, accept.
    
    # To be faster, we only accept if the specific min triangle area increases 
    # AND no other triangle area drops below the current global min.
    
    current_min_area = np.min(_all_triangle_areas(best_pts))
    
    # Helper to check if a perturbation is valid
    def try_move(pts, idx, delta):
        new_pts = pts.copy()
        new_pts[idx] += delta
        areas = _all_triangle_areas(new_pts)
        new_min = np.min(areas)
        return new_pts, new_min
    
    # Simple hill climbing with random steps
    step_size = 0.05
    rng = np.random.RandomState(42)
    
    while time.time() - start < time_limit:
        pts = best_pts
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        
        # Identify vertices of the min triangle
        # Reconstruct indices from linear index is annoying, so let's just re-find i,j,k
        n = N
        # Find i,j,k for min_idx
        count = 0
        min_i, min_j, min_k = -1, -1, -1
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if count == min_idx:
                        min_i, min_j, min_k = i, j, k
                        break
                    count += 1
                if min_i != -1: break
            if min_i != -1: break
            
        if min_i == -1: break
        
        vertices = [min_i, min_j, min_k]
        improved = False
        
        # Try moving each vertex of the min triangle
        for v in vertices:
            # Try a few random directions
            for _ in range(10):
                direction = rng.normal(0, 1, size=2)
                direction /= (np.linalg.norm(direction) + 1e-9)
                # Adaptive step size
                delta = step_size * direction
                
                new_pts, new_min_area = try_move(pts, v, delta)
                
                # Accept if new global min area is strictly better
                if new_min_area > current_min_area + 1e-9:
                    best_pts = new_pts
                    current_min_area = new_min_area
                    best_score = _score(best_pts)
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            # Reduce step size or break? 
            # Let's reduce step size slightly to refine
            step_size *= 0.9
            if step_size < 1e-5:
                break
                
    return best_pts, best_score

def construct():
    start_time = time.time()
    
    # 1. Differential Evolution
    # Bounds: Let's use [-10, 10] for x and y.
    bounds = [(-10, 10)] * (2 * N)
    
    # DE parameters
    # popsize=20, maxiter=100 is a start. 
    # We need to monitor time.
    
    best_de_pts = None
    best_de_score = 0.0
    
    try:
        # Run DE
        # tol=1e-6, maxiter=50 might be enough if we have multiple restarts
        # But DE is slow. Let's run one big DE run.
        result = differential_evolution(
            _objective_de, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            polish=False, # We will do our own polish
            init='sobol' # Better initialization
        )
        
        if result.success:
            best_de_pts = result.x.reshape(-1, 2)
            best_de_score = _score(best_de_pts)
            
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE didn't run or failed, use a random init
    if best_de_pts is None:
        rng = np.random.RandomState(42)
        best_de_pts = rng.uniform(-10, 10, size=(N, 2))
        best_de_score = _score(best_de_pts)

    # 2. Local Polish
    # Use remaining time for local polish
    elapsed = time.time() - start_time
    time_left = DEADLINE - elapsed - 5.0 # 5s safety
    
    if time_left > 5.0:
        polished_pts, polished_score = _local_polish(best_de_pts, time_limit=time_left)
        
        if polished_score > best_de_score:
            best_de_pts = polished_pts
            best_de_score = polished_score

    # 3. Final Normalization
    # The score is scale invariant, but we should return points in a reasonable range
    # Normalize so hull area is 1? The task says "normalized to unit area".
    # "The notebook normalizes by the point set convex hull area."
    # This implies the evaluator calculates score = min_area / hull_area.
    # So the actual scale of the returned points doesn't matter for the score,
    # but returning them in a standard scale is good practice.
    
    # Let's normalize to unit area hull for the return value.
    final_pts = best_de_pts
    hull_a = _hull_area(final_pts)
    if hull_a > 1e-10:
        final_pts = final_pts / np.sqrt(hull_a)
        
    return final_pts

if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
