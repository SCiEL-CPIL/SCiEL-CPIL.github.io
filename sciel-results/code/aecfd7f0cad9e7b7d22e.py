import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations

N = 13
# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Strategy: Search for 13 points in a bounded integer grid.
    # The score is scale-invariant, so we can search in a grid of size GxG.
    # We want to maximize min_area / hull_area.
    # For integer points, areas are multiples of 0.5.
    
    # A good initial guess is a regular 13-gon, but we are now on a grid.
    # We will generate a set of candidate points from a grid.
    # To ensure diversity and avoid clustering, we can use a "spread" heuristic.
    
    # Let's try a few grid sizes.
    # Small grids limit the search space.
    # Large grids increase it.
    # We will use a randomized search over the grid.
    
    best_score = 0.0
    best_points = None
    
    # Initial candidate: Regular polygon scaled to fit in a box
    # This serves as a baseline, though it won't be integer.
    # We will focus on integer points.
    
    # Let's define a function to generate random integer points in a box [-G, G]
    # and then filter/evaluate.
    
    # To be efficient, we can pre-generate a pool of "good" candidate points.
    # Points on the boundary of the convex hull of the grid are often good for maximizing hull area.
    # Points well-distributed in the interior are good for minimizing small triangles.
    
    # Let's try a population-based search on the integer grid.
    # Population size: 50
    # Generations: enough to fill time budget.
    
    # Grid size: Let's start with G=10 (coordinates from -10 to 10).
    # Total points in grid: 21*21 = 441.
    # We need to choose 13.
    
    # We can represent a configuration as a set of 13 indices from the grid.
    # Or just the coordinates.
    
    # Let's use a simple local search on the integer grid.
    # 1. Start with a random configuration of 13 distinct integer points.
    # 2. Calculate score.
    # 3. Try moving one point to a nearby integer point (within radius 1 or 2).
    # 4. If score improves, accept.
    # 5. Repeat.
    
    # To avoid local minima, we will run multiple restarts.
    
    G = 15  # Grid half-size
    coords = np.arange(-G, G + 1)
    grid_points = np.array(list(combinations(coords, 2)))
    num_grid_points = len(grid_points)
    
    # Map from coordinate tuple to index for quick lookup
    # Not strictly necessary if we just keep coordinate arrays, but useful for uniqueness.
    
    def _random_config():
        """Generate a random configuration of N distinct integer points."""
        # Randomly sample N distinct points from the grid
        indices = np.random.choice(num_grid_points, N, replace=False)
        return grid_points[indices].astype(float)
    
    def _neighbor_configs(points, radius=1):
        """Generate neighbor configurations by moving one point by a small integer step."""
        # This can be expensive if done naively.
        # Instead, we'll just propose a single random move per step.
        pass
    
    # Simulated Annealing approach on the integer grid.
    
    num_restarts = 10
    time_per_restart = (deadline - start_time) / (num_restarts + 1)
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize with a random configuration
        current_points = _random_config()
        current_score = calculate_score(current_points)
        
        # If this is a degenerate hull, skip
        if current_score == 0:
            continue
            
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.95
        max_steps = 2000 # Adjust based on time
        
        step_time = (time_per_restart - 0.1) / max_steps
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Propose a move: pick a random point and move it by a random small integer vector
            idx = np.random.randint(N)
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            if dx == 0 and dy == 0:
                continue
            
            new_points = current_points.copy()
            new_points[idx] = current_points[idx] + np.array([dx, dy])
            
            # Check if the new point is within the grid bounds and not colliding with others
            # Bounds check
            if not (-G <= new_points[idx, 0] <= G and -G <= new_points[idx, 1] <= G):
                continue
            
            # Collision check: is the new point identical to any other point?
            # Since we are on a grid, we can check exact match.
            # Note: floating point might be an issue if we don't keep integers.
            # We keep them as float but they are integer values.
            
            collision = False
            for j in range(N):
                if j == idx:
                    continue
                if np.allclose(new_points[idx], current_points[j], atol=1e-9):
                    collision = True
                    break
            
            if collision:
                continue
            
            new_score = calculate_score(new_points)
            
            # Acceptance probability
            if new_score > current_score:
                # Accept improvement
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept worse with probability exp((new - current) / T)
                # Note: scores are positive. We want to maximize.
                # Delta = new_score - current_score (negative)
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current_points = new_points
                        current_score = new_score
            
            T *= cooling
            
            # If T is very small, we are in greedy mode.
            if T < 1e-6:
                break
                
    # If we found integer points, we can try to "polish" them using continuous optimization
    # starting from the integer configuration, but allowing real coordinates.
    # This might help if the optimal solution is not on the integer grid.
    
    if best_points is not None and time.time() < deadline - 20:
        # Use scipy.optimize.minimize to refine the integer solution
        # We need to handle the non-smoothness of min_area.
        # We can use the "target" method or just simple gradient descent on a smoothed objective.
        # However, the integer solution is already a valid candidate.
        # Let's try a simple local search with continuous steps.
        
        def _continuous_local_search(points, time_limit):
            start_t = time.time()
            best_pts = points.copy()
            best_sc = calculate_score(best_pts)
            
            step_size = 0.1
            for _ in range(500):
                if time.time() > time_limit:
                    break
                if step_size < 1e-6:
                    break
                    
                # Random direction
                dir = np.random.randn(N, 2)
                dir /= np.linalg.norm(dir, axis=1, keepdims=True)
                
                new_pts = best_pts + step_size * dir
                new_sc = calculate_score(new_pts)
                
                if new_sc > best_sc:
                    best_pts = new_pts
                    best_sc = new_sc
                else:
                    step_size *= 0.9
                    
            return best_pts, best_sc
        
        refined_pts, refined_sc = _continuous_local_search(best_points, deadline - 5)
        if refined_sc > best_score:
            best_score = refined_sc
            best_points = refined_pts
            
    # Fallback if no valid points found (should not happen with random init)
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
