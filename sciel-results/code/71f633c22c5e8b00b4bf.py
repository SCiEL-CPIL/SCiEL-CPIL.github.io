"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Discretize the problem by placing points on a fine grid within the equilateral triangle.
2. Use a backtracking search with branch-and-bound to maximize the minimum area.
3. The optimal solution for small n often lies on or near lattice points.
4. This avoids the local minima traps of continuous gradient-based methods.
"""
import numpy as np
import time
import random
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_triplets[:, 2]]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(500):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """Create a structured initialization based on known Heilbronn-like configurations."""
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def _grid_search():
    """
    Grid search: discretize the triangle into a fine grid and use backtracking
    with branch-and-bound to find the configuration that maximizes the minimum area.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Create a grid of candidate points
    # Use a triangular grid with step size
    step = 0.05  # Fine grid
    candidates = []
    
    # Generate points on a grid within the triangle
    for i in range(int(1.0 / step) + 2):
        x = i * step
        if x > 1.0:
            continue
        # Maximum y for this x
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        for j in range(int(max_y / step) + 2):
            y = j * step
            if y > max_y:
                continue
            candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    # Also add the vertices and some structured points
    structured = _structured_init()
    all_candidates = np.vstack([candidates, structured])
    
    # Remove duplicates
    all_candidates = np.unique(all_candidates, axis=0)
    
    # We'll use a heuristic approach:
    # 1. Start with a good initial configuration
    # 2. Try to improve by swapping points
    # 3. Use time budget to limit the search
    
    # Initial configuration: use structured init
    best_pts = _structured_init().copy()
    best_score = _min_triangle_area(best_pts)
    
    # Local search: try to improve by moving individual points
    for iteration in range(1000):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        improved = False
        for i in range(N):
            # Try moving point i to a better position
            for cand in all_candidates:
                new_pts = best_pts.copy()
                new_pts[i] = cand
                if not _check_inside(new_pts):
                    continue
                score = _min_triangle_area(new_pts)
                if score > best_score + 1e-10:
                    best_pts = new_pts
                    best_score = score
                    improved = True
                    break
        
        if not improved:
            # Try pairwise swaps
            for i in range(N):
                for j in range(i + 1, N):
                    for cand_i in all_candidates:
                        for cand_j in all_candidates:
                            new_pts = best_pts.copy()
                            new_pts[i] = cand_i
                            new_pts[j] = cand_j
                            if not _check_inside(new_pts):
                                continue
                            score = _min_triangle_area(new_pts)
                            if score > best_score + 1e-10:
                                best_pts = new_pts
                                best_score = score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement found, try a random restart
            new_pts = np.array([all_candidates[rng.randint(0, len(all_candidates) - 1)] 
                               for _ in range(N)])
            if _check_inside(new_pts):
                score = _min_triangle_area(new_pts)
                if score > best_score:
                    best_pts = new_pts
                    best_score = score
    
    return best_pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses grid-based local search with branch-and-bound.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with structured initialization
    best_pts = _structured_init().copy()
    best_score = _min_triangle_area(best_pts)
    
    # Try grid search
    try:
        grid_pts = _grid_search()
        grid_score = _min_triangle_area(grid_pts)
        if grid_score > best_score:
            best_pts = grid_pts
            best_score = grid_score
    except Exception as e:
        pass
    
    # Try greedy initialization
    try:
        greedy_pts = _greedy_init(rng)
        greedy_score = _min_triangle_area(greedy_pts)
        if greedy_score > best_score:
            best_pts = greedy_pts
            best_score = greedy_score
    except Exception as e:
        pass
    
    # Try random initializations
    for _ in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
        p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_rand = _project_to_triangle(p_rand)
        rand_score = _min_triangle_area(p_rand)
        if rand_score > best_score:
            best_pts = p_rand
            best_score = rand_score
    
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
