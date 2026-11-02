import numpy as np
import time
import itertools
import random

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
    p_k = pts[_k_idx]
    
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

def _generate_grid_points(resolution):
    """
    Generate candidate points on a triangular grid.
    We use barycentric coordinates (u, v, w) with u+v+w=1, u,v,w >= 0.
    Grid points are defined by integer steps.
    """
    points = []
    # Barycentric coordinates: (1-i/k - j/k, i/k, j/k)
    # i, j >= 0, i+j <= k
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            u = 1.0 - i / resolution - j / resolution
            v = i / resolution
            w = j / resolution
            # Convert to Cartesian:
            # A=(0,0), B=(1,0), C=(0.5, S3H)
            # P = u*A + v*B + w*C
            # x = v*1 + w*0.5
            # y = w*S3H
            x = v + 0.5 * w
            y = w * S3H
            points.append([x, y])
    return np.array(points)

def _local_refinement(pts, max_iters=50, step_size=0.05, rng=None):
    """
    Geometric local refinement.
    Identify the triplet with minimum area.
    Try to move one of its points slightly away from the other two to increase area.
    This is a heuristic "push" method.
    """
    if rng is None:
        rng = random.Random(42)
    
    pts = np.array(pts, dtype=float)
    current_min_area = _min_triangle_area(pts)
    
    for _ in range(max_iters):
        # Find the worst triplet
        p_i = pts[_i_idx]
        p_j = pts[_j_idx]
        p_k = pts[_k_idx]
        dx1 = p_j - p_i
        dx2 = p_k - p_i
        cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        if min_area > current_min_area * 1.01:
            # Improved significantly, update current best
            current_min_area = _min_triangle_area(pts)
        
        # Get the indices of the worst triplet
        i, j, k = _triplets[min_idx]
        
        # Strategy: Try to move point i, j, or k
        # Moving a point increases area of triangles it's part of,
        # but decreases others. We want to increase the MIN area.
        # A good heuristic: move the point that is "closest" to the line segment of the other two?
        # Or simply try small random perturbations and keep if global min improves.
        
        improved = False
        # Try perturbing each of the 3 points in the worst triplet
        for idx in [i, j, k]:
            original_pt = pts[idx].copy()
            
            # Generate a few candidate moves
            for _ in range(20):
                # Random direction, small step
                angle = rng.uniform(0, 2 * np.pi)
                dx = step_size * np.cos(angle)
                dy = step_size * np.sin(angle)
                
                new_pt = original_pt + np.array([dx, dy])
                
                # Check if new point is inside triangle
                x, y = new_pt
                if y < 0 or y > np.sqrt(3) * x or y > np.sqrt(3) * (1 - x) or x < 0 or x > 1:
                    continue
                
                # Test
                test_pts = pts.copy()
                test_pts[idx] = new_pt
                
                new_min_area = _min_triangle_area(test_pts)
                
                if new_min_area > current_min_area:
                    pts = test_pts
                    current_min_area = new_min_area
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # No improvement found in this iteration, maybe reduce step size or break
            step_size *= 0.9
            if step_size < 1e-4:
                break

    return pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Strategy:
    1. Generate a set of candidate points on a coarse grid.
    2. Select 11 points from this grid to maximize the minimum triangle area (combinatorial search).
       Since C(~100, 11) is too large, we use a greedy + local search approach on the grid indices.
    3. Refine the selected points using continuous local search.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Stage 1: Grid-based Combinatorial Search
    # We need 11 points. A grid of resolution 10 gives ~66 points. C(66, 11) is huge.
    # Instead, we use a greedy construction on the grid, then local swap search.
    
    resolutions = [10, 15, 20]
    
    for res in resolutions:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        grid_pts = _generate_grid_points(res)
        n_grid = len(grid_pts)
        
        # Greedy initialization on grid
        # Start with a random point, then add points that maximize min area with existing set
        # This is O(N * N_grid * 165) per step, which is fast.
        
        best_grid_indices = None
        best_grid_score = -1.0
        
        # Try a few random greedy starts
        for start_iter in range(5):
            if time.time() - start_time > DEADLINE - 30:
                break
                
            selected = []
            # Pick a random start point
            start_idx = rng.randint(0, n_grid - 1)
            selected.append(start_idx)
            
            for _ in range(N - 1):
                best_next_idx = -1
                best_next_score = -1.0
                
                # Evaluate all candidates not yet selected
                # Optimization: Only evaluate candidates that are "far" from existing ones?
                # For now, evaluate all. 66 * 11 * 165 is small.
                current_pts = grid_pts[selected]
                
                for cand_idx in range(n_grid):
                    if cand_idx in selected:
                        continue
                    
                    test_pts = np.vstack([current_pts, grid_pts[cand_idx:cand_idx+1]])
                    # Calculate min area for the new set
                    # We only need to check triangles involving the new point?
                    # No, min area could be among existing points, but that didn't change.
                    # So we only need to check triangles involving the new point.
                    # However, _min_triangle_area checks all 165. 
                    # For speed, we can just call it. It's vectorized and fast.
                    
                    score = _min_triangle_area(test_pts)
                    
                    if score > best_next_score:
                        best_next_score = score
                        best_next_idx = cand_idx
                
                if best_next_idx != -1:
                    selected.append(best_next_idx)
                else:
                    break
            
            if len(selected) == N:
                final_grid_pts = grid_pts[selected]
                score = _min_triangle_area(final_grid_pts)
                if score > best_grid_score:
                    best_grid_score = score
                    best_grid_indices = selected.copy()
        
        # Local Search on Grid Indices (Swaps)
        if best_grid_indices is not None:
            current_indices = best_grid_indices.copy()
            current_score = best_grid_score
            current_pts = grid_pts[current_indices]
            
            # Swap search: try swapping one selected index with an unselected one
            # Limit swaps to avoid getting stuck in local minima of the grid
            max_swaps = 50
            swap_iter = 0
            while swap_iter < max_swaps and time.time() - start_time < DEADLINE - 20:
                improved = False
                # Shuffle to break symmetry
                rng.shuffle(current_indices) # Wait, shuffling indices changes the set? No, it's a list of indices.
                # Actually, we should iterate through selected indices and try swapping with unselected.
                
                unselected = set(range(n_grid)) - set(current_indices)
                
                for i, idx in enumerate(current_indices):
                    # Try swapping idx with each unselected index
                    # To save time, only try a few random unselected indices
                    candidates = list(unselected)
                    rng.shuffle(candidates)
                    for new_idx in candidates[:10]: # Try 10 random swaps per selected point
                    
                        new_indices = current_indices.copy()
                        new_indices[i] = new_idx
                        
                        # Check for duplicates (shouldn't happen if we manage sets properly, but safe)
                        if len(set(new_indices)) < N:
                            continue
                            
                        test_pts = grid_pts[new_indices]
                        score = _min_triangle_area(test_pts)
                        
                        if score > current_score + 1e-9:
                            current_indices = new_indices
                            current_score = score
                            current_pts = test_pts
                            improved = True
                            break
                    if improved:
                        break # Restart swap loop with new best
                
                if not improved:
                    break
                
                swap_iter += 1
            
            # Refine the best grid solution with continuous local search
            refined_pts = _local_refinement(current_pts, max_iters=100, step_size=0.02, rng=rng)
            
            if _check_inside(refined_pts):
                final_score = _min_triangle_area(refined_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_pts = refined_pts.copy()
    
    # Fallback: If grid search didn't yield good results or time ran out, use a simple structured init
    if best_pts is None or time.time() - start_time > DEADLINE - 5:
        # Simple structured points
        p_struct = [
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ]
        best_pts = _project_to_triangle(np.array(p_struct))
        best_score = _min_triangle_area(best_pts)
        
        # Quick refinement if time permits
        if time.time() - start_time < DEADLINE - 10:
            refined_pts = _local_refinement(best_pts, max_iters=50, step_size=0.01, rng=rng)
            if _check_inside(refined_pts):
                ref_score = _min_triangle_area(refined_pts)
                if ref_score > best_score:
                    best_pts = refined_pts
                    best_score = ref_score

    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
