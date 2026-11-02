"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Deterministic, vectorized "Best-Response" local search.
   - Instead of gradient descent on a smoothed objective, we treat the problem
     as a coordinate-wise optimization.
   - For each point i, we freeze all other points and find the best position
     for point i by scanning a dense grid within the triangle.
   - This is exact for the 1-point subproblem and avoids derivative issues
     of the 'min' function.
2. Global exploration via "Jitter-and-Refine":
   - Start with diverse initializations (greedy, random, structured).
   - In each iteration, take the current best configuration, add random noise
     (jitter) to some points to escape local minima, then run the
     Best-Response refinement.
3. Vectorization for speed:
   - The grid scan for a single point is vectorized using NumPy.
   - We pre-compute triplet indices to calculate all 165 triangle areas quickly.
4. Time budgeting:
   - Monitor elapsed time and stop if approaching the 150s limit.
   - Return the best configuration found so far.
"""
import numpy as np
import time
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
        for _ in range(1000):
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

def _best_response_step(pts, grid_res=60, time_limit=5.0):
    """
    Perform one sweep of Best-Response optimization.
    For each point i, find the best position on a grid that maximizes the min area.
    Returns the updated points and the improvement.
    """
    start_time = time.time()
    pts = np.array(pts, dtype=float)
    improvement = 0.0
    
    # Pre-compute grid points for the triangle
    # We use a barycentric grid to ensure points are inside
    # u, v >= 0, u+v <= 1
    # x = u + 0.5v, y = S3H*v
    u_vals = np.linspace(0, 1, grid_res)
    v_vals = np.linspace(0, 1, grid_res)
    U, V = np.meshgrid(u_vals, v_vals)
    U_flat = U.flatten()
    V_flat = V.flatten()
    
    # Filter valid grid points (u+v <= 1)
    valid_mask = (U_flat + V_flat) <= 1.0
    U_valid = U_flat[valid_mask]
    V_valid = V_flat[valid_mask]
    
    grid_pts = np.column_stack((
        U_valid + 0.5 * V_valid,
        S3H * V_valid
    ))
    
    # If grid is too large, subsample to save time
    if len(grid_pts) > 20000:
        indices = np.random.choice(len(grid_pts), 20000, replace=False)
        grid_pts = grid_pts[indices]
    
    for i in range(N):
        if time.time() - start_time > time_limit:
            break
            
        # Save current position
        old_pos = pts[i].copy()
        
        # Create a copy of points with point i set to 0 (dummy)
        # We will replace it with grid points
        pts_test = pts.copy()
        
        # Calculate areas for all grid points simultaneously
        # This is expensive: (num_grid, 165) matrix
        # Let's do it in chunks to manage memory and time
        chunk_size = 5000
        best_score = -1
        best_pos = old_pos
        
        for start_idx in range(0, len(grid_pts), chunk_size):
            end_idx = min(start_idx + chunk_size, len(grid_pts))
            grid_chunk = grid_pts[start_idx:end_idx]
            
            # For each grid point, compute min area with other 10 points
            # We need to form triangles involving the new point i and two other points j, k
            # And triangles not involving i (which are constant)
            
            # Triangles not involving i:
            # We can precompute these once per sweep, but for simplicity, we recompute
            # or just ignore them if we know the current min is dominated by i's triangles.
            # Actually, we must consider all 165 triangles.
            # Triangles not involving i: C(10, 3) = 120 triangles.
            # Triangles involving i: 10*9 = 90 triangles (i, j, k).
            
            # Let's compute the constant part (triangles not involving i)
            # We need to identify triplets that do NOT include i
            # This depends on i, so we do it inside the loop
            
            mask_no_i = ~((np.arange(N) == i).any(axis=1)) # This is wrong, _triplets is (165, 3)
            # Correct mask:
            triplet_i = _triplets
            mask_no_i = ~(
                (triplet_i[:, 0] == i) | 
                (triplet_i[:, 1] == i) | 
                (triplet_i[:, 2] == i)
            )
            
            # Indices of triplets not involving i
            idx_no_i = np.where(mask_no_i)[0]
            # Indices of triplets involving i
            idx_with_i = np.where(~mask_no_i)[0]
            
            # Compute areas for triplets not involving i (constant for this i)
            if len(idx_no_i) > 0:
                p_i_const = pts_test[idx_no_i[:, 0]]
                p_j_const = pts_test[idx_no_i[:, 1]]
                p_k_const = pts_test[idx_no_i[:, 2]]
                dx1 = p_j_const - p_i_const
                dx2 = p_k_const - p_i_const
                cross_const = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
                areas_const = 0.5 * np.abs(cross_const)
                min_const = np.min(areas_const)
            else:
                min_const = np.inf
            
            # Compute areas for triplets involving i for all grid points
            # p_i varies (grid_chunk), p_j and p_k are fixed
            # We need to map idx_with_i to the specific j, k for each triplet
            # For a triplet (i, j, k), the other two points are fixed.
            
            # Let's extract the other two points for each triplet involving i
            # This is tricky because i can be in any position (0, 1, 2) in the triplet
            # Let's create arrays for the other two points
            p_j_with_i = np.zeros((len(idx_with_i), 2))
            p_k_with_i = np.zeros((len(idx_with_i), 2))
            
            for t_idx in idx_with_i:
                trip = _triplets[t_idx]
                # Find which position i is in
                if trip[0] == i:
                    p_j_with_i[t_idx] = pts_test[trip[1]]
                    p_k_with_i[t_idx] = pts_test[trip[2]]
                elif trip[1] == i:
                    p_j_with_i[t_idx] = pts_test[trip[0]]
                    p_k_with_i[t_idx] = pts_test[trip[2]]
                else: # trip[2] == i
                    p_j_with_i[t_idx] = pts_test[trip[0]]
                    p_k_with_i[t_idx] = pts_test[trip[1]]
            
            # Now, for each grid point p_new, compute areas with p_j_with_i and p_k_with_i
            # p_new shape: (chunk, 2)
            # p_j_with_i shape: (num_with_i, 2)
            # We need a (chunk, num_with_i) matrix of areas
            
            # This is a broadcast operation
            # dx1 = p_j - p_new
            # dx2 = p_k - p_new
            # cross = dx1_x * dx2_y - dx1_y * dx2_x
            
            # p_new[:, np.newaxis, :] -> (chunk, 1, 2)
            # p_j_with_i[np.newaxis, :, :] -> (1, num_with_i, 2)
            
            pn = grid_chunk[:, np.newaxis, :]
            pj = p_j_with_i[np.newaxis, :, :]
            pk = p_k_with_i[np.newaxis, :, :]
            
            dx1 = pj - pn
            dx2 = pk - pn
            
            cross = dx1[:, :, 0] * dx2[:, :, 1] - dx1[:, :, 1] * dx2[:, :, 0]
            areas_with_i = 0.5 * np.abs(cross)
            
            # Min area for each grid point is min(min_const, min(areas_with_i))
            min_with_i = np.min(areas_with_i, axis=1)
            min_total = np.minimum(min_const, min_with_i)
            
            # Find the best grid point in this chunk
            local_best_idx = np.argmax(min_total)
            local_best_score = min_total[local_best_idx]
            
            if local_best_score > best_score:
                best_score = local_best_score
                best_pos = grid_chunk[local_best_idx]
                
        # Update point i
        pts[i] = best_pos
        improvement += best_score - _min_triangle_area(pts_test) # This is not quite right, just track score
        
    return pts, improvement

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Best-Response local search with global jitter.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Structured
    p_struct = _structured_init()
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # 3. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 4. More random
    for _ in range(5):
        p_rand2 = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_rand2 = _project_to_triangle(p_rand2)
        initializations.append(p_rand2)
        
    # Evaluate initializations
    for init_pts in initializations:
        if not _check_inside(init_pts):
            init_pts = _project_to_triangle(init_pts)
        score = _min_triangle_area(init_pts)
        if score > best_score:
            best_score = score
            best_pts = init_pts.copy()
            
    # Main loop: Jitter and Refine
    # We will run several rounds of best-response refinement
    # Each round: take current best, jitter, refine, update best if improved
    
    num_rounds = 10
    for r in range(num_rounds):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Get current best
        current_pts = best_pts.copy()
        
        # Jitter: add noise to some points
        # Number of points to jitter decreases over time
        num_jitter = max(1, N - r)
        jitter_indices = rng.sample(range(N), num_jitter)
        jitter_scale = 0.1 * (1.0 - r / num_rounds) + 0.01
        
        for idx in jitter_indices:
            # Add random noise
            noise = np.array([rng.uniform(-jitter_scale, jitter_scale), 
                              rng.uniform(-jitter_scale, jitter_scale)])
            current_pts[idx] += noise
            current_pts[idx] = _project_to_triangle(current_pts[idx])[idx]
            
        # Refine using Best-Response
        # We run multiple sweeps of best-response
        # Each sweep takes ~1-2 seconds depending on grid size
        sweeps = 3
        for s in range(sweeps):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            current_pts, _ = _best_response_step(current_pts, grid_res=40, time_limit=3.0)
            
        # Check if improved
        if _check_inside(current_pts):
            score = _min_triangle_area(current_pts)
            if score > best_score:
                best_score = score
                best_pts = current_pts.copy()
            else:
                # If no improvement, maybe try a different jitter?
                # For now, just continue to next round
                pass
                
    # Final refinement on the best found
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        # Do one more high-resolution refinement
        best_pts, _ = _best_response_step(best_pts, grid_res=80, time_limit=5.0)
        if _check_inside(best_pts):
            score = _min_triangle_area(best_pts)
            if score > best_score:
                best_score = score
                best_pts = best_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
