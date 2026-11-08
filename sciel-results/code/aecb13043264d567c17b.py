import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Grid resolution. Higher resolution allows finer positioning but slower precomputation.
    # 100x100 grid = 10,000 points. 26 circles.
    # Pairwise precomputation: 10000^2 is too big.
    # We only need distances between occupied cells? No, we need distances between any two potential centers.
    # However, we can restrict the search space.
    # Let's use a grid of 50x50 for the centers.
    # Number of cells = 2500.
    # Pairwise matrix 2500x2500 = 6.25M floats = 50MB. Feasible.
    
    grid_res = 50
    cell_size = 1.0 / grid_res
    # Centers are at (i + 0.5) * cell_size
    xs = (np.arange(grid_res) + 0.5) * cell_size
    ys = (np.arange(grid_res) + 0.5) * cell_size
    
    # Precompute pairwise distances between all grid points
    # X: (G, G) meshgrid of x coords
    # Y: (G, G) meshgrid of y coords
    X, Y = np.meshgrid(xs, ys)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Distance matrix D[i, j] = distance between point i and point j
    # This is 2500x2500
    dx = X_flat[:, np.newaxis] - X_flat[np.newaxis, :]
    dy = Y_flat[:, np.newaxis] - Y_flat[np.newaxis, :]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    
    # For each pair of grid points (i, j), the max radius sum is dist(i, j)
    # Also, for each point i, the max radius is min(distance to boundary)
    # dist to boundary for point (x, y): min(x, 1-x, y, 1-y)
    # Since x, y are grid centers:
    # x = (i+0.5)/G, 1-x = (G-i-0.5)/G
    # min(x, 1-x) is distance to vertical walls
    # min(y, 1-y) is distance to horizontal walls
    # Let's compute max possible radius for each grid point based on boundaries
    # r_max_boundary[i] = min(X_flat[i], 1-X_flat[i], Y_flat[i], 1-Y_flat[i])
    
    r_max_boundary = np.minimum(
        np.minimum(X_flat, 1 - X_flat),
        np.minimum(Y_flat, 1 - Y_flat)
    )
    
    # State representation:
    # centers: array of N integers (indices into the flattened grid)
    # We will track the current radii implicitly?
    # No, in this model, if we fix centers, the radii are determined by the constraints.
    # But we want to MAXIMIZE sum of radii.
    # If we fix centers, the optimal radii for each circle i is:
    # r_i = min(
    #     r_max_boundary[center_i],
    #     min_{j != i} (dist(center_i, center_j) - r_j)  <-- This is coupled!
    # )
    # This is a system of linear inequalities:
    # r_i + r_j <= d_ij
    # r_i <= b_i
    # r_i >= 0
    # Maximize sum(r_i)
    
    # Solving this LP for every step of SA is too slow.
    # Alternative:
    # We can approximate the radii.
    # Or, we can define the objective of SA as the sum of "potential radii".
    # Let's try a different angle:
    # Fix the centers.
    # Calculate the "tightest" constraint for each circle.
    # If we assume all circles have the same radius r, then 2r <= d_ij => r <= d_ij/2.
    # But we want to maximize SUM.
    
    # Let's use a heuristic for radii given centers:
    # 1. Start with r_i = r_max_boundary[center_i]
    # 2. Iterate: for each pair (i, j), if r_i + r_j > d_ij, reduce them proportionally or to the minimum.
    # This is a relaxation.
    
    def get_radii_for_centers(centers):
        """
        Given a set of grid indices, compute valid radii that maximize sum.
        Uses a simple iterative relaxation.
        """
        # Initial radii = boundary limits
        r = r_max_boundary[centers].copy()
        
        # Iterative push
        # To speed up, we can just do a few iterations
        # Or, we can solve the LP?
        # With N=26, LP is small. But calling scipy.optimize.linprog in SA loop is too slow.
        
        # Heuristic:
        # For each pair, if r_i + r_j > d_ij:
        #   excess = r_i + r_j - d_ij
        #   r_i -= excess / 2
        #   r_j -= excess / 2
        #   clamp to 0
        
        # This might oscillate.
        # Let's try a simpler approach:
        # r_i = min(b_i, min_{j} (d_ij - r_j)) is not directly solvable.
        
        # Let's use the "min cut" idea or just greedy reduction.
        # Since N is small, we can do a few passes of constraint enforcement.
        
        d_matrix_subset = dist_matrix[centers][:, centers]
        
        # Run 10 iterations of relaxation
        for _ in range(20):
            changed = False
            for i in range(N):
                # Check against boundaries
                if r[i] > r_max_boundary[centers[i]]:
                    r[i] = r_max_boundary[centers[i]]
                    changed = True
                # Check against other circles
                for j in range(N):
                    if i == j: continue
                    d = d_matrix_subset[i, j]
                    if r[i] + r[j] > d:
                        # Need to reduce
                        # Reduce the larger one more? Or split?
                        # Split is fair.
                        excess = r[i] + r[j] - d
                        # If one is 0, take all from other
                        if r[i] == 0:
                            r[j] -= excess
                        elif r[j] == 0:
                            r[i] -= excess
                        else:
                            # Reduce both
                            # But don't go below 0
                            # Let's just reduce r[i] by excess and see if r[j] needs adjustment later
                            # Better: reduce both by excess/2
                            half_excess = excess * 0.5
                            r[i] -= half_excess
                            r[j] -= half_excess
                        if r[i] < 0: r[i] = 0
                        if r[j] < 0: r[j] = 0
                        changed = True
            if not changed:
                break
        
        # Final validation/clamping
        # Ensure no negative radii
        r = np.maximum(r, 0)
        
        # One more pass to ensure strict validity for the score
        # If there are still violations, shrink further
        for i in range(N):
            for j in range(i+1, N):
                d = d_matrix_subset[i, j]
                if r[i] + r[j] > d:
                    # Shrink the one with larger radius? Or both?
                    # To be safe, shrink both until valid
                    # r_i + r_j = d
                    # Let's keep ratio?
                    # Simple: shrink r_i to max(0, d - r_j)
                    # But this might break r_j's other constraints.
                    # Since this is just for scoring/return, let's just ensure validity by shrinking.
                    # We want to maximize sum, so this is suboptimal if we shrink too much.
                    # But the relaxation above should have handled most of it.
                    # If we get here, it's a small violation.
                    # Let's just zero out the excess from the larger radius.
                    if r[i] > r[j]:
                        r[i] = max(0, d - r[j])
                    else:
                        r[j] = max(0, d - r[i])
        
        return r

    # Simulated Annealing
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Initial centers: random or grid-based
    # Let's start with a grid of 5x5 (25 circles) + 1 random
    # Or just random sampling from the grid
    
    # Precompute neighbors for fast move generation?
    # A move is changing one circle's center to a nearby grid point.
    # Neighbors: 8-connectivity in the grid.
    
    # Flatten grid indices
    # index = row * grid_res + col
    # Neighbors of (r, c): (r-1, c-1), (r-1, c), ..., (r+1, c+1)
    
    # To speed up SA, we can precompute the list of neighbors for each grid index.
    neighbors = {}
    for r in range(grid_res):
        for c in range(grid_res):
            idx = r * grid_res + c
            nbrs = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_res and 0 <= nc < grid_res:
                        nbrs.append(nr * grid_res + nc)
            neighbors[idx] = nbrs

    def score_centers(centers):
        radii = get_radii_for_centers(centers)
        return np.sum(radii), radii

    # Initialize centers
    # Try to place them in a grid pattern initially
    init_centers = []
    side = int(np.ceil(np.sqrt(N)))
    for i in range(N):
        row = i // side
        col = i % side
        # Map to grid indices
        # Spread them out
        g_row = int(row * grid_res / side)
        g_col = int(col * grid_res / side)
        # Clamp
        g_row = np.clip(g_row, 0, grid_res - 1)
        g_col = np.clip(g_col, 0, grid_res - 1)
        init_centers.append(g_row * grid_res + g_col)
    
    current_centers = np.array(init_centers, dtype=int)
    current_score, current_radii = score_centers(current_centers)
    
    if current_score > best_score:
        best_score = current_score
        best_centers = current_centers.copy()
        best_radii = current_radii.copy()
    
    T = 0.01 # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    step = 0
    while time.time() < deadline and T > T_min:
        if time.time() > deadline - 5:
            break
            
        # Pick a random circle
        i = np.random.randint(N)
        old_center = current_centers[i]
        
        # Pick a random neighbor
        nbrs = neighbors[old_center]
        if len(nbrs) == 0:
            T *= cooling
            continue
            
        new_center = nbrs[np.random.randint(len(nbrs))]
        
        # If the new center is already occupied by another circle, skip?
        # Or allow overlap and let radii become 0?
        # If centers are identical, dist=0, r_i+r_j <= 0 => r_i=r_j=0.
        # This is a valid (but bad) state.
        # However, if we allow identical centers, we might get stuck in "clusters" of zero radius.
        # Better to forbid identical centers?
        # If we forbid, we need to check occupancy.
        # Occupancy check: is new_center in current_centers (excluding i)?
        # This is O(N).
        if new_center in current_centers[np.arange(N) != i]:
            T *= cooling
            continue
            
        # Evaluate new state
        current_centers[i] = new_center
        new_score, new_radii = score_centers(current_centers)
        
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            current_radii = new_radii
            if new_score > best_score:
                best_score = new_score
                best_centers = current_centers.copy()
                best_radii = new_radii.copy()
        else:
            # Revert
            current_centers[i] = old_center
            
        T *= cooling
        
    # Final Phase: Local Optimization / "Inflation"
    # The SA found good centers.
    # Now we can try to improve the radii assignment?
    # The get_radii_for_centers is a heuristic.
    # We can try a more rigorous LP solve on the best centers found?
    # Or just use the heuristic result.
    
    # Let's try to polish the best solution found.
    # We can try to move centers to nearby grid points if it increases the "slack".
    # But the score is the sum of radii.
    
    # One more improvement: 
    # The heuristic `get_radii_for_centers` might be suboptimal.
    # Let's try a simple "expand" step:
    # Start with r_i = 0.
    # Increase all r_i uniformly until a constraint is hit.
    # Then increase the others?
    # This is complex.
    
    # Let's stick with the SA result.
    # To ensure validity, we do a final strict check and repair.
    
    final_centers = best_centers
    final_radii = best_radii
    
    # Convert grid indices to coordinates
    rows = final_centers // grid_res
    cols = final_centers % grid_res
    xs_final = (cols + 0.5) * cell_size
    ys_final = (rows + 0.5) * cell_size
    
    # Final validity check and repair
    # If any r_i + r_j > dist_ij, shrink.
    # We already did this in get_radii_for_centers, but let's double check.
    
    # Re-calculate radii from scratch using the best centers to ensure consistency
    final_radii = get_radii_for_centers(final_centers)
    
    # Check if valid
    valid = True
    for i in range(N):
        if xs_final[i] < final_radii[i] - 1e-9 or xs_final[i] > 1 - final_radii[i] + 1e-9:
            valid = False
        if ys_final[i] < final_radii[i] - 1e-9 or ys_final[i] > 1 - final_radii[i] + 1e-9:
            valid = False
            
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = xs_final[i] - xs_final[j]
                dy = ys_final[i] - ys_final[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < final_radii[i] + final_radii[j] - 1e-9:
                    valid = False
                    break
            if not valid: break
    
    if not valid:
        # Fallback: simple grid with small radii
        # 5x5 grid + 1
        side = 5
        xs_fallback = np.zeros(N)
        ys_fallback = np.zeros(N)
        rs_fallback = np.zeros(N)
        for i in range(N):
            r = i // side
            c = i % side
            xs_fallback[i] = (c + 0.5) / side
            ys_fallback[i] = (r + 0.5) / side
            rs_fallback[i] = 0.45 / side
            
        # Check fallback validity
        # If invalid, shrink further
        for _ in range(10):
            ok = True
            for i in range(N):
                if xs_fallback[i] < rs_fallback[i] or xs_fallback[i] > 1 - rs_fallback[i]:
                    ok = False
                if ys_fallback[i] < rs_fallback[i] or ys_fallback[i] > 1 - rs_fallback[i]:
                    ok = False
            if ok:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs_fallback[i] - xs_fallback[j]
                        dy = ys_fallback[i] - ys_fallback[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < rs_fallback[i] + rs_fallback[j]:
                            ok = False
                            break
                    if not ok: break
            
            if ok:
                break
            rs_fallback *= 0.9
            
        xs_final = xs_fallback
        ys_final = ys_fallback
        final_radii = rs_fallback

    return np.column_stack((xs_final, ys_final, final_radii))

# Dummy main to allow direct execution if needed, though construct is the entry point
if __name__ == "__main__":
    res = construct()
    print("Sum of radii:", np.sum(res[:, 2]))
