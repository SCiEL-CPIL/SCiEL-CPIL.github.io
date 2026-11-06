"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Place 200 equal circles in a unit square to maximize the minimum radius."""
    t_start = time.time()
    deadline = t_start + 145  # Safety margin before 150s kill

    # 1. Initialization: Hexagonal grid is a better starting point than square
    def hex_init(n):
        # Approximate hex packing in a square
        # Spacing s. Area per point ~ s^2 * sqrt(3)/2.
        # N points in area 1. s ~ sqrt(2 / (N * sqrt(3)))
        s = np.sqrt(2.0 / (N * np.sqrt(3)))
        pts = []
        y = -0.5 + s / 2
        row = 0
        while y < 0.5 and len(pts) < n:
            x_offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + x_offset
            while x < 0.5 and len(pts) < n:
                # Check if inside bounds
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        
        pts = np.array(pts)
        if len(pts) < n:
            # Fill remaining with random or grid
            remaining = n - len(pts)
            # Just append random points inside
            rand_pts = np.random.rand(remaining, 2) - 0.5
            pts = np.vstack([pts, rand_pts])
        return pts[:n]

    pts = hex_init(N).copy()
    # Clip to be strictly inside for safety
    pts = np.clip(pts, -0.49, 0.49)

    def get_min_radius(p):
        """Calculate the max possible radius for the current configuration."""
        if len(p) < 2:
            return np.inf
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist) / 2

    current_r = get_min_radius(pts)
    best_r = current_r
    best_pts = pts.copy()

    # Precompute distance matrix and wall distances for fast updates
    def calc_dists(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        return d

    def calc_wall_dists(p):
        return np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )

    dist_matrix = calc_dists(pts)
    wall_dists = calc_wall_dists(pts)
    
    # Mask for off-diagonal elements to find min pair dist
    mask = ~np.eye(N, dtype=bool)
    
    # Phase 1: Global Gradient Ascent on "Soft" Objective
    # Objective: sum of min(d_ij, target) + sum of min(wall_i, target)
    # We want to maximize the minimum distance.
    # Using a soft penalty for violations allows global movement.
    
    # We will perform gradient steps where we move points away from their nearest neighbors/walls.
    # This is effectively a force-directed layout where forces are repulsive when too close.
    
    T = 0.01
    cooling = 0.999
    min_T = 1e-6
    
    # Gradient Step Helper
    def gradient_step(pts, dist_matrix, wall_dists, step_size):
        """
        Compute gradient of the 'soft' objective.
        Objective J = sum_{i<j} min(d_ij, R_target) + sum_i min(wall_i, R_target)
        We want to maximize J.
        dJ/dp_i = sum_{j != i, d_ij < R_target} (p_i - p_j) / d_ij + wall_grad_i (if wall_i < R_target)
        """
        # We don't know R_target exactly, but we can use current best_r * 2 as a proxy for "too close"
        # Or simply use a fixed small threshold like 0.1 * side.
        # Let's use current_r * 2 as the threshold for "active" constraints.
        threshold = max(current_r * 2, 0.01)
        
        grad = np.zeros_like(pts)
        
        # Pairwise forces
        # Only consider pairs where dist < threshold
        # Vectorized:
        # dist_matrix is NxN.
        # active = dist_matrix < threshold
        # active.fill_diagonal(False)
        # 
        # For each i, grad[i] += sum_{j active} (pts[i] - pts[j]) / dist_matrix[i,j]
        
        # Create a weight matrix
        # W[i,j] = 1/d_ij if d_ij < threshold else 0
        # grad[i] = sum_j W[i,j] * (pts[i] - pts[j])
        #          = (sum_j W[i,j]) * pts[i] - sum_j W[i,j] * pts[j]
        
        W = np.zeros((N, N))
        active = (dist_matrix < threshold) & (dist_matrix > 1e-9)
        np.fill_diagonal(active, False)
        
        # Avoid division by zero
        # W = np.where(active, 1.0 / dist_matrix, 0.0)
        # This creates a full NxN array.
        
        # Let's do it efficiently:
        # W = np.zeros_like(dist_matrix)
        # np.divide(1.0, dist_matrix, out=W, where=active)
        
        W = np.zeros_like(dist_matrix)
        np.divide(1.0, dist_matrix, out=W, where=active)
        
        # row_sums = W.sum(axis=1)
        # grad = row_sums[:, np.newaxis] * pts - W.dot(pts)
        
        row_sums = W.sum(axis=1)
        grad_pairs = row_sums[:, np.newaxis] * pts - W.dot(pts)
        
        # Wall forces
        # wall_dists[i] = min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
        # If wall_dists[i] < threshold, push away from the closest wall.
        
        # Calculate which wall is closest for each point
        # dx_pos = 0.5 - x, dx_neg = x + 0.5
        # dy_pos = 0.5 - y, dy_neg = y + 0.5
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_right = 0.5 - x
        d_left = x + 0.5
        d_top = 0.5 - y
        d_bottom = y + 0.5
        
        # Gradient contribution from walls
        grad_walls = np.zeros_like(pts)
        
        # If d_right < threshold and d_right is the min, push left (negative x)
        # Actually, if we are too close to right wall (x large), we want to decrease x.
        # Force is in direction of increasing wall distance.
        # If d_right is the limiting factor, grad_x should be -1 (move left).
        # If d_left is limiting, grad_x should be +1 (move right).
        
        # Let's just check if any wall is < threshold
        wall_active = wall_dists < threshold
        
        for i in np.where(wall_active)[0]:
            # Find min wall dist
            d_r = 0.5 - pts[i, 0]
            d_l = pts[i, 0] + 0.5
            d_t = 0.5 - pts[i, 1]
            d_b = pts[i, 1] + 0.5
            
            min_d = min(d_r, d_l, d_t, d_b)
            
            # Weight is 1/min_d (inverse distance force)
            w = 1.0 / max(min_d, 1e-9)
            
            if d_r == min_d:
                grad_walls[i, 0] -= w
            if d_l == min_d:
                grad_walls[i, 0] += w
            if d_t == min_d:
                grad_walls[i, 1] -= w
            if d_b == min_d:
                grad_walls[i, 1] += w
                
        grad = grad_pairs + grad_walls
        
        # Normalize gradient?
        # The magnitude of force can vary.
        # Let's just use a fixed step size scaled by step_size param.
        
        new_pts = pts + step_size * grad
        
        # Clip to bounds
        new_pts = np.clip(new_pts, -0.49, 0.49)
        
        return new_pts

    # Phase 1: Gradient Descent/Ascent
    # We want to maximize radius, so we follow the gradient of the "soft" objective.
    # However, pure gradient ascent on the sum of mins can be noisy.
    # Let's combine with local search.
    
    # Actually, the "soft" objective gradient is useful for global rearrangement.
    # Let's run a few iterations of this.
    
    steps = 0
    # Use a larger step size initially for global movement
    step_size = 0.01
    
    while time.time() < deadline - 20:
        # Calculate gradient
        # We need current dist_matrix and wall_dists
        # They are up to date if we update them after each move.
        
        new_pts = gradient_step(pts, dist_matrix, wall_dists, step_size)
        
        # Evaluate new radius
        new_dist_matrix = calc_dists(new_pts)
        new_wall_dists = calc_wall_dists(new_pts)
        new_r = min(np.min(new_dist_matrix[mask]), np.min(new_wall_dists)) / 2
        
        if new_r > current_r:
            # Accept
            pts = new_pts
            dist_matrix = new_dist_matrix
            wall_dists = new_wall_dists
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
            # Reduce step size slightly
            step_size *= 0.99
        else:
            # Reject
            step_size *= 0.95
            # If step size is too small, switch to local search
            if step_size < 1e-5:
                break
                
        steps += 1

    # Phase 2: Local Search (Hill Climbing) for fine-tuning
    # Now that we are in a good basin, do local moves.
    T = 0.005
    cooling = 0.999
    min_T = 1e-6
    
    # Reinitialize dist_matrix and wall_dists to be sure
    dist_matrix = calc_dists(pts)
    wall_dists = calc_wall_dists(pts)
    current_r = min(np.min(dist_matrix[mask]), np.min(wall_dists)) / 2
    
    while time.time() < deadline - 5:
        if T < min_T:
            # Restart with higher T? Or break.
            # Let's try a few restarts if time permits
            if steps > 5000:
                break
            else:
                T = 0.005
                # Randomize slightly
                pts += np.random.randn(N, 2) * 0.005
                pts = np.clip(pts, -0.49, 0.49)
                dist_matrix = calc_dists(pts)
                wall_dists = calc_wall_dists(pts)
                current_r = min(np.min(dist_matrix[mask]), np.min(wall_dists)) / 2
                if current_r > best_r:
                    best_r = current_r
                    best_pts = pts.copy()
        
        # Pick a random point
        i = np.random.randint(N)
        
        # Propose move
        delta = np.random.randn(2) * T
        old_pos = pts[i].copy()
        new_pos = old_pos + delta
        
        # Check bounds
        if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
            T *= cooling
            continue
            
        # Calculate new distances for point i
        new_dists_i = np.sqrt(np.sum((pts - new_pos)**2, axis=1))
        new_wall_i = np.minimum(
            np.minimum(0.5 - new_pos[0], new_pos[0] + 0.5),
            np.minimum(0.5 - new_pos[1], new_pos[1] + 0.5)
        )
        
        # Update dist_matrix and wall_dists temporarily
        old_dists_i = dist_matrix[i, :].copy()
        old_wall_i = wall_dists[i]
        
        dist_matrix[i, :] = new_dists_i
        dist_matrix[:, i] = new_dists_i
        dist_matrix[i, i] = 0
        wall_dists[i] = new_wall_i
        
        # Calculate new global min radius
        min_pair = np.min(dist_matrix[mask])
        min_wall = np.min(wall_dists)
        new_r = min(min_pair, min_wall) / 2
        
        if new_r > current_r:
            # Accept
            pts[i] = new_pos
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
            T *= cooling
        else:
            # Reject
            pts[i] = old_pos
            dist_matrix[i, :] = old_dists_i
            dist_matrix[:, i] = old_dists_i
            dist_matrix[i, i] = 0
            wall_dists[i] = old_wall_i
            T *= cooling
            
        steps += 1

    return best_pts

# EVOLVE-BLOCK-END
