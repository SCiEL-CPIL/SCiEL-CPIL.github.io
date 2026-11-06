"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Place 200 equal circles in a unit square to maximize the minimum radius."""
    t_start = time.time()
    deadline = t_start + 110  # Safety margin

    def hex_init(n):
        """Create a hexagonal grid initialization."""
        # For N points in unit square, estimate spacing
        # Hexagonal packing density: each point occupies area ~ s^2 * sqrt(3)/2
        # N points in area 1: N * s^2 * sqrt(3)/2 = 1 => s = sqrt(2/(N*sqrt(3)))
        s = np.sqrt(2.0 / (n * np.sqrt(3)))
        pts = []
        y = -0.5 + s / 2
        row = 0
        while y < 0.5 and len(pts) < n:
            x_offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + x_offset
            while x < 0.5 and len(pts) < n:
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    pts.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        
        pts = np.array(pts)
        if len(pts) < n:
            remaining = n - len(pts)
            rand_pts = np.random.rand(remaining, 2) - 0.5
            pts = np.vstack([pts, rand_pts])
        return pts[:n]

    def get_min_radius(p):
        """Calculate the max possible radius for the current configuration."""
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist) / 2

    def calc_dists(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        return d

    def calc_wall_dists(p):
        return np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )

    # Start with hexagonal grid
    pts = hex_init(N).copy()
    current_r = get_min_radius(pts)
    best_r = current_r
    best_pts = pts.copy()

    # Precompute
    dist_matrix = calc_dists(pts)
    wall_dists = calc_wall_dists(pts)
    eye_mask = ~np.eye(N, dtype=bool)

    # Local search with repulsion forces
    # Strategy: For each point, compute repulsion from all other points and walls
    # Move point in direction of net force, accept if it improves the minimum radius
    
    max_steps = 50000
    step_size = 0.01
    no_improve = 0
    max_no_improve = 2000

    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random point
        i = np.random.randint(N)
        
        # Compute force on point i
        # Repulsion from other points: F = sum_j (1/d_ij^2) * (p_i - p_j) for d_ij < threshold
        # Repulsion from walls: push away from nearest wall
        
        # Get distances from i to all others
        d_i = dist_matrix[i, :].copy()
        d_i[i] = np.inf  # Exclude self
        
        # Only consider points within a certain range for efficiency
        # Use threshold based on current radius
        threshold = max(current_r * 4, 0.05)
        
        # Compute repulsion force from other points
        force = np.zeros(2)
        mask = d_i < threshold
        for j in range(N):
            if j == i or not mask[j]:
                continue
            d = d_i[j]
            if d < 1e-9:
                continue
            # Force magnitude: 1/d^2, direction: away from j
            mag = 1.0 / (d * d)
            force += mag * (pts[i] - pts[j])
        
        # Wall repulsion
        x, y = pts[i]
        d_right = 0.5 - x
        d_left = x + 0.5
        d_top = 0.5 - y
        d_bottom = y + 0.5
        
        min_wall = min(d_right, d_left, d_top, d_bottom)
        if min_wall < threshold:
            w_mag = 1.0 / max(min_wall * min_wall, 1e-12)
            if d_right == min_wall:
                force[0] -= w_mag
            if d_left == min_wall:
                force[0] += w_mag
            if d_top == min_wall:
                force[1] -= w_mag
            if d_bottom == min_wall:
                force[1] += w_mag
        
        # Normalize force and take step
        force_norm = np.linalg.norm(force)
        if force_norm < 1e-12:
            no_improve += 1
            continue
        
        direction = force / force_norm
        new_pos = pts[i] + step_size * direction
        
        # Check bounds
        if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
            # Try smaller step
            new_pos = pts[i] + 0.5 * step_size * direction
            if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
                no_improve += 1
                continue
        
        # Evaluate new configuration
        old_pos = pts[i].copy()
        pts[i] = new_pos
        
        # Update distances for point i
        new_dists_i = np.sqrt(np.sum((pts - new_pos)**2, axis=1))
        new_wall_i = np.minimum(
            np.minimum(0.5 - new_pos[0], new_pos[0] + 0.5),
            np.minimum(0.5 - new_pos[1], new_pos[1] + 0.5)
        )
        
        # Update dist matrix
        dist_matrix[i, :] = new_dists_i
        dist_matrix[:, i] = new_dists_i
        dist_matrix[i, i] = 0
        wall_dists[i] = new_wall_i
        
        # Calculate new min radius
        min_pair = np.min(dist_matrix[eye_mask])
        min_wall = np.min(wall_dists)
        candidate_r = min(min_pair, min_wall) / 2
        
        if candidate_r > current_r:
            current_r = candidate_r
            no_improve = 0
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            # Revert
            pts[i] = old_pos
            dist_matrix[i, :] = np.sqrt(np.sum((pts - old_pos)**2, axis=1))
            dist_matrix[:, i] = dist_matrix[i, :]
            dist_matrix[i, i] = 0
            wall_dists[i] = np.minimum(
                np.minimum(0.5 - old_pos[0], old_pos[0] + 0.5),
                np.minimum(0.5 - old_pos[1], old_pos[1] + 0.5)
            )
            no_improve += 1
        
        # If stuck for too long, try a random perturbation
        if no_improve > max_no_improve:
            # Random restart with small noise
            pts += np.random.randn(N, 2) * 0.01
            pts = np.clip(pts, -0.49, 0.49)
            dist_matrix = calc_dists(pts)
            wall_dists = calc_wall_dists(pts)
            current_r = min(np.min(dist_matrix[eye_mask]), np.min(wall_dists)) / 2
            no_improve = 0
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
            # Reduce step size when stuck
            step_size *= 0.9

    # Final verification
    final_r = get_min_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
        best_pts = best_pts.copy()

    return best_pts

# EVOLVE-BLOCK-END
