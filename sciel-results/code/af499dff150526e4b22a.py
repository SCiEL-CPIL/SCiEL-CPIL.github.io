import numpy as np
import time
from scipy.spatial import cKDTree

N = 200

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin

    def hex_grid_init(n):
        """Initialize with a hexagonal lattice pattern."""
        pts = []
        # Approximate hex spacing
        # Area per point ~ 1/n. Hex cell area ~ s^2 * sqrt(3)/2.
        s = np.sqrt(2.0 / (n * np.sqrt(3)))
        
        y = -0.5 + s/2
        row = 0
        while y < 0.5 and len(pts) < n:
            x_offset = (row % 2) * s / 2
            x = -0.5 + s/2 + x_offset
            while x < 0.5 and len(pts) < n:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
            
        if len(pts) < n:
            # Fill remaining with random points if hex grid is too sparse
            remaining = n - len(pts)
            rand_pts = np.random.rand(remaining, 2) - 0.5
            pts.extend(rand_pts.tolist())
            
        return np.array(pts[:n])

    def calc_radius(p):
        """Calculate the maximum common radius for a given configuration."""
        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(p)
        # Query for 2 nearest neighbors (self + nearest)
        dists, _ = tree.query(p, k=2)
        min_pair = np.min(dists[:, 1])
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall) / 2

    # 1. Initialization
    pts = hex_grid_init(N)
    pts = np.clip(pts, -0.49, 0.49)
    
    current_r = calc_radius(pts)
    best_r = current_r
    best_pts = pts.copy()

    # 2. Optimization: Simulated Annealing with KDTree acceleration
    # We use a hybrid approach:
    # - Random moves with acceptance criterion based on energy change (approximated by radius change)
    # - Occasionally, a "relaxation" step using gradient descent to settle local minima
    
    T = 0.01
    cooling = 0.9995
    min_T = 1e-6
    steps = 0
    no_improve_count = 0
    max_no_improve = 500

    while time.time() < deadline - 2.0:
        if T < min_T:
            # Restart with higher temperature if stuck
            if no_improve_count > max_no_improve:
                T = 0.01
                # Perturb points to escape local minima
                pts += np.random.randn(N, 2) * 0.02
                pts = np.clip(pts, -0.49, 0.49)
                no_improve_count = 0
            else:
                break
        
        # Pick a random point
        i = np.random.randint(N)
        
        # Propose a move
        # Step size scales with temperature
        delta = np.random.randn(2) * T
        old_pos = pts[i].copy()
        new_pos = old_pos + delta
        
        # Check bounds
        if new_pos[0] < -0.49 or new_pos[0] > 0.49 or new_pos[1] < -0.49 or new_pos[1] > 0.49:
            T *= cooling
            continue
            
        # Calculate change in radius efficiently
        # We only need to check distances involving point i
        # 1. Distance to other points
        dists_to_i = np.linalg.norm(pts - new_pos, axis=1)
        dists_to_i[i] = np.inf
        min_pair_new = np.min(dists_to_i)
        
        # 2. Wall distance
        wall_dist_new = np.minimum(
            np.minimum(0.5 - new_pos[0], new_pos[0] + 0.5),
            np.minimum(0.5 - new_pos[1], new_pos[1] + 0.5)
        )
        
        # The new global radius is min(global_min_excluding_i, min_pair_new, wall_dist_new) / 2
        # But we don't have global_min_excluding_i easily.
        # Instead, let's just compute the full radius if the move looks promising.
        # A move is promising if min_pair_new and wall_dist_new are not significantly worse than current_r*2.
        
        # Approximate check: if the new point's constraints are worse than current radius, reject immediately
        if min(min_pair_new, wall_dist_new) < 2 * current_r * 0.9:
            T *= cooling
            continue
            
        # Accept if it improves or with probability exp(-delta_E/T)
        # For simplicity, accept if it doesn't degrade too much
        pts[i] = new_pos
        
        # Calculate new radius
        new_r = calc_radius(pts)
        
        if new_r > current_r:
            current_r = new_r
            no_improve_count = 0
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            # Accept with probability exp(-(current_r - new_r) / T)
            # But since we want to maximize r, the "energy" is -r
            # delta_E = -new_r - (-current_r) = current_r - new_r
            # Accept if exp(-delta_E/T) > random
            if np.random.rand() < np.exp(-(current_r - new_r) / T):
                pass # Accepted, keep new position
            else:
                pts[i] = old_pos # Revert
                no_improve_count += 1
                
        T *= cooling
        steps += 1
        
        # Every 100 steps, do a gradient relaxation step
        if steps % 100 == 0 and time.time() < deadline - 5.0:
            # Gradient Descent Step
            eta = 0.005
            for _ in range(10):
                # Calculate forces
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
                dists_sq = np.sum(diff**2, axis=2)
                dists_sq[dists_sq < 1e-12] = 1e-12
                W = 1.0 / (dists_sq * dists_sq)
                np.fill_diagonal(W, 0)
                
                row_sums = W.sum(axis=1)
                F_pairs = row_sums[:, np.newaxis] * pts - W.dot(pts)
                
                F_walls = np.zeros_like(pts)
                for i in range(N):
                    dx_r = 0.5 - pts[i, 0]
                    dx_l = pts[i, 0] + 0.5
                    dy_t = 0.5 - pts[i, 1]
                    dy_b = pts[i, 1] + 0.5
                    min_d = min(dx_r, dx_l, dy_t, dy_b)
                    
                    grad_x = 0
                    grad_y = 0
                    if dx_r == min_d:
                        grad_x += 2.0 / (dx_r**3)
                    if dx_l == min_d:
                        grad_x -= 2.0 / (dx_l**3)
                    if dy_t == min_d:
                        grad_y += 2.0 / (dy_t**3)
                    if dy_b == min_d:
                        grad_y -= 2.0 / (dy_b**3)
                        
                    F_walls[i, 0] = grad_x
                    F_walls[i, 1] = grad_y
                    
                pts += eta * (F_pairs + F_walls)
                pts = np.clip(pts, -0.49, 0.49)
                eta *= 0.9

            # Check improvement after relaxation
            relaxed_r = calc_radius(pts)
            if relaxed_r > current_r:
                current_r = relaxed_r
                if current_r > best_r:
                    best_r = current_r
                    best_pts = pts.copy()
                no_improve_count = 0
            else:
                # Revert if relaxation made it worse
                pts = best_pts.copy()
                current_r = best_r

    return best_pts
