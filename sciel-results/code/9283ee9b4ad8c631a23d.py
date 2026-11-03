import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        if len(centers) < 2:
            return min_wall
            
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def repulsive_energy(centers, k=2):
        """
        Compute a repulsive energy that encourages spreading.
        We want to maximize this.
        Using inverse squared distance for strong local repulsion.
        To avoid numerical instability with very close points, we add a small epsilon.
        """
        # Efficiently compute pairwise distances
        # For N=200, O(N^2) is fine (40k ops)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Set diagonal to 1 to avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Inverse squared distance
        # We want to maximize sum of 1/d^2
        # This is equivalent to minimizing sum of d^2, but 1/d^2 is more sensitive to close pairs
        # Let's use a softer potential: -sum(1/d^2) ? No, we want to MAXIMIZE repulsion.
        # So objective = sum(1/d^2). Gradient points in direction to increase distances.
        
        # To prevent explosion if d->0, clip d^2
        dist_sq_clipped = np.maximum(dist_sq, 1e-10)
        
        energy = np.sum(1.0 / dist_sq_clipped)
        
        # Compute gradient of energy w.r.t centers
        # d/dx_i (1/||x_i - x_j||^2) = -2 (x_i - x_j) / ||x_i - x_j||^4
        # We want to move in direction of gradient ascent: + grad
        # So force on i from j is: 2 * (x_i - x_j) / d^4
        # Wait, if we maximize E = sum 1/d^2, grad E points to increase 1/d^2, which means decrease d.
        # That's wrong. We want to INCREASE distances.
        # So we should MINIMIZE sum(1/d^2) or MAXIMIZE sum(d^2).
        # Maximizing sum(d^2) is a standard "spread out" objective.
        # Let's use E = sum(d^2). Gradient is linear.
        # But sum(d^2) is not very sensitive to the *minimum* distance.
        # A better objective for packing is to maximize the minimum distance.
        # Since that's non-smooth, let's use a soft version or a potential that penalizes close pairs heavily.
        
        # Let's try: Minimize V = sum(1/d^2). 
        # Gradient descent on V will push points apart.
        # dV/dx_i = sum_{j!=i} -2 (x_i - x_j) / d_ij^4
        # Step: x_i -= lr * dV/dx_i = x_i += lr * 2 * sum (x_i - x_j) / d_ij^4
        # This is a repulsive force.
        
        # Let's compute the force directly
        force = np.zeros_like(centers)
        # dist_sq is N x N
        # force_i = sum_j 2 * (x_i - x_j) / dist_sq_ij^2
        # Note: dist_sq_ij^2 is (d^2)^2 = d^4
        inv_d4 = 1.0 / (dist_sq_clipped ** 2)
        # Zero out diagonal
        np.fill_diagonal(inv_d4, 0.0)
        
        # force_i = 2 * sum_j (x_i - x_j) * inv_d4_ij
        # (x_i - x_j) is diff[i, j]
        # force = 2 * sum_j diff[i, j] * inv_d4[i, j]
        # diff is N x N x 2
        # inv_d4 is N x N
        # We can do: force = 2 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        force = 2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        return -energy, force # Return negative energy if we want to maximize, or just return force for gradient descent on V.
                               # Let's just return the force for repulsion (gradient ascent on distance, or descent on 1/d^2)

    def force_directed_optimization(centers, max_iters=5000, lr=0.001, momentum=0.9):
        """
        Optimize using force-directed repulsion.
        """
        centers = centers.copy()
        v = np.zeros_like(centers) # velocity for momentum
        
        best_r = compute_radius(centers)
        best_centers = centers.copy()
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute repulsive force
            # We want to move points apart.
            # Using the force derived from minimizing sum(1/d^2)
            # Force on i is sum_j 2 * (x_i - x_j) / d_ij^4
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            dist_sq_clipped = np.maximum(dist_sq, 1e-10)
            
            inv_d4 = 1.0 / (dist_sq_clipped ** 2)
            np.fill_diagonal(inv_d4, 0.0)
            
            force = 2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
            
            # Normalize force to prevent huge steps when points are very close
            force_norms = np.linalg.norm(force, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            # Cap the force magnitude?
            # The force scales as 1/d^3. If d is small, force is huge.
            # Let's normalize by the max force magnitude in the system to keep steps stable
            max_force = np.max(force_norms)
            if max_force > 0:
                force = force / max_force
                
            # Update velocity with momentum
            v = momentum * v + lr * force
            
            # Update positions
            new_centers = centers + v
            
            # Project back to square [-0.5, 0.5]
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            # Check radius
            new_r = compute_radius(new_centers)
            
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centers = new_centers.copy()
                no_improve_count = 0
                # Decay learning rate slightly on improvement?
                # lr *= 0.999
            else:
                no_improve_count += 1
                # If stuck, try a larger step or random perturbation
                if no_improve_count % 100 == 0:
                    # Random perturbation
                    centers += np.random.randn(*centers.shape) * 0.01
                    centers = np.clip(centers, -0.5, 0.5)
                    v = np.zeros_like(centers)
                    # Reset learning rate
                    lr = 0.005
                    
            centers = new_centers
            
            # Adaptive learning rate
            if no_improve_count > 100:
                lr *= 0.9
            elif no_improve_count == 0:
                lr = min(lr * 1.01, 0.01)

        return best_centers, best_r

    def hexagonal_initialization(n_rows, n_cols):
        """Create a hexagonal packing layout."""
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
            
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
                
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
                
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
            
        return pts

    best_centers = None
    best_radius = 0.0

    # Generate diverse initial configurations
    configs = []
    
    # Hexagonal
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None:
            configs.append(pts)
            
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    configs.append(grid[:N])
    
    # Random
    configs.append(np.random.uniform(-0.4, 0.4, (N, 2)))

    # Optimize each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 20:
            break
            
        # Evaluate initial
        r = compute_radius(init_pts)
        if r > best_radius:
            best_radius = r
            best_centers = init_pts.copy()
            
        # Force Directed Optimization
        if time.time() < deadline - 10:
            try:
                fd_centers, fd_r = force_directed_optimization(init_pts)
                if fd_r > best_radius:
                    best_radius = fd_r
                    best_centers = fd_centers.copy()
            except Exception:
                pass

    # Final polish: If we have time, run a simple local search on the best result
    # to fix any minor overlaps or wall violations that might have slipped through
    # or to squeeze out more radius.
    if best_centers is not None and time.time() < deadline - 5:
        # Simple local search: move points to increase min distance
        # This is similar to the previous program's DLS but simpler
        centers = best_centers.copy()
        current_r = compute_radius(centers)
        
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            min_idx = np.argmin(dists[:, 1])
            min_pair_dist = dists[min_idx, 1]
            
            # Check walls
            wall_dists = 0.5 - np.abs(centers)
            min_wall_idx = np.argmin(wall_dists)
            min_wall_dist = wall_dists[min_wall_idx]
            
            if min_pair_dist / 2.0 < min_wall_dist:
                # Move pair apart
                i = min_idx
                j = _ # Need to get j from tree query? 
                # tree.query returns indices. Let's re-query properly.
                _, idxs = tree.query(centers, k=2)
                i = min_idx
                j = idxs[min_idx, 1]
                
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-9:
                    u = diff / dist
                    step = 0.001
                    centers[i] += u * step
                    centers[j] -= u * step
            else:
                # Move point away from wall
                i = min_wall_idx
                # Determine wall
                if centers[i, 0] < 0:
                    centers[i, 0] += 0.001
                elif centers[i, 0] > 0:
                    centers[i, 0] -= 0.001
                elif centers[i, 1] < 0:
                    centers[i, 1] += 0.001
                else:
                    centers[i, 1] -= 0.001
            
            centers = np.clip(centers, -0.5, 0.5)
            new_r = compute_radius(centers)
            if new_r > current_r:
                current_r = new_r
                best_radius = current_r
                best_centers = centers.copy()
            else:
                # Revert if worse? Or just accept if not much worse?
                # For safety, revert if significantly worse
                if new_r < current_r - 1e-5:
                    # Revert last move is complex here. Just break or continue.
                    pass

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
