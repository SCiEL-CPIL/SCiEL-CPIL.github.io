import numpy as np
import time
from scipy.optimize import minimize

def construct():
    N = 200
    start_time = time.time()
    deadline = 110.0  # Safety margin

    def compute_score(centers):
        if not np.all(np.isfinite(centers)):
            return -np.inf
        # Check bounds
        if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
            return -np.inf
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)

    def hex_init(N):
        # Approximate radius for 200 circles in square ~ 0.036
        r_target = 0.035
        dx = 2 * r_target
        dy = np.sqrt(3) * r_target
        
        points = []
        y = -0.5 + r_target
        row = 0
        while y <= 0.5 - r_target + 1e-6:
            x_offset = 0 if row % 2 == 0 else dx / 2
            x = -0.5 + r_target + x_offset
            while x <= 0.5 - r_target + 1e-6:
                points.append((x, y))
                x += dx
            y += dy
            row += 1
            
        points = np.array(points)
        
        if len(points) >= N:
            # Select N points closest to center to maximize symmetry/packing
            dists_to_center = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists_to_center)
            return points[sorted_indices[:N]]
        else:
            # If not enough, add random points in the valid region
            remaining = N - len(points)
            random_points = np.random.uniform(-0.45, 0.45, (remaining, 2))
            return np.vstack([points, random_points])

    def grid_init(N):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def anneal(centers, max_iters=5000, time_limit=None):
        """
        Physics-based annealing: circles as charged particles with repulsive forces
        and boundary constraints.
        """
        centers = centers.copy()
        N = len(centers)
        
        # Parameters
        k_repel = 1.0       # Repulsion strength
        k_boundary = 50.0   # Boundary constraint strength
        damping = 0.95      # Velocity damping
        dt = 0.005          # Time step
        temp = 0.01         # Initial temperature for random perturbations
        
        best_centers = centers.copy()
        best_score = compute_score(centers)
        
        velocities = np.zeros_like(centers)
        
        for it in range(max_iters):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion (only for close pairs to save time)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Only consider pairs closer than some threshold
            threshold = 0.15
            close_mask = dists < threshold
            np.fill_diagonal(close_mask, False)
            
            if np.any(close_mask):
                # For close pairs, compute repulsive force
                for i in range(N):
                    for j in range(N):
                        if close_mask[i, j] and i < j:
                            d = dists[i, j]
                            if d > 1e-10:
                                # Force magnitude: k / d^2 (Coulomb-like)
                                force_mag = k_repel / (d * d)
                                # Direction: away from each other
                                direction = (centers[i] - centers[j]) / d
                                forces[i] += force_mag * direction
                                forces[j] -= force_mag * direction
            
            # Boundary forces: push circles away from walls
            for i in range(N):
                for dim in range(2):
                    pos = centers[i, dim]
                    if pos < -0.48:
                        # Push right (toward center)
                        dist_to_wall = -0.5 - pos
                        if dist_to_wall > 0:
                            forces[i, dim] += k_boundary * dist_to_wall * (-1)
                    elif pos > 0.48:
                        # Push left (toward center)
                        dist_to_wall = pos - 0.5
                        if dist_to_wall > 0:
                            forces[i, dim] += k_boundary * dist_to_wall * (-1)
            
            # Update velocities and positions
            velocities = damping * velocities + forces * dt
            centers += velocities
            
            # Keep centers within bounds
            centers = np.clip(centers, -0.5, 0.5)
            
            # Add random perturbation (annealing)
            if temp > 1e-6:
                centers += np.random.normal(0, temp, centers.shape)
                centers = np.clip(centers, -0.5, 0.5)
                temp *= 0.999  # Cool down
            
            # Check score periodically
            if it % 10 == 0:
                score = compute_score(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
        
        return best_centers, best_score

    def energy_and_grad(x, p=4.0, eps=1e-6):
        """
        Minimize the sum of inverse powers of distances.
        """
        centers = x.reshape(N, 2)
        
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Clip distances to avoid division by zero/inf
        dists_clipped = np.maximum(dists, eps)
        
        # Energy term: sum(d^-p)
        energy_pairs = np.sum(np.power(dists_clipped, -p))
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        wall_dists_clipped = np.maximum(wall_dists, eps)
        energy_walls = np.sum(np.power(wall_dists_clipped, -p))
        
        total_energy = energy_pairs + energy_walls
        
        # Gradient
        grads = np.zeros_like(centers)
        
        # Pairwise gradient
        weights = -p * np.power(dists_clipped, -p - 2)
        np.fill_diagonal(weights, 0.0)
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        grads += centers * sum_weights[:, np.newaxis] - weighted_sum_x
        
        # Wall gradient
        d_left = 0.5 + centers[:, 0]
        d_right = 0.5 - centers[:, 0]
        d_bottom = 0.5 + centers[:, 1]
        d_top = 0.5 - centers[:, 1]
        
        for i in range(N):
            w_val = wall_dists_clipped[i]
            coeff = -p * np.power(w_val, -p - 1)
            
            gx = 0.0
            gy = 0.0
            
            if np.isclose(d_left[i], w_val, rtol=1e-9, atol=1e-12):
                gx += 1.0
            if np.isclose(d_right[i], w_val, rtol=1e-9, atol=1e-12):
                gx += -1.0
            if np.isclose(d_bottom[i], w_val, rtol=1e-9, atol=1e-12):
                gy += 1.0
            if np.isclose(d_top[i], w_val, rtol=1e-9, atol=1e-12):
                gy += -1.0
            
            grads[i, 0] += coeff * gx
            grads[i, 1] += coeff * gy

        return total_energy, grads.reshape(-1)

    best_centers = None
    best_score = -np.inf
    
    # Try multiple initializations and combine annealing with L-BFGS-B
    inits = [
        hex_init(N),
        grid_init(N),
        np.random.uniform(-0.45, 0.45, (N, 2))
    ]
    
    for init in inits:
        if time.time() - start_time > deadline - 30:
            break
        
        # First, run annealing to get a good starting point
        annealed, anneal_score = anneal(init, max_iters=3000, time_limit=deadline - 20)
        
        if anneal_score > best_score:
            best_score = anneal_score
            best_centers = annealed.copy()
        
        # Then polish with L-BFGS-B
        if time.time() - start_time < deadline - 10:
            try:
                x0 = annealed.flatten()
                bounds = [(-0.5, 0.5)] * (2 * N)
                
                result = minimize(
                    energy_and_grad, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if time.time() - start_time < deadline - 5:
                    new_centers = result.x.reshape(N, 2)
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    new_score = compute_score(new_centers)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
            except Exception as e:
                pass
    
    # If no valid result, fallback to grid
    if best_centers is None:
        best_centers = grid_init(N)
        best_score = compute_score(best_centers)
        
    return best_centers
