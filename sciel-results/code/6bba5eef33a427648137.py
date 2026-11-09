import numpy as np
import time
from scipy.optimize import minimize

def construct():
    N = 200
    start_time = time.time()
    max_time = 115.0  # 5s safety margin for 120s limit

    def compute_score(centers):
        """Compute the exact score (largest common radius)."""
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            return 0.0
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)

    def neg_score(centers):
        """Negative score for minimization."""
        return -compute_score(centers)

    def soft_min_score(centers, k=50.0):
        """
        Smooth approximation of the score using log-sum-exp.
        Returns negative soft-min for minimization.
        """
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        
        # Collect all constraint values
        valid_dists = dists[~np.isinf(dists)]
        all_vals = np.concatenate([valid_dists, wall_dists])
        
        # Soft min: -1/k * log(sum(exp(-k * x)))
        # Use log-sum-exp trick for stability
        k_val = k
        scaled = -k_val * all_vals
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -log_sum / k_val
        
        return -soft_min_val  # Negative for minimization

    def soft_min_grad(centers, k=50.0):
        """
        Gradient of the negative soft-min score.
        """
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        
        # Weights for pairwise terms
        pair_weights = np.exp(-k * dists)
        np.fill_diagonal(pair_weights, 0)
        
        # Weights for wall terms
        wall_weights = np.exp(-k * wall_dists)
        
        Z = np.sum(pair_weights) + np.sum(wall_weights)
        if Z < 1e-100:
            return np.zeros_like(centers)
        
        pair_weights_norm = pair_weights / Z
        wall_weights_norm = wall_weights / Z
        
        grads = np.zeros_like(centers)
        
        # Pairwise gradient contribution
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = diff / dists[:, :, np.newaxis]
        np.fill_diagonal(unit_vecs[:, :, 0], 0)
        np.fill_diagonal(unit_vecs[:, :, 1], 0)
        unit_vecs = np.nan_to_num(unit_vecs)
        
        weighted_vecs = pair_weights_norm[:, :, np.newaxis] * unit_vecs
        grads -= np.sum(weighted_vecs, axis=1)  # Negative because we minimize -soft_min
        
        # Wall gradient contribution
        for idx in range(N):
            x, y = centers[idx]
            dx = 0.5 - abs(x)
            dy = 0.5 - abs(y)
            
            if dx < dy:
                gx = -np.sign(x) if x != 0 else 0
                gy = 0
            elif dy < dx:
                gx = 0
                gy = -np.sign(y) if y != 0 else 0
            else:
                gx = -np.sign(x) if x != 0 else 0
                gy = -np.sign(y) if y != 0 else 0
                norm = np.sqrt(gx*gx + gy*gy)
                if norm > 0:
                    gx /= norm
                    gy /= norm
            
            grads[idx, 0] -= wall_weights_norm[idx] * gx
            grads[idx, 1] -= wall_weights_norm[idx] * gy
        
        return grads

    def hex_init(N, r_target=0.036):
        """Generate hexagonal lattice initialization."""
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
            # Select N points closest to center
            dists_to_center = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists_to_center)
            return points[sorted_indices[:N]]
        else:
            # Pad with random points
            remaining = N - len(points)
            random_points = np.random.uniform(-0.4, 0.4, (remaining, 2))
            return np.vstack([points, random_points])

    def grid_init(N):
        """Generate square grid initialization."""
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def repulsive_relaxation(centers, max_iters=500, force_strength=0.1, time_limit=None):
        """
        Repulsive force-based relaxation to push circles apart.
        """
        centers = centers.copy()
        
        for i in range(max_iters):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            
            # Compute all pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Compute repulsive forces (inverse square law, clipped)
            with np.errstate(divide='ignore', invalid='ignore'):
                forces = diff / (dists[:, :, np.newaxis] ** 3 + 1e-10)
            np.fill_diagonal(forces[:, :, 0], 0)
            np.fill_diagonal(forces[:, :, 1], 0)
            forces = np.nan_to_num(forces)
            
            net_forces = np.sum(forces, axis=1)
            
            # Add wall repulsion
            for idx in range(N):
                x, y = centers[idx]
                # Distance to each wall
                dl = x + 0.5
                dr = 0.5 - x
                db = y + 0.5
                dt = 0.5 - y
                
                # Repulsive force from walls (proportional to 1/d^2)
                fx = 0
                fy = 0
                if dl < 0.1:
                    fx += force_strength / (dl * dl + 1e-6)
                if dr < 0.1:
                    fx -= force_strength / (dr * dr + 1e-6)
                if db < 0.1:
                    fy += force_strength / (db * db + 1e-6)
                if dt < 0.1:
                    fy -= force_strength / (dt * dt + 1e-6)
                
                net_forces[idx, 0] += fx
                net_forces[idx, 1] += fy
            
            # Update positions with damping
            step_size = 0.001
            centers += step_size * net_forces
            centers = np.clip(centers, -0.499, 0.499)
        
        return centers

    # Generate initial configurations
    inits = [
        hex_init(N),
        hex_init(N, r_target=0.035),
        grid_init(N),
        np.random.uniform(-0.4, 0.4, (N, 2))
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() - start_time > max_time:
            break
        
        # Phase 1: Repulsive relaxation
        c1 = repulsive_relaxation(init, max_iters=300, force_strength=0.1, time_limit=max_time - 5)
        
        # Phase 2: Gradient-based optimization using L-BFGS-B
        try:
            result = minimize(
                soft_min_score,
                c1,
                jac=soft_min_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            c2 = result.x
            s2 = compute_score(c2)
        except Exception:
            c2 = c1
            s2 = compute_score(c2)
        
        if s2 > best_score:
            best_score = s2
            best_centers = c2.copy()
    
    # Final check
    if best_centers is None:
        best_centers = grid_init(N)
        best_score = compute_score(best_centers)
    
    return best_centers
