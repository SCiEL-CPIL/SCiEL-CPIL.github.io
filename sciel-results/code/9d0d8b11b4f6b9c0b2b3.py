"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a force-based relaxation approach:
    1. Initialize points using various lattice and random patterns.
    2. Relax the configuration by applying repulsive forces for overlaps 
       and boundary violations, using an adaptive step size (cooling).
    3. Polish the best configuration using scipy.optimize.minimize.
    """
    
    def compute_radius(centers):
        """Compute the feasible radius for given centers."""
        # Minimum distance to boundary
        dist_to_boundary = 1.0 - np.linalg.norm(centers, axis=1)
        min_boundary = np.min(dist_to_boundary)
        
        # Minimum pairwise distance / 2
        if len(centers) < 2:
            min_pair = np.inf
        else:
            # Compute all pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            # Zero out diagonal
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    def force_and_penalty(centers, r):
        """
        Compute forces on centers and total penalty for given radius r.
        Forces are repulsive if constraints are violated.
        """
        n = len(centers)
        forces = np.zeros_like(centers)
        penalty = 0.0
        
        # Boundary forces: push inward if center + r > 1
        dists_to_origin = np.linalg.norm(centers, axis=1)
        # Violation if dist_to_origin > 1 - r
        violations = np.maximum(0, dists_to_origin - (1.0 - r))
        if np.any(violations > 0):
            # Force is proportional to violation, pointing towards origin
            # Normalize direction
            directions = centers / np.maximum(dists_to_origin, 1e-10)[:, np.newaxis]
            # Only apply force where violation exists
            mask = violations > 0
            forces[mask] -= directions[mask] * violations[mask] * 10.0
            penalty += np.sum(violations ** 2)
        
        # Pairwise repulsion: push apart if dist < 2r
        if n >= 2:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Violation if dist < 2r
            overlap = np.maximum(0, 2 * r - dists)
            
            if np.any(overlap > 0):
                # Force on i due to j is along (ci - cj) * overlap
                # Sum over all j for each i
                # F_i = sum_j (c_i - c_j) / dist * overlap_ij
                # We can compute this efficiently
                # Note: diff[i,j] = c_i - c_j
                # Force on i is sum_j diff[i,j] * overlap[i,j] / dists[i,j]
                
                # To avoid division by zero (though dists diag is inf, off-diag can be 0 if coincident)
                safe_dists = np.where(dists > 1e-10, dists, 1.0)
                
                # Unit vectors from j to i
                unit_vectors = diff / safe_dists[:, :, np.newaxis]
                
                # Weighted by overlap
                # F_i = sum_j unit_vector_ij * overlap_ij
                forces += np.sum(unit_vectors * overlap[:, :, np.newaxis], axis=1) * 10.0
                
                penalty += np.sum(overlap ** 2)
        
        return forces, penalty

    def relax(centers, r_init, max_iter=1000, step_init=0.01):
        """
        Relax the configuration to maximize radius.
        """
        r = r_init
        centers = centers.copy()
        step = step_init
        
        for it in range(max_iter):
            forces, penalty = force_and_penalty(centers, r)
            
            # Update positions
            centers += forces * step
            
            # Keep centers inside the container roughly to prevent explosion
            # If a center goes far out, pull it back
            norms = np.linalg.norm(centers, axis=1)
            out_mask = norms > 1.5
            if np.any(out_mask):
                centers[out_mask] /= norms[out_mask, np.newaxis] * 0.5
            
            # Try to increase r if penalty is low
            # If penalty is 0, we can increase r
            if penalty < 1e-8:
                # Estimate how much we can increase r
                # Just do a small increase
                r += 0.001
            else:
                # If penalty is high, maybe reduce step size
                # If we are stuck, cool down
                pass
            
            # Cooling schedule
            step *= 0.995
            
            if step < 1e-6:
                break
        
        return centers, r

    def polish(centers, r_init):
        """
        Polish the configuration using scipy.optimize.minimize.
        """
        n = len(centers)
        
        def objective(params):
            r = params[0]
            c = params[1:].reshape(n, 2)
            
            # Compute actual radius
            actual_r = compute_radius(c)
            
            # Penalty for invalid r (if r > actual_r, it's invalid)
            # We want to maximize r, so we minimize -r
            # But we must ensure r <= actual_r
            # If r > actual_r, the circles overlap or go out of bounds.
            # The objective should reflect this.
            # Let's use the penalty approach again but with a tight bound.
            
            # Actually, simpler: just return -actual_r? No, that doesn't enforce r.
            # The variable r is just a parameter. We should optimize positions for a fixed r?
            # No, we want to find the best r and positions.
            
            # Let's stick to the penalty method but with SLSQP which handles bounds well.
            # However, SLSQP can be slow.
            
            # Alternative: Minimize -r subject to constraints?
            # scipy.optimize.minimize doesn't handle inequality constraints directly in all methods efficiently for this many variables.
            
            # Let's use the penalty method again, but with a very high weight.
            dists_to_origin = np.linalg.norm(c, axis=1)
            boundary_violation = np.sum(np.maximum(0, dists_to_origin - (1.0 - r)) ** 2)
            
            if n >= 2:
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dists, np.inf)
                overlap_violation = np.sum(np.maximum(0, 2 * r - dists) ** 2)
            else:
                overlap_violation = 0
            
            return -r + 1000.0 * (boundary_violation + overlap_violation)
        
        # Initial params
        params = np.concatenate([[r_init], centers.flatten()])
        
        # Bounds
        bounds = [(0.01, 0.15)] + [(-1.0, 1.0)] * (2 * n)
        
        try:
            result = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12})
            
            if result.fun < 0: # Check if it's a valid improvement (penalty not too high)
                r_opt = result.x[0]
                c_opt = result.x[1:].reshape(n, 2)
                actual_r = compute_radius(c_opt)
                if actual_r > r_init:
                    return c_opt, actual_r
        except Exception:
            pass
            
        return centers, r_init

    def make_initial_hexagonal(n, scale=0.15):
        """Create initial configuration using hexagonal lattice."""
        points = []
        # Generate hexagonal lattice points
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than n points, select n closest to center
        if len(points) > n:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:n]
            points = points[indices]
        elif len(points) < n:
            # Add random points if not enough
            while len(points) < n:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.9)
                points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
        
        return points

    def make_initial_ring(n, rings=5):
        """Create initial configuration using concentric rings."""
        points = [[0.0, 0.0]]
        remaining = n - 1
        for ring in range(1, rings + 1):
            if remaining <= 0:
                break
            # Number of points on this ring
            n_on_ring = min(remaining, 6 * ring)
            radius = ring * 0.15
            for i in range(n_on_ring):
                angle = 2 * np.pi * i / n_on_ring
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
            remaining -= n_on_ring
        
        # If still not enough, add more
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.9)
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        
        return np.array(points[:n])

    def make_initial_random(n, seed=42):
        """Create random initial configuration."""
        np.random.seed(seed)
        points = []
        for _ in range(n):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.95)
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        return np.array(points)

    # Try multiple initial configurations
    best_centers = None
    best_r = -np.inf
    
    initial_configs = []
    
    # Hexagonal with different scales
    for scale in [0.12, 0.14, 0.16, 0.18]:
        initial_configs.append(make_initial_hexagonal(N, scale=scale))
    
    # Ring configurations
    initial_configs.append(make_initial_ring(N, rings=4))
    initial_configs.append(make_initial_ring(N, rings=6))
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        initial_configs.append(make_initial_random(N, seed=seed))
    
    # Optimize from each initial configuration
    for i, centers in enumerate(initial_configs):
        # Estimate initial radius
        r_init = compute_radius(centers)
        r_init = max(r_init, 0.05)
        
        # Relax
        centers_relaxed, r_relaxed = relax(centers, r_init, max_iter=500, step_init=0.02)
        
        # Polish
        centers_polished, r_polished = polish(centers_relaxed, r_relaxed)
        
        if r_polished > best_r:
            best_r = r_polished
            best_centers = centers_polished
    
    # If optimization failed, fall back to hexagonal lattice
    if best_centers is None:
        best_centers = make_initial_hexagonal(N, scale=0.15)
        best_r = compute_radius(best_centers)
    
    return best_centers
# EVOLVE-BLOCK-END
