"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 maximizing the sum of radii.
    Uses a coverage-based formulation with differentiable soft-min operators
    to enforce that a grid of probe points is covered by the union of disks.
    This helps distribute circles evenly and avoid local minima.
    """
    n = N
    
    def hexagonal_init(seed):
        """Generate a hexagonal packing initialization."""
        np.random.seed(seed)
        centers = []
        # Approximate hexagonal grid
        # 26 circles
        # Try to fit rows of 5, 5, 5, 5, 5, 1 or similar
        # Let's use a standard hexagonal lattice and pick the best 26 spots
        # or just generate a grid and jitter it.
        
        # Simple robust init: Grid with jitter
        grid_size = 5
        probe_x = np.linspace(0.1, 0.9, grid_size)
        probe_y = np.linspace(0.1, 0.9, grid_size)
        init_centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Add jitter to break symmetry
                jitter_x = np.random.uniform(-0.02, 0.02)
                jitter_y = np.random.uniform(-0.02, 0.02)
                init_centers.append([probe_x[i] + jitter_x, probe_y[j] + jitter_y])
        
        # Add the 26th circle in the center or a random empty spot
        init_centers.append([0.5 + np.random.uniform(-0.1, 0.1), 0.5 + np.random.uniform(-0.1, 0.1)])
        
        init_centers = np.array(init_centers)
        # Clip to bounds
        init_centers[:, 0] = np.clip(init_centers[:, 0], 0.05, 0.95)
        init_centers[:, 1] = np.clip(init_centers[:, 1], 0.05, 0.95)
        init_radii = np.full(n, 0.1)
        
        return init_centers, init_radii

    def solve_instance(seed):
        np.random.seed(seed)
        
        # Define probe points: 6x6 grid for better coverage enforcement
        grid_size = 6
        probe_x = np.linspace(0.05, 0.95, grid_size)
        probe_y = np.linspace(0.05, 0.95, grid_size)
        probes = np.array([(x, y) for x in probe_x for y in probe_y])
        num_probes = len(probes)
        
        init_centers, init_radii = hexagonal_init(seed)
        
        x0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], init_radii])
        
        def unpack(params):
            x = params[0:n]
            y = params[n:2*n]
            r = params[2*n:3*n]
            return x, y, r
        
        def penalized_objective(params, mu=100.0, t=10.0):
            x, y, r = unpack(params)
            
            # Base objective: minimize negative sum of radii
            obj = -np.sum(r)
            
            penalty = 0.0
            
            # Non-overlap penalty
            # Vectorized for speed
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist_mat = np.sqrt(dx*dx + dy*dy + 1e-12)
            rad_sum_mat = r[:, np.newaxis] + r[np.newaxis, :]
            
            # Only upper triangle
            iu = np.triu_indices(n, k=1)
            overlaps = np.maximum(0, rad_sum_mat[iu] - dist_mat[iu])
            penalty += np.sum(overlaps**2)
            
            # Boundary penalty
            # x < r or 1-x < r
            bound_pen_x = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
            bound_pen_y = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
            penalty += np.sum(bound_pen_x**2) + np.sum(bound_pen_y**2)
            
            # Coverage penalty
            # For each probe, if it is not covered (softmin > 0), penalize.
            # Vectorized soft-min
            # dists: (num_probes, n)
            dx_p = x[np.newaxis, :] - probes[:, 0][:, np.newaxis]
            dy_p = y[np.newaxis, :] - probes[:, 1][:, np.newaxis]
            dists = np.sqrt(dx_p*dx_p + dy_p*dy_p + 1e-12)
            z = dists - r[np.newaxis, :]
            
            # Soft min along axis 1
            z_max = np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(-t * (z - z_max))
            sum_exp = np.sum(exp_z, axis=1, keepdims=True)
            soft_min_val = z_max - (1.0/t) * np.log(sum_exp)
            
            # soft_min_val is (num_probes, 1)
            # If > 0, uncovered
            uncovered = np.maximum(0, soft_min_val)
            penalty += np.sum(uncovered**2)
            
            return obj + mu * penalty
        
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((1e-4, 0.5))
            
        best_params = x0.copy()
        best_val = penalized_objective(x0, mu=50.0, t=5.0)
        
        # Multi-stage optimization with increasing mu and t
        stages = [
            (10.0, 5.0),
            (50.0, 10.0),
            (100.0, 20.0),
            (200.0, 40.0),
            (500.0, 80.0),
            (1000.0, 150.0)
        ]
        
        for mu, t in stages:
            res = minimize(
                penalized_objective,
                best_params,
                args=(mu, t),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        
        x, y, r = unpack(best_params)
        
        # Cleanup
        r = np.maximum(r, 1e-6)
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
            
        # Final overlap resolution
        for _ in range(20):
            adjusted = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    required = r[i] + r[j]
                    if dist < required:
                        scale = dist / (required + 1e-12)
                        r[i] *= scale
                        r[j] *= scale
                        adjusted = True
            if not adjusted:
                break
                
        result = np.column_stack([x, y, r])
        return result, np.sum(r)

    # Multi-start
    best_sum_r = 0
    best_result = None
    
    # Try a few seeds
    seeds = [42, 123, 456, 789, 1011]
    for seed in seeds:
        try:
            res, val = solve_instance(seed)
            # Check validity
            valid = True
            x, y, r = res[:, 0], res[:, 1], res[:, 2]
            if np.any(r < 0) or np.any(x < 0) or np.any(y < 0) or np.any(x > 1) or np.any(y > 1):
                valid = False
            if np.any(x < r) or np.any(y < r) or np.any(1-x < r) or np.any(1-y < r):
                valid = False
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-6:
                        valid = False
            if valid and val > best_sum_r:
                best_sum_r = val
                best_result = res
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback
        best_result = solve_instance(42)[0]
        
    return best_result

# EVOLVE-BLOCK-END
