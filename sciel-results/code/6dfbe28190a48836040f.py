"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import itertools
import time


def _compute_radius(centers):
    """Compute the max-min radius for given centers."""
    if len(centers) < 2:
        return 0.0
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    # Square is [-0.5, 0.5]^2
    boundary_dists = np.minimum(0.5 - np.abs(centers), axis=0)
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _construct_initial_grid():
    """Create a reasonable initial grid packing."""
    m = 8  # 8x8 grid gives 64 spots, we need 50
    # We'll use a subset of an 8x8 grid, or try different configurations
    # Let's try a 7x7 grid (49 circles) plus one more, or 8x8 with 50 selected
    
    # Better: use a hexagonal-like arrangement
    # For now, let's use a grid and then optimize
    
    # Start with a square grid of 7x7 = 49, add 1 more
    # Or use 8x8 and pick 50
    
    m = 8
    x = np.linspace(-0.5, 0.5, m)
    y = np.linspace(-0.5, 0.5, m)
    X, Y = np.meshgrid(x, y)
    centers = np.column_stack([X.ravel(), Y.ravel()])
    
    # Shuffle and take first N
    idx = np.random.default_rng(42).permutation(len(centers))[:N]
    return centers[idx]


def _construct_hexagonal():
    """Create a hexagonal packing initial guess."""
    centers = []
    # Hexagonal packing with spacing s
    # Try to fit 50 circles
    # Rows alternate offset
    s = 0.12  # initial spacing estimate
    row_y = -0.45
    while len(centers) < N:
        row_x_start = -0.45 if int((row_y + 0.5) / s) % 2 == 0 else -0.45 + s/2
        x = row_x_start
        while x <= 0.45:
            if len(centers) < N:
                centers.append([x, row_y])
            x += s
        row_y += s * np.sqrt(3) / 2
    
    return np.array(centers[:N])


def construct():
    """Main construct function."""
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 10s safety margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Config 1: Grid-based
    initial_configs.append(_construct_initial_grid())
    
    # Config 2: Hexagonal
    initial_configs.append(_construct_hexagonal())
    
    # Config 3: Random perturbation of grid
    rng = np.random.default_rng(123)
    base = _construct_initial_grid()
    perturbed = base + rng.uniform(-0.02, 0.02, size=base.shape)
    # Clip to valid region
    perturbed[:, 0] = np.clip(perturbed[:, 0], -0.45, 0.45)
    perturbed[:, 1] = np.clip(perturbed[:, 1], -0.45, 0.45)
    initial_configs.append(perturbed)
    
    best_centers = None
    best_radius = 0.0
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        # Try SLSQP optimization with epigraph formulation
        # Variables: 2*N centers + 1 radius r
        # Maximize r subject to:
        #   |c_i - c_j| >= 2*r for all i < j
        #   |c_ik| <= 0.5 - r for all i, k
        
        n_vars = 2 * N + 1
        
        def objective(x):
            """Negative radius to minimize."""
            return -x[-1]
        
        # Inequality constraints: g(x) >= 0
        # For pairs: |c_i - c_j|^2 - (2*r)^2 >= 0
        # For boundaries: 0.5 - r - |c_ik| >= 0
        
        def pair_constraints(x):
            centers = x[:-1].reshape(N, 2)
            r = x[-1]
            # Compute all pairwise distances squared
            diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diffs ** 2, axis=2)
            # For i < j, we need dists_sq[i,j] >= (2*r)^2
            i_idx, j_idx = np.triu_indices(N, k=1)
            return dists_sq[i_idx, j_idx] - (2 * r) ** 2
        
        def boundary_constraints(x):
            centers = x[:-1].reshape(N, 2)
            r = x[-1]
            # 0.5 - r - |c_ik| >= 0
            return 0.5 - r - np.abs(centers)
        
        # Initial guess
        x0 = np.zeros(n_vars)
        x0[:-1] = config.ravel()
        # Initial radius estimate
        x0[-1] = _compute_radius(config) * 0.9  # slightly less to be feasible
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10},
                constraints=[
                    {'type': 'ineq', 'fun': pair_constraints},
                    {'type': 'ineq', 'fun': boundary_constraints}
                ]
            )
            
            if result.success or result.fun < 0:
                candidate = result.x[:-1].reshape(N, 2)
                candidate_r = result.x[-1]
                actual_r = _compute_radius(candidate)
                
                if actual_r > best_radius:
                    best_radius = actual_r
                    best_centers = candidate.copy()
                    
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If no optimization succeeded, fall back to best initial config
    if best_centers is None:
        for config in initial_configs:
            r = _compute_radius(config)
            if r > best_radius:
                best_radius = r
                best_centers = config.copy()
    
    # Final polish: try a few more random restarts if time permits
    if time.time() < deadline - 20:
        rng2 = np.random.default_rng(456)
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb best solution
            if best_centers is not None:
                perturbed = best_centers + rng2.normal(0, 0.005, size=best_centers.shape)
                perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
                perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
            else:
                perturbed = _construct_initial_grid()
            
            n_vars = 2 * N + 1
            x0 = np.zeros(n_vars)
            x0[:-1] = perturbed.ravel()
            x0[-1] = _compute_radius(perturbed) * 0.95
            
            def objective(x):
                return -x[-1]
            
            def pair_constraints(x):
                centers = x[:-1].reshape(N, 2)
                r = x[-1]
                diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists_sq = np.sum(diffs ** 2, axis=2)
                i_idx, j_idx = np.triu_indices(N, k=1)
                return dists_sq[i_idx, j_idx] - (2 * r) ** 2
            
            def boundary_constraints(x):
                centers = x[:-1].reshape(N, 2)
                r = x[-1]
                return 0.5 - r - np.abs(centers)
            
            bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10},
                    constraints=[
                        {'type': 'ineq', 'fun': pair_constraints},
                        {'type': 'ineq', 'fun': boundary_constraints}
                    ]
                )
                
                if result.success or result.fun < 0:
                    candidate = result.x[:-1].reshape(N, 2)
                    actual_r = _compute_radius(candidate)
                    
                    if actual_r > best_radius:
                        best_radius = actual_r
                        best_centers = candidate.copy()
                        
            except Exception:
                pass
    
    # Ensure validity: clip centers to square
    if best_centers is not None:
        best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
        best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
        return best_centers
    else:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

# EVOLVE-BLOCK-END
