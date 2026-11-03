"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit
    
    def min_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def make_initial_grid(n):
        """Create a better initial configuration using FCC-like arrangement."""
        # Use a 4x4x4 grid with FCC offsets
        m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        
        # Generate FCC-like positions
        fcc_offsets = [
            (0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
        ]
        
        centers = []
        for z in ticks:
            for y in ticks:
                for x in ticks:
                    for ox, oy, oz in fcc_offsets:
                        cx = (x + ox) - 0.25
                        cy = (y + oy) - 0.25
                        cz = (z + oz) - 0.25
                        if -0.5 < cx < 0.5 and -0.5 < cy < 0.5 and -0.5 < cz < 0.5:
                            centers.append([cx, cy, cz])
        
        # If we have more than n, take the first n
        centers = np.array(centers[:n])
        
        # If we have fewer, pad with random points
        if len(centers) < n:
            remaining = n - len(centers)
            random_pts = (np.random.rand(remaining, 3) - 0.5) * 0.8
            centers = np.vstack([centers, random_pts])
        
        return centers
    
    def soft_min_radius_and_grad(x, beta):
        """Compute soft-min radius and its gradient efficiently."""
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        # Combine all constraints
        pair_half = d / 2.0
        iu = np.triu_indices(N, k=1)
        all_vals = np.concatenate([pair_half[iu], dist_to_walls])
        
        # Soft min with numerical stability
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        log_sum = np.log(sum_exp)
        soft_min = max_val - log_sum / beta
        
        # Gradient computation (vectorized)
        grad_c = np.zeros_like(c)
        
        # Pair contributions
        pi, pj = iu
        pair_weights = weights[:N*(N-1)//2]
        
        # For each pair, add weighted unit vector contributions
        for i, j, w in zip(pi, pj, pair_weights):
            if w > 1e-15:
                vec = c[i] - c[j]
                norm = d[i, j]
                if norm > 1e-9:
                    unit = vec / norm
                    grad_c[i] += w * unit
                    grad_c[j] -= w * unit
        
        # Wall contributions
        wall_weights = weights[N*(N-1)//2:]
        for i, w in enumerate(wall_weights):
            if w > 1e-15:
                for k in range(3):
                    if c[i, k] > 0:
                        grad_c[i, k] += w * (-1.0)
                    else:
                        grad_c[i, k] += w * (1.0)
        
        # We minimize -soft_min, so gradient is negative
        return -soft_min, -grad_c.flatten()
    
    def optimize_from(x0, beta_schedule, max_iter_per_beta=300):
        """Run continuation optimization from initial point."""
        x = x0.copy()
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - 10:
                break
            
            try:
                res = minimize(
                    fun=lambda x, b=beta: soft_min_radius_and_grad(x, b)[0],
                    x0=x,
                    jac=lambda x, b=beta: soft_min_radius_and_grad(x, b)[1],
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                    options={'maxiter': max_iter_per_beta, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                x = res.x
            except Exception:
                pass
        
        return x
    
    # Generate multiple initial configurations
    np.random.seed(42)
    initial_configs = []
    
    # Config 1: FCC-like grid
    initial_configs.append(make_initial_grid(N))
    
    # Config 2: Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_configs.append(grid[:N].copy())
    
    # Config 3: Perturbed FCC
    fcc = make_initial_grid(N)
    fcc_perturbed = fcc + np.random.randn(N, 3) * 0.02
    fcc_perturbed = np.clip(fcc_perturbed, -0.45, 0.45)
    initial_configs.append(fcc_perturbed)
    
    best_centers = None
    best_r = -np.inf
    
    # Try each initial configuration
    for idx, init in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Continuation schedule
        betas = [10, 30, 100, 300, 1000, 3000]
        
        x = optimize_from(init.flatten(), betas, max_iter_per_beta=200)
        c = x.reshape(N, 3)
        r = min_radius(c)
        
        if r > best_r:
            best_r = r
            best_centers = c.copy()
        
        # Final high-precision polish
        if time.time() - start_time < time_limit - 10:
            beta_final = 10000
            try:
                res = minimize(
                    fun=lambda x, b=beta_final: soft_min_radius_and_grad(x, b)[0],
                    x0=x,
                    jac=lambda x, b=beta_final: soft_min_radius_and_grad(x, b)[1],
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
                )
                x = res.x
                c = x.reshape(N, 3)
                r = min_radius(c)
                if r > best_r:
                    best_r = r
                    best_centers = c.copy()
            except Exception:
                pass
    
    # If no valid result, fall back to simple grid
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
