"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Generate hexagonal lattice initialization
    def hex_lattice_init(n):
        # Try different spacings to get at least n points in unit disk
        for spacing in np.linspace(0.035, 0.055, 40):
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= n:
                # Sort by radius and take first n
                radii = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(radii, kind="stable")[:n]
                return points[idx]
        # Fallback: uniform random
        angles = np.random.uniform(0, 2 * np.pi, n)
        r = np.sqrt(np.random.uniform(0, 1, n)) * 0.95
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    def soft_min_value(points, alpha=10.0):
        """Compute soft-min of pairwise distances and boundary distances."""
        # Pairwise distances
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary = 1.0 - radii
        
        # Soft min of all constraints (we want to maximize the minimum)
        # Use negative log-sum-exp for soft-min
        all_vals = np.concatenate([dists[np.triu_indices(len(points), k=1)], 2 * boundary])
        # Soft min: -logsumexp(-alpha * vals) / alpha
        m = np.min(all_vals)
        log_sum = np.log(np.sum(np.exp(-alpha * (all_vals - m))))
        soft_min = m - log_sum / alpha
        return soft_min

    def objective_and_grad(points, alpha=10.0):
        """Objective to maximize: soft-min of constraints. Returns negative for minimization."""
        points = points.reshape(N, 2)
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary = 1.0 - radii
        
        # We want to maximize the minimum of {d_ij/2 for i<j, 1-|c_i|}
        # Let f = min(d_ij/2, 1-|c_i|). Maximize f.
        # Use soft-min: f_soft = -logsumexp(-alpha * vals) / alpha where vals = [d_ij/2, 1-|c_i|]
        
        half_dists = dists / 2.0
        vals = np.concatenate([half_dists[np.triu_indices(N, k=1)], boundary])
        
        m = np.min(vals)
        log_sum = np.log(np.sum(np.exp(-alpha * (vals - m))))
        f_soft = m - log_sum / alpha
        
        # Gradient of f_soft w.r.t. points
        # d f_soft / d vals_k = exp(-alpha*(vals_k - m)) / sum_j exp(-alpha*(vals_j - m)) * (-1/alpha) * (-alpha) ... 
        # Actually: f_soft = m - (1/alpha)*log(sum_j exp(-alpha*(vals_j - m)))
        # Let S = sum_j exp(-alpha*(vals_j - m))
        # d f_soft / d m = 1 - (1/alpha) * d/dm [log S]
        # d/dm [log S] = (1/S) * sum_j exp(-alpha*(vals_j - m)) * alpha = alpha
        # So d f_soft / d m = 1 - 1 = 0. Good, m cancels.
        # d f_soft / d vals_k = (1/alpha) * d/dvals_k [log S] = (1/alpha) * (1/S) * exp(-alpha*(vals_k - m)) * (-alpha)
        # = -exp(-alpha*(vals_k - m)) / S
        # But also m depends on vals, this gets complicated. Let's just do finite difference or use a simpler approach.
        
        # Simpler: just return the value, let scipy do numerical gradient or use a different formulation
        return -f_soft  # Negate for minimization

    def objective_smooth(points_flat, alpha=10.0):
        """Smooth objective for L-BFGS-B."""
        points = points_flat.reshape(N, 2)
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary = 1.0 - radii
        
        half_dists = dists / 2.0
        vals = np.concatenate([half_dists[np.triu_indices(N, k=1)], boundary])
        
        # Soft max (we want to maximize the min, so use soft-min which is -soft-max of negatives)
        # soft_min = -logsumexp(-alpha * vals) / alpha
        m = np.min(vals)
        shifted = -alpha * (vals - m)
        # Clip for numerical stability
        shifted = np.clip(shifted, -500, 500)
        log_sum = np.log(np.sum(np.exp(shifted)))
        f_soft = m - log_sum / alpha
        
        return -f_soft

    # Initialize
    x0 = hex_lattice_init(N)
    
    # First, try L-BFGS-B with a moderate alpha
    best_x = x0.copy()
    best_score = -objective_smooth(x0.flatten(), alpha=5.0)
    
    # Multiple restarts with different initializations and alpha values
    alphas = [5.0, 10.0, 20.0]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        
        # Try a few different initializations
        for init_idx in range(3):
            if time.time() > deadline:
                break
            
            if init_idx == 0:
                x_init = x0.copy()
            elif init_idx == 1:
                # Add small random perturbation
                x_init = x0 + np.random.normal(0, 0.001, (N, 2))
            else:
                # Different lattice spacing
                for spacing in np.linspace(0.032, 0.058, 20):
                    k = int(1.0 / spacing) + 2
                    points = []
                    for j in range(-k, k + 1):
                        for i in range(-k, k + 1):
                            x = (i + 0.5 * (j % 2)) * spacing
                            y = j * spacing * np.sqrt(3.0) / 2.0
                            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                                points.append([x, y])
                    points = np.array(points)
                    if len(points) >= N:
                        radii = np.hypot(points[:, 0], points[:, 1])
                        idx = np.argsort(radii, kind="stable")[:N]
                        x_init = points[idx]
                        break
                else:
                    continue
            
            # Optimize
            try:
                res = minimize(
                    objective_smooth,
                    x_init.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000},
                )
                if res.success or res.fun < best_score:
                    x_opt = res.x.reshape(N, 2)
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
            except Exception:
                pass
        
        if time.time() > deadline:
            break
    
    # Final polish: use higher alpha for more precise min
    if time.time() < deadline - 20:
        try:
            res = minimize(
                objective_smooth,
                best_x.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000},
            )
            if res.fun < best_score:
                best_x = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception:
            pass
    
    # Compute actual score
    dists = cdist(best_x, best_x)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.hypot(best_x[:, 0], best_x[:, 1])
    min_boundary = np.min(1.0 - radii)
    actual_score = min(min_pair, min_boundary)
    
    # Ensure validity: all points inside, no coincident points
    if np.any(~np.isfinite(best_x)) or np.any(radii >= 1.0) or min_pair <= 0:
        # Fallback to hex lattice
        return hex_lattice_init(N)
    
    return best_x
# EVOLVE-BLOCK-END
