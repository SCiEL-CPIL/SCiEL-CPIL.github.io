"""Pack 50 equal circles in a unit square [-0.5, 0.5]^2 to maximize radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """Compute the minimum radius for a given set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Minimum distance between pairs (vectorized)
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _objective_smooth(flat, beta=100.0):
    """Smooth objective function to minimize (approx negative of min radius)."""
    centers = flat.reshape(-1, 2)
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # We want to maximize the minimum of (pair_dists/2) and (boundary_dists)
    # soft_min(v) = -1/beta * log(sum(exp(-beta * v)))
    # We minimize -soft_min(v)
    
    # To save memory, we compute the sum of exponentials for pairs and boundaries separately
    # Pairs: upper triangle only
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundaries
    bound_dists = dist_to_boundary
    
    # Calculate sum of exps
    sum_exp_pairs = np.sum(np.exp(-beta * pair_dists))
    sum_exp_bounds = np.sum(np.exp(-beta * bound_dists))
    
    total_sum = sum_exp_pairs + sum_exp_bounds
    
    # Prevent log(0)
    if total_sum == 0:
        return 0.0
        
    soft_min_val = -1.0 / beta * np.log(total_sum)
    
    return -soft_min_val

def _hex_lattice_init(row_counts, dx, dy):
    """Create hex lattice centers with given row counts and spacings."""
    centers = []
    total_h = (len(row_counts) - 1) * dy
    y_offset = -total_h / 2.0
    
    for i, n_cols in enumerate(row_counts):
        y = y_offset + i * dy
        if n_cols == 1:
            x_offset = 0.0
        else:
            row_width = (n_cols - 1) * dx
            x_offset = -row_width / 2.0
            if i % 2 == 1:
                x_offset += dx / 2.0
        for j in range(n_cols):
            x = x_offset + j * dx
            centers.append([x, y])
    
    return np.array(centers)

def _generate_hex_configs():
    """Generate various hexagonal lattice configurations for 50 circles."""
    configs = []
    
    # Pattern: rows of varying lengths
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 7, 6, 7, 6, 7, 6, 4],
        [5, 7, 6, 7, 6, 7, 6, 6],
        [6, 6, 7, 6, 7, 6, 7, 5],
        [8, 6, 7, 6, 7, 6, 7, 3],
        [7, 6, 6, 7, 6, 7, 7, 4],
        [6, 7, 7, 6, 7, 6, 6, 5],
        [7, 7, 7, 6, 6, 7, 6, 4],
        [6, 6, 6, 7, 7, 6, 7, 5],
    ]
    
    for pattern in patterns:
        if sum(pattern) != N:
            continue
        configs.append(pattern)
    
    return configs

def _optimize_from_start(centers, start_time, deadline):
    """Optimize starting from given centers using L-BFGS-B with smooth objective."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    flat = centers.flatten()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try optimization with smooth objective
    try:
        result = minimize(
            lambda x: _objective_smooth(x, beta=100.0), 
            flat, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        opt_centers = result.x.reshape(-1, 2)
        opt_score = _score(opt_centers)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers.copy()
            
        # Try with higher beta for sharper min approximation
        if time.time() - start_time < deadline - 15:
            result2 = minimize(
                lambda x: _objective_smooth(x, beta=200.0), 
                opt_centers.flatten(), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_centers2 = result2.x.reshape(-1, 2)
            opt_score2 = _score(opt_centers2)
            
            if opt_score2 > best_score:
                best_score = opt_score2
                best_centers = opt_centers2.copy()
                
        # High-precision polish with beta=500
        if time.time() - start_time < deadline - 10:
            result3 = minimize(
                lambda x: _objective_smooth(x, beta=500.0), 
                best_centers.flatten(), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            opt_centers3 = result3.x.reshape(-1, 2)
            opt_score3 = _score(opt_centers3)
            
            if opt_score3 > best_score:
                best_score = opt_score3
                best_centers = opt_centers3.copy()
                
        # Final polish with beta=1000
        if time.time() - start_time < deadline - 5:
            result4 = minimize(
                lambda x: _objective_smooth(x, beta=1000.0), 
                best_centers.flatten(), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            opt_centers4 = result4.x.reshape(-1, 2)
            opt_score4 = _score(opt_centers4)
            
            if opt_score4 > best_score:
                best_score = opt_score4
                best_centers = opt_centers4.copy()

    except Exception:
        pass
    
    return best_centers, best_score

def _local_refinement(centers, start_time, deadline):
    """Additional local refinement with small perturbations."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Try small random perturbations and re-optimize
    for attempt in range(20):
        if time.time() - start_time > deadline - 5:
            break
        
        # Add small random noise
        noise_scale = 0.01 / (attempt + 1)
        perturbed = centers + np.random.randn(*centers.shape) * noise_scale
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        flat = perturbed.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                lambda x: _objective_smooth(x, beta=100.0),
                flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_centers = result.x.reshape(-1, 2)
            opt_score = _score(opt_centers)
            
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # Final high-beta polish on best result
    if time.time() - start_time < deadline - 2:
        try:
            result = minimize(
                lambda x: _objective_smooth(x, beta=1000.0),
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            opt_centers = result.x.reshape(-1, 2)
            opt_score = _score(opt_centers)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = -np.inf
    
    # Generate hex lattice configurations
    patterns = _generate_hex_configs()
    
    # Try different spacings for each pattern, with jitter
    np.random.seed(42)
    
    for pattern in patterns[:5]: # Limit to top 5 patterns to save time
        if time.time() - start_time > deadline - 30:
            break
        
        n_rows = len(pattern)
        max_cols = max(pattern)
        
        # Try a few dx values
        for dx in np.linspace(0.10, 0.14, 5):
            if time.time() - start_time > deadline - 30:
                break
            
            dy = dx * np.sqrt(3) / 2.0
            
            centers = _hex_lattice_init(pattern, dx, dy)
            
            if len(centers) < N:
                continue
            
            if np.any(np.abs(centers) > 0.5):
                continue
            
            # Add jitter to break symmetry
            jittered_centers = centers + np.random.randn(*centers.shape) * 0.01
            jittered_centers = np.clip(jittered_centers, -0.49, 0.49)
            
            s = _score(jittered_centers)
            if s > best_score:
                best_score = s
                best_centers = jittered_centers.copy()
            else:
                # Try original if jitter made it worse
                s_orig = _score(centers)
                if s_orig > best_score:
                    best_score = s_orig
                    best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Optimize from best initial configuration
    best_centers, best_score = _optimize_from_start(best_centers, start_time, deadline)
    
    # Local refinement
    if time.time() - start_time < deadline - 10:
        best_centers, best_score = _local_refinement(best_centers, start_time, deadline)
    
    return best_centers
