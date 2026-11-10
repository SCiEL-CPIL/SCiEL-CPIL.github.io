import numpy as np
import time

N = 100

def _radius_score(c):
    """Compute the exact packing radius for a configuration."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _smooth_min_grad(c, alpha):
    """
    Compute the smooth minimum of all constraints (pair distances and wall distances)
    and its gradient with respect to c.
    
    We want to maximize r, where r is the minimum of:
      - d_ij / 2 for all i < j
      - (0.5 - |c_ik|) for all i, k
    
    We use a log-sum-exp style smooth minimum:
      S(c) = -(1/alpha) * log( sum_j exp(-alpha * x_j) )
    where x_j are the constraint values.
    
    gradient dS/dc = (1/sum(exp(-alpha*x))) * sum(exp(-alpha*x_j) * dx_j/dc)
    
    This is a weighted average of the gradients of individual constraints,
    where weights are softmax(-alpha * x).
    """
    c = np.asarray(c, dtype=np.float64)
    
    # Collect all constraint values
    # Pair constraints: d_ij / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-30)
    np.fill_diagonal(d, np.inf)
    
    # Only upper triangle for pairs
    iu = np.triu_indices(N, 1)
    pair_vals = d[iu] / 2.0
    n_pairs = len(pair_vals)
    
    # Wall constraints: 0.5 - |c_ik|
    wall_vals = 0.5 - np.abs(c)
    # Flatten
    wall_vals_flat = wall_vals.flatten()
    n_walls = N * 3
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals, wall_vals_flat])
    n_total = len(all_vals)
    
    # Softmax weights: w_j = exp(-alpha * x_j) / sum(exp(-alpha * x_k))
    # For numerical stability, subtract max
    shifted = -alpha * all_vals
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Smooth minimum value
    S = -(1.0 / alpha) * (np.log(sum_exp) + max_shifted)
    
    # Gradient
    # We need dS/dc. S = smooth_min(x_1, ..., x_n)
    # dS/dc = sum_j w_j * (dx_j/dc)
    
    # For pairs: d(d_ij/2)/dc_a = (c_a - c_b) / (2 * d_ij) for a in pair
    # For walls: d(0.5 - |c_ik|)/dc_jk = -sign(c_jk)
    
    grad = np.zeros_like(c)
    
    # Pair gradients
    pair_grads = np.zeros((N, 3))
    i_idx, j_idx = iu
    for idx in range(n_pairs):
        i, j = i_idx[idx], j_idx[idx]
        w = weights[idx]
        d_ij = d[i, j]
        if d_ij < 1e-12:
            continue
        vec = (c[i] - c[j]) / (2.0 * d_ij)
        pair_grads[i] += w * vec
        pair_grads[j] -= w * vec
    grad += pair_grads
    
    # Wall gradients
    wall_grads = np.zeros((N, 3))
    for idx in range(N):
        for k in range(3):
            w = weights[n_pairs + idx * 3 + k]
            c_ik = c[idx, k]
            if c_ik > 1e-12:
                wall_grads[idx, k] -= w
            elif c_ik < -1e-12:
                wall_grads[idx, k] += w
            else:
                wall_grads[idx, k] -= w  # arbitrary
    grad += wall_grads
    
    return S, grad

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Phase 1: Multi-start with different lattice configurations
    # Generate several initial configurations from different lattice spacings
    
    rng = np.random.default_rng(42)
    
    s_values = np.linspace(0.17, 0.22, 8)
    
    for s in s_values:
        if time.time() > deadline - 40:
            break
        
        # Generate lattice
        n_dim = int(np.ceil(1.0 / s)) + 1
        coords = np.arange(-n_dim // 2, n_dim // 2 + 1) * s
        grid = np.array([(x, y, z) for z in coords for y in coords for x in coords])
        
        # Filter to inside cube
        inside_mask = np.all(np.abs(grid) <= 0.5 - 1e-6, axis=1)
        if np.sum(inside_mask) < N:
            continue
        inside_grid = grid[inside_mask]
        
        # Pick N closest to center
        dists_center = np.linalg.norm(inside_grid, axis=1)
        sorted_idx = np.argsort(dists_center)
        centers = inside_grid[sorted_idx[:N]].copy()
        
        # Add small random perturbation to break symmetry
        centers += rng.uniform(-0.005, 0.005, size=(N, 3))
        centers = np.clip(centers, -0.5, 0.5)
        
        r = _radius_score(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback
        best_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        best_radius = _radius_score(best_centers)
    
    # Phase 2: Gradient-based optimization using smooth minimum surrogate
    # We maximize S(c) = smooth_min of all constraints
    # Using Adam optimizer with adaptive alpha
    
    c = best_centers.copy()
    r = _radius_score(c)
    
    # Adam optimizer state
    m = np.zeros_like(c)
    v = np.zeros_like(c)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    
    # Learning rate schedule
    lr_start = 0.02
    lr_end = 0.0005
    total_steps = 3000
    
    # Alpha schedule for smooth min
    alpha_start = 20.0
    alpha_end = 200.0
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Progress
        t = step / total_steps
        
        # Interpolate learning rate
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Interpolate alpha
        alpha = alpha_start * (alpha_end / alpha_start) ** t
        
        # Compute smooth min and gradient
        S, grad = _smooth_min_grad(c, alpha)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        
        c_new = c + lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        
        # Project to feasible region (cube)
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Check if this improves the true radius
        r_new = _radius_score(c_new)
        
        if r_new > r:
            c = c_new
            r = r_new
            if r > best_radius:
                best_radius = r
                best_centers = c.copy()
        else:
            # Keep old position, but still update Adam state with gradient
            c = c  # unchanged
    
    # Phase 3: Local refinement - move individual spheres to increase radius
    # Identify bottleneck spheres and try to move them
    
    max_local_steps = 200
    for local_step in range(max_local_steps):
        if time.time() > deadline - 3:
            break
        
        c = best_centers.copy()
        r = _radius_score(c)
        
        # Compute constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        # Determine which constraint is tight
        if min_pair / 2.0 <= min_wall + 1e-9:
            # Pair constraint is tight
            i, j = min_pair_idx
            # Try to move i away from j
            direction = (c[i] - c[j])
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-12:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm_dir
            
            # Try small moves
            improved = False
            for step_size in [0.005, 0.002, 0.001, 0.0005]:
                c_trial = c.copy()
                c_trial[i] = c[i] + step_size * direction
                c_trial = np.clip(c_trial, -0.5, 0.5)
                r_trial = _radius_score(c_trial)
                if r_trial > r + 1e-9:
                    c = c_trial
                    r = r_trial
                    improved = True
                    break
                # Also try moving j in opposite direction
                c_trial = c.copy()
                c_trial[j] = c[j] - step_size * direction
                c_trial = np.clip(c_trial, -0.5, 0.5)
                r_trial = _radius_score(c_trial)
                if r_trial > r + 1e-9:
                    c = c_trial
                    r = r_trial
                    improved = True
                    break
            
            if not improved:
                # Try moving both
                for step_size in [0.005, 0.002, 0.001]:
                    c_trial = c.copy()
                    c_trial[i] = c[i] + step_size * direction
                    c_trial[j] = c[j] - step_size * direction
                    c_trial = np.clip(c_trial, -0.5, 0.5)
                    r_trial = _radius_score(c_trial)
                    if r_trial > r + 1e-9:
                        c = c_trial
                        r = r_trial
                        improved = True
                        break
        else:
            # Wall constraint is tight
            i, k = min_wall_idx
            # Move away from wall
            c_ik = c[i, k]
            if c_ik >= 0:
                direction = np.zeros(3)
                direction[k] = -1.0  # move toward negative
            else:
                direction = np.zeros(3)
                direction[k] = 1.0  # move toward positive
            
            improved = False
            for step_size in [0.005, 0.002, 0.001, 0.0005]:
                c_trial = c.copy()
                c_trial[i] = c[i] + step_size * direction
                c_trial = np.clip(c_trial, -0.5, 0.5)
                r_trial = _radius_score(c_trial)
                if r_trial > r + 1e-9:
                    c = c_trial
                    r = r_trial
                    improved = True
                    break
        
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        if not improved:
            # No improvement, try a different pair
            # Randomly select a tight pair
            eps = 1e-5
            tight_pairs = []
            for ii in range(N):
                for jj in range(ii + 1, N):
                    if dists[ii, jj] <= 2.0 * r + eps:
                        tight_pairs.append((ii, jj))
            
            if tight_pairs:
                idx_pair = rng.integers(len(tight_pairs))
                i, j = tight_pairs[idx_pair]
                direction = (c[i] - c[j])
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-12:
                    direction = rng.standard_normal(3)
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = direction / norm_dir
                
                for step_size in [0.005, 0.002, 0.001]:
                    c_trial = c.copy()
                    c_trial[i] = c[i] + step_size * direction
                    c_trial[j] = c[j] - step_size * direction
                    c_trial = np.clip(c_trial, -0.5, 0.5)
                    r_trial = _radius_score(c_trial)
                    if r_trial > r + 1e-9:
                        c = c_trial
                        r = r_trial
                        improved = True
                        break
                    c_trial = c.copy()
                    c_trial[i] = c[i] - step_size * direction
                    c_trial[j] = c[j] + step_size * direction
                    c_trial = np.clip(c_trial, -0.5, 0.5)
                    r_trial = _radius_score(c_trial)
                    if r_trial > r + 1e-9:
                        c = c_trial
                        r = r_trial
                        improved = True
                        break
            
            if not improved:
                # Try random perturbations of all spheres
                c_trial = c.copy()
                c_trial += rng.normal(0, 0.002, size=(N, 3))
                c_trial = np.clip(c_trial, -0.5, 0.5)
                r_trial = _radius_score(c_trial)
                if r_trial > r + 1e-9:
                    c = c_trial
                    r = r_trial
                    improved = True
            
            if not improved:
                break
        
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
    
    # Final validation
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        best_centers = np.clip(rng.uniform(-0.4, 0.4, size=(N, 3)), -0.5, 0.5)
    
    return best_centers
