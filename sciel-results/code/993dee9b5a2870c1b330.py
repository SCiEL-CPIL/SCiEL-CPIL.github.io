"""Optimized sphere packing for 100 equal spheres in a unit cube."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 100
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _score(centers):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: distance from center to nearest face
    boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(boundary_dists)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(centers)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_boundary)

def _soft_min_energy(centers, beta=50.0):
    """
    Compute a smooth approximation of -min(pairwise_dists, boundary_dists).
    We want to maximize the min distance, so we minimize the negative.
    Uses log-sum-exp trick for smooth min.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary_per_sphere = np.min(boundary_dists, axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(centers))
    np.fill_diagonal(D, np.inf)
    
    # All distances: pairwise + boundary
    # We want min of all these
    all_dists = np.concatenate([D.ravel()[np.triu_indices(n, k=1)[1]], min_boundary_per_sphere])
    
    # Soft min: -log(sum(exp(-beta * d_i))) / beta
    # For numerical stability
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    soft_min_val = min_d - np.log(np.sum(exp_terms)) / beta
    
    return -soft_min_val  # We want to minimize this (maximize soft_min_val)

def _soft_min_energy_and_grad(centers, beta=50.0):
    """
    Compute soft-min energy and its gradient.
    """
    centers = np.asarray(centers, dtype=np.float64).copy()
    n = len(centers)
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary_per_sphere = np.min(boundary_dists, axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(centers))
    np.fill_diagonal(D, np.inf)
    
    # Collect all distances
    # Pairwise: for i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = D[i_idx, j_idx]
    
    # All distances
    all_dists = np.concatenate([pair_dists, min_boundary_per_sphere])
    
    # Soft min value
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    sum_exp = np.sum(exp_terms)
    soft_min_val = min_d - np.log(sum_exp) / beta
    
    # Gradient of soft_min_val w.r.t. centers
    # d(soft_min_val)/d(centers) = sum_j [w_j * d(d_j)/d(centers)]
    # where w_j = exp(-beta * d_j) / sum_exp
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Pairwise distance gradients
    # d|c_i - c_j|/d c_i = (c_i - c_j) / |c_i - c_j|
    # d|c_i - c_j|/d c_j = (c_j - c_i) / |c_i - c_j|
    for k in range(len(pair_dists)):
        i, j = i_idx[k], j_idx[k]
        d = pair_dists[k]
        if d < 1e-12:
            continue
        w = weights[k]
        diff = centers[i] - centers[j]
        grad[i] += w * diff / d
        grad[j] += w * (-diff) / d
    
    # Boundary distance gradients
    # d(0.5 - |c_ik|)/d c_ik = -sign(c_ik)
    for i in range(n):
        w = weights[len(pair_dists) + i]
        for k in range(3):
            c = centers[i, k]
            if c > 0:
                grad[i, k] += w * (-1.0)
            elif c < 0:
                grad[i, k] += w * (1.0)
            else:
                grad[i, k] += 0.0
    
    # Energy = -soft_min_val, so gradient of energy = -gradient of soft_min_val
    energy = -soft_min_val
    grad_energy = -grad
    
    return energy, grad_energy

def _initial_grid_bcc():
    """
    Create an initial configuration using a body-centered cubic-like arrangement.
    For 100 spheres, we use a 5x5x4 grid with BCC-like offsets.
    """
    centers = []
    
    # Use a 5x5x4 arrangement = 100 spheres
    # Simple grid first, then we'll optimize
    m_x, m_y, m_z = 5, 5, 4
    
    for iz in range(m_z):
        for iy in range(m_y):
            for ix in range(m_x):
                # Center of each cell
                x = (ix + 0.5) / m_x - 0.5
                y = (iy + 0.5) / m_y - 0.5
                z = (iz + 0.5) / m_z - 0.5
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers

def _initial_grid_fcc_like():
    """
    Create a face-centered cubic-like initial arrangement.
    This gives better initial spacing than simple cubic.
    """
    centers = []
    
    # Use layers with alternating offsets
    # 100 spheres: try 10 layers of 10, or 5 layers of 20, etc.
    # Let's do 10 layers, 10 per layer, with FCC-like offsets
    
    n_layers = 10
    per_layer = 10
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        # Alternate between two offset grids
        if layer % 2 == 0:
            # 5x2 grid
            for iy in range(2):
                for ix in range(5):
                    x = (ix + 0.5) / 5 - 0.5
                    y = (iy + 0.5) / 2 - 0.5
                    centers.append([x, y, z])
        else:
            # 5x2 grid with offset
            for iy in range(2):
                for ix in range(5):
                    x = (ix + 0.25) / 5 - 0.5
                    y = (iy + 0.25) / 2 - 0.5
                    centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers

def _optimize_from_start(centers0, time_limit=20.0):
    """
    Optimize from a starting configuration using L-BFGS-B with soft-min objective.
    """
    start_time = time.time()
    
    # Flatten centers
    x0 = centers0.flatten()
    
    def objective(x):
        c = x.reshape(N, 3)
        # Check boundaries
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1e10
        energy, _ = _soft_min_energy_and_grad(c, beta=50.0)
        return energy
    
    def gradient(x):
        c = x.reshape(N, 3)
        _, grad = _soft_min_energy_and_grad(c, beta=50.0)
        return grad.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'maxfun': 5000
            }
        )
        
        if result.success or result.fun < 0:
            return result.x.reshape(N, 3)
        else:
            return centers0
            
    except Exception:
        return centers0
    
    return centers0

def _polish(centers, time_limit=10.0):
    """
    Final high-precision polish using smaller beta and more iterations.
    """
    start_time = time.time()
    
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(N, 3)
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1e10
        energy, _ = _soft_min_energy_and_grad(c, beta=100.0)
        return energy
    
    def gradient(x):
        c = x.reshape(N, 3)
        _, grad = _soft_min_energy_and_grad(c, beta=100.0)
        return grad.flatten()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 100,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 30
            }
        )
        
        if result.success or result.fun < 0:
            return result.x.reshape(N, 3)
        else:
            return centers
            
    except Exception:
        return centers
    
    return centers

def construct():
    """
    Construct the optimal packing of 100 equal spheres in a unit cube.
    Returns array of shape (N, 3) with sphere centers.
    """
    global_start = time.time()
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # BCC-like grid
    initial_configs.append(_initial_grid_bcc())
    
    # FCC-like arrangement
    initial_configs.append(_initial_grid_fcc_like())
    
    # Simple cubic with slight random perturbation
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base_grid = grid[:N].copy()
    
    # Add small random perturbations to break symmetry
    rng = np.random.RandomState(42)
    for i in range(3):
        perturbed = base_grid.copy()
        perturbed += rng.uniform(-0.01, 0.01, size=perturbed.shape)
        # Clip to valid range
        perturbed = np.clip(perturbed, -0.49, 0.49)
        initial_configs.append(perturbed)
    
    # Another perturbation seed
    rng2 = np.random.RandomState(123)
    perturbed2 = base_grid.copy()
    perturbed2 += rng2.uniform(-0.02, 0.02, size=perturbed2.shape)
    perturbed2 = np.clip(perturbed2, -0.49, 0.49)
    initial_configs.append(perturbed2)
    
    # Evaluate each configuration and optimize
    best_centers = None
    best_score = -1.0
    
    for i, config in enumerate(initial_configs):
        # Check time
        if time.time() - global_start > DEADLINE - 30:
            break
        
        # Get baseline score
        base_score = _score(config)
        
        # Optimize
        time_left = DEADLINE - 20 - (time.time() - global_start)
        if time_left < 5:
            break
            
        optimized = _optimize_from_start(config, time_limit=max(5, time_left - 15))
        
        # Polish
        if time.time() - global_start < DEADLINE - 15:
            optimized = _polish(optimized, time_limit=10)
        
        # Get score
        score = _score(optimized)
        
        if score > best_score:
            best_score = score
            best_centers = optimized
    
    # If we haven't found anything good, use the best initial config
    if best_centers is None:
        best_centers = initial_configs[0]
    
    # Final safety check: ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
