import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hex_lattice_init():
    """
    Generate a hexagonal lattice initial packing.
    This provides a high-density structured start.
    """
    points = []
    # Use a spacing that is slightly larger than optimal to ensure validity initially
    # Optimal r ~ 0.125, so spacing ~ 0.25. Let's use 0.26 for safety.
    spacing = 0.26 
    sqrt3_2 = np.sqrt(3) / 2.0
    
    # Generate grid
    # Range needs to cover the unit circle
    # Max radius 1, so x, y in [-1.2, 1.2]
    # i, j indices
    for j in range(-6, 7):
        for i in range(-6, 7):
            x = i * spacing + (j % 2) * spacing / 2.0
            y = j * spacing * sqrt3_2
            if np.hypot(x, y) < 1.0 - spacing / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # Sort by distance to center to pick the most central N points
    if len(points) >= N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        return points[idx]
    
    # Fallback if not enough points (should not happen with this range)
    # Expand grid
    points = []
    spacing = 0.24
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = i * spacing + (j % 2) * spacing / 2.0
            y = j * spacing * sqrt3_2
            if np.hypot(x, y) < 1.0 - spacing / 2.0:
                points.append([x, y])
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists)[:N]
    return points[idx]

def _soft_min_energy_and_grad(x, beta):
    """
    Calculate the soft-minimum energy and its gradient.
    
    Energy = -softmin(pairwise_dists) + penalty_boundary
    
    We want to maximize the minimum distance. 
    Let D be the matrix of pairwise distances.
    Let B be the vector of distances to boundary.
    
    We define the objective to minimize as:
    E = - (1/beta) * log( sum_{i<j} exp(-beta * d_ij) )  -- wait, this minimizes the sum, not the min.
    
    Actually, to maximize the minimum, we can minimize a smooth approximation of the minimum.
    min(d) ≈ (1/beta) * log( sum exp(-beta * d) ) is a soft-min.
    So we want to MINIMIZE this value? No, we want to MAXIMIZE the min distance.
    So we want to MINIMIZE -softmin(d).
    
    Let's define:
    d_pairs: flattened upper triangle of distance matrix.
    d_bound: vector of distances to boundary.
    
    We want to maximize r = min(min(d_pairs)/2, min(d_bound)).
    This is equivalent to maximizing min(d_pairs, 2*d_bound).
    Let z = concat(d_pairs, 2*d_bound).
    We want to maximize min(z).
    
    Soft-min of z: s(z) = (1/beta) * log( sum_{k} exp(-beta * z_k) )
    As beta -> infinity, s(z) -> min(z).
    So we want to MAXIMIZE s(z).
    Therefore, we MINIMIZE -s(z).
    
    Objective: f(x) = -s(z)
    
    Gradient:
    s(z) = (1/beta) * log( sum_{k} exp(-beta * z_k) )
    d s / d z_k = (1/beta) * (1 / sum(exp)) * exp(-beta * z_k) * (-beta)
                = - exp(-beta * z_k) / sum(exp(-beta * z_j))
                = - p_k
    where p_k = exp(-beta * z_k) / sum(exp(-beta * z_j))
    
    So d f / d z_k = p_k
    
    Now we need d z_k / d x.
    
    z consists of:
    1. d_ij = ||x_i - x_j||
    2. 2 * d_i = 2 * (1 - ||x_i||)
    
    Let's compute gradients carefully.
    
    For d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2):
    d d_ij / d x_i = (x_i - x_j) / d_ij
    d d_ij / d x_j = (x_j - x_i) / d_ij
    
    For 2 * d_i = 2 * (1 - sqrt(x_i^2 + y_i^2)):
    d (2*d_i) / d x_i = -2 * x_i / ||x_i||
    d (2*d_i) / d y_i = -2 * y_i / ||x_i||
    
    So:
    grad_x = sum_k p_k * (d z_k / d x)
    
    Note: z_k includes all pairwise distances and all boundary distances (scaled by 2).
    """
    x = x.reshape(N, 2)
    
    # Pairwise distances
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    dists = np.sqrt(dists_sq + 1e-12) # avoid 0
    np.fill_diagonal(dists, np.inf)
    
    # Get upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs = dists[i_idx, j_idx]
    
    # Boundary distances
    norms = np.hypot(x[:, 0], x[:, 1])
    d_bound = 1.0 - norms
    
    # Scale boundary distances to be comparable to pairwise distances
    # The radius r is min(min(d_pairs)/2, min(d_bound)).
    # So the "value" we are maximizing is r.
    # Let's work with the variables directly related to r.
    # Let v_pairs = d_pairs / 2.0
    # Let v_bound = d_bound
    # We want to maximize min(v_pairs, v_bound).
    # Let z = concat(v_pairs, v_bound)
    
    v_pairs = d_pairs / 2.0
    v_bound = d_bound
    
    z = np.concatenate([v_pairs, v_bound])
    
    # Compute soft-min
    # To avoid overflow/underflow in exp, subtract min
    z_min = np.min(z)
    exp_z = np.exp(-beta * (z - z_min))
    sum_exp = np.sum(exp_z)
    
    if sum_exp == 0:
        # Fallback
        energy = -z_min
        grad = np.zeros_like(x)
        return energy, grad
        
    p = exp_z / sum_exp
    
    # Soft-min value
    s_z = z_min - (1.0 / beta) * np.log(sum_exp)
    
    # Objective to minimize: -s_z
    energy = -s_z
    
    # Gradient
    # d energy / d x = - d s_z / d x
    # d s_z / d z_k = -p_k
    # So d energy / d z_k = p_k
    
    grad_z = p
    
    # Now map grad_z back to grad_x
    grad_x = np.zeros_like(x)
    
    n_pairs = len(v_pairs)
    grad_pairs = grad_z[:n_pairs]
    grad_bound = grad_z[n_pairs:]
    
    # Boundary part
    # z_k = d_bound_i = 1 - ||x_i||
    # d z_k / d x_i = - x_i / ||x_i||
    for i in range(N):
        norm = norms[i]
        if norm > 1e-6:
            # grad_bound[i] is the weight for this boundary term
            grad_x[i] += grad_bound[i] * (-x[i] / norm)
    
    # Pairwise part
    # z_k = d_ij / 2
    # d z_k / d x_i = 0.5 * (x_i - x_j) / d_ij
    # d z_k / d x_j = 0.5 * (x_j - x_i) / d_ij
    
    for k in range(len(v_pairs)):
        i = i_idx[k]
        j = j_idx[k]
        w = grad_pairs[k]
        d = d_pairs[k]
        if d < 1e-6:
            continue
        diff = x[i] - x[j]
        # d (d_ij/2) / d x_i = 0.5 * diff / d
        grad_x[i] += w * 0.5 * diff / d
        # d (d_ij/2) / d x_j = 0.5 * (x_j - x_i) / d = -0.5 * diff / d
        grad_x[j] -= w * 0.5 * diff / d
        
    return energy, grad_x

def _physics_relax(centers, deadline):
    """
    Multi-stage optimization using L-BFGS-B.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Stages of beta (sharpness of soft-min)
    # Start low (smooth) to escape local minima, increase to refine
    betas = [50, 100, 200, 400, 800, 1600]
    
    time_start = time.time()
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        # Define objective and gradient for scipy
        def objective(x_flat):
            e, g = _soft_min_energy_and_grad(x_flat, beta)
            return e, g
            
        # L-BFGS-B supports gradient
        # We don't have explicit bounds because the penalty handles it,
        # but we can add loose bounds to help the optimizer.
        # Actually, the function is smooth everywhere except at 0 distance, which we handled.
        # Let's not use bounds to allow the optimizer to push points if needed,
        # relying on the energy function to penalize invalid states.
        # However, if points go outside, d_bound becomes negative, exp(-beta * neg) explodes.
        # So we should probably keep them somewhat close.
        # Let's just run it.
        
        res = minimize(
            objective, 
            current, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
        )
        
        if res.success or res.fun < 0: # res.fun is energy, we want it low (negative soft-min)
             new_centers = res.x.reshape(N, 2)
             new_r = _compute_radius(new_centers)
             
             if new_r > current_r:
                 current = new_centers
                 current_r = new_r
                 if current_r > best_r:
                     best = current.copy()
                     best_r = current_r
        else:
             # If optimization failed or didn't improve, try a smaller step or different beta?
             # For now, just keep current.
             pass
             
    return best, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with hexagonal lattice
    centers = _hex_lattice_init()
    
    # Ensure initial validity
    r_init = _compute_radius(centers)
    if r_init <= 0:
        # Fallback: try concentric if hex fails (unlikely)
        ring_counts = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        centers = []
        for k, count in enumerate(ring_counts):
            r_k = radii[k]
            if count == 1:
                centers.append([0.0, 0.0])
            else:
                angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                if k % 2 == 1:
                    angles += np.pi / count
                for a in angles:
                    x = r_k * np.cos(a)
                    y = r_k * np.sin(a)
                    centers.append([x, y])
        centers = np.array(centers)
        r_init = _compute_radius(centers)
        
    # Run gradient-based optimization
    best_centers, best_r = _physics_relax(centers, deadline)
    
    return best_centers
