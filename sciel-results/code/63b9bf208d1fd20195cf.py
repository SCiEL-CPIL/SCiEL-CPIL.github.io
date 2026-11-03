"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_lattice(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    if len(pts) < 2:
        return 0.0
    # Min distance between any pair / 2
    min_pair_dist = np.inf
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d < min_pair_dist:
                min_pair_dist = d
    # Min distance to boundary
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _nn_forward(z, W1, b1, W2, b2):
    """Neural network forward pass: z -> hidden -> output (N, 2)."""
    h = np.tanh(z @ W1 + b1)
    return h @ W2 + b2


def _nn_params_to_points(params, hidden_dim):
    """Unpack parameters into NN weights and compute points."""
    n = N
    z_dim = 10  # latent dimension
    w1_size = z_dim * hidden_dim
    b1_size = hidden_dim
    w2_size = hidden_dim * 2
    b2_size = 2
    
    idx = 0
    W1 = params[idx:idx + w1_size].reshape(z_dim, hidden_dim)
    idx += w1_size
    b1 = params[idx:idx + b1_size]
    idx += b1_size
    W2 = params[idx:idx + w2_size].reshape(hidden_dim, 2)
    idx += w2_size
    b2 = params[idx:idx + b2_size]
    idx += b2_size
    z = params[idx:idx + z_dim]
    idx += z_dim
    
    return _nn_forward(z, W1, b1, W2, b2), (W1, b1, W2, b2, z)


def _soft_min_loss(params, hidden_dim, temperature):
    """Compute soft-min packing radius loss (negative for maximization)."""
    pts, _ = _nn_params_to_points(params, hidden_dim)
    
    # Compute all pairwise distances
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of pairwise distances / 2
    min_pair = -np.logaddexp.reduce(np.log(1e-10) - dists / (2 * temperature)) * temperature
    
    # Boundary constraint
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    min_boundary = -np.logaddexp.reduce(np.log(1e-10) - boundary_dists / temperature) * temperature
    
    # Overall soft-min radius
    soft_r = -np.logaddexp(np.log(1e-10) - min_pair / temperature,
                           np.log(1e-10) - min_boundary / temperature) * temperature
    
    # Penalty for points outside or too close to boundary
    outside_penalty = np.sum(np.maximum(0, radii - 1.0) ** 2) * 1000
    too_close_penalty = np.sum(np.maximum(0, 0.01 - boundary_dists) ** 2) * 1000
    
    return -(soft_r - outside_penalty - too_close_penalty)


def _soft_min_grad(params, hidden_dim, temperature, eps=1e-7):
    """Numerical gradient of soft-min loss."""
    grad = np.zeros_like(params)
    base_loss = _soft_min_loss(params, hidden_dim, temperature)
    for i in range(len(params)):
        params[i] += eps
        loss_plus = _soft_min_loss(params, hidden_dim, temperature)
        params[i] -= 2 * eps
        loss_minus = _soft_min_loss(params, hidden_dim, temperature)
        params[i] += eps
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def _optimize_nn(params, hidden_dim, deadline):
    """Optimize NN parameters using L-BFGS-B."""
    start_time = time.time()
    
    # Start with moderate temperature and anneal
    temperatures = [0.05, 0.03, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    best_params = params.copy()
    best_loss = _soft_min_loss(params, hidden_dim, 0.05)
    
    for temp in temperatures:
        if time.time() > deadline:
            break
        
        # Use analytical gradient via finite differences for L-BFGS-B
        def objective(p):
            return _soft_min_loss(p, hidden_dim, temp)
        
        def gradient(p):
            return _soft_min_grad(p, hidden_dim, temp)
        
        # Optimize with L-BFGS-B
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 50, 'maxfun': 200, 'gtol': 1e-4, 'ftol': 1e-8}
        )
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x.copy()
        
        params = result.x.copy()
    
    return best_params, best_loss


def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Get initial hexagonal packing
    best_pts = None
    best_r = 0.0
    
    for spacing in np.linspace(0.09, 0.13, 40):
        if time.time() > deadline:
            break
        pts = _hex_lattice(spacing)
        if len(pts) < N:
            continue
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                best_r = _compute_radius(best_pts)
                break
            spacing *= 0.98
    
    # Direct local optimization to improve the hex packing
    if best_pts is not None and time.time() < deadline - 10.0:
        # Reshape points to 1D for optimization
        x0 = best_pts.flatten()
        
        def objective(x):
            pts = x.reshape(-1, 2)
            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Compute boundary distances
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_dists = 1.0 - radii
            
            # Soft-min of pairwise distances
            # Use a small temperature for sharpness
            temp = 0.01
            # Log-sum-exp for min: -log(sum(exp(-x/temp))) * temp
            # To avoid underflow, shift by min
            min_pair_raw = np.min(dists)
            log_sum_exp_pair = np.log(np.sum(np.exp(-(dists - min_pair_raw) / temp)))
            soft_min_pair = min_pair_raw - temp * log_sum_exp_pair
            
            # Soft-min of boundary distances
            min_bound_raw = np.min(boundary_dists)
            log_sum_exp_bound = np.log(np.sum(np.exp(-(boundary_dists - min_bound_raw) / temp)))
            soft_min_bound = min_bound_raw - temp * log_sum_exp_bound
            
            # The radius is the min of the two soft-mins
            # We want to maximize this, so minimize the negative
            # r = min(soft_min_pair / 2, soft_min_bound)
            # To make it smooth, use soft-min of these two values as well?
            # Or just take the min. The min is non-smooth.
            # Let's use a soft-min of the two values.
            val1 = soft_min_pair / 2.0
            val2 = soft_min_bound
            min_val_raw = np.min([val1, val2])
            log_sum_exp_final = np.log(np.exp(-(val1 - min_val_raw) / temp) + np.exp(-(val2 - min_val_raw) / temp))
            soft_r = min_val_raw - temp * log_sum_exp_final
            
            # Penalty for points going outside (though bounds should prevent this, 
            # soft-min might push them out if not careful, but boundary_dist becomes negative)
            # The soft-min of boundary_dists will be negative if any point is outside.
            # Minimizing -soft_r will try to make soft_r large, which requires boundary_dists > 0.
            
            return -soft_r
        
        # Define bounds for x and y: [-1, 1]
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < objective(x0):
                new_pts = result.x.reshape(-1, 2)
                # Check validity
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception as e:
            pass # Ignore optimization errors and keep initial hex packing

    # Final safety check: ensure shape is (N, 2)
    if best_pts is None:
        # Fallback: create a grid if hex failed completely
        # This should not happen, but better than None
        best_pts = np.zeros((N, 2))
        
    if best_pts.shape != (N, 2):
        # Reshape or pad if necessary
        if best_pts.size >= 2 * N:
            best_pts = best_pts[:2 * N].reshape(N, 2)
        else:
            # Pad with zeros (invalid, but prevents crash)
            temp = np.zeros((N, 2))
            temp[:best_pts.size // 2, :] = best_pts.reshape(-1, 2) if best_pts.size > 0 else 0
            best_pts = temp

    return best_pts
# EVOLVE-BLOCK-END
