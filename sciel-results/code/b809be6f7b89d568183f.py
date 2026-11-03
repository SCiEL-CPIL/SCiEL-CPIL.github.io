"""Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distances
    # Use efficient computation: for each pair, compute distance
    # For N=100, we can use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy_and_grad(centers, beta=50.0):
    """
    Smooth surrogate: minimize a soft-min of all constraints.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Use soft-min: softmin(x_1,...,x_m) = -(1/beta) * log(sum exp(-beta * x_i))
    As beta -> large, softmin -> min.
    """
    centers = centers.reshape(N, 3)
    
    # Compute all pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Get all unique pairs (i < j)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0  # radius from pair constraint
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # All constraint values (radii)
    all_r = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min: we want to maximize the minimum, so minimize softmin
    # softmin(x) = -(1/beta) * log(sum(exp(-beta * x)))
    # For numerical stability, shift by max
    x = -beta * all_r
    x_max = np.max(x)
    log_sum = x_max + np.log(np.sum(np.exp(x - x_max)))
    
    softmin_val = -log_sum / beta
    return softmin_val  # We minimize this, which maximizes softmin(all_r)

def _grad_energy(centers, beta=50.0):
    """Gradient of the soft-min objective."""
    centers = centers.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    all_r = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min gradient
    # softmin(x) = -(1/beta) * log(sum(exp(-beta * x)))
    # d(softmin)/dx_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
    
    x = -beta * all_r
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x)
    weights = exp_x / sum_exp  # These are the softmax weights
    
    # Gradient with respect to centers
    grad = np.zeros_like(centers)
    
    n_pairs = len(i_idx)
    
    # Pairwise contributions (vectorized)
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        d = centers[i] - centers[j]
        dist = np.sqrt(np.sum(d * d))
        if dist < 1e-12:
            continue
        grad[i] -= w * d / (2.0 * dist)
        grad[j] += w * d / (2.0 * dist)
    
    # Wall contributions
    for i in range(N):
        w = weights[n_pairs + i]
        abs_c = np.abs(centers[i])
        k_max = np.argmax(abs_c)
        # dr/dc = -sign(c_k) for k=argmax
        grad[i, k_max] += w * (-np.sign(centers[i, k_max]))
    
    return grad

def _local_optimize(centers, max_time=15.0, beta_start=30.0, beta_end=200.0):
    """Run L-BFGS-B local optimization with increasing beta for better convergence."""
    start_time = time.time()
    n_betas = 3  # Number of beta stages
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for stage in range(n_betas):
        if time.time() > start_time + max_time:
            break
        
        # Interpolate beta from start to end
        beta = beta_start + (beta_end - beta_start) * (stage / max(1, n_betas - 1))
        
        def objective(x, b=beta):
            return _energy_and_grad(x.reshape(N, 3), b)
        
        def grad(x, b=beta):
            return _grad_energy(x.reshape(N, 3), b).flatten()
        
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        maxiter = 300 if stage < n_betas - 1 else 500
        ftol = 1e-10 if stage < n_betas - 1 else 1e-12
        
        result = minimize(
            objective,
            best_centers.flatten(),
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol, 'gtol': 1e-8}
        )
        
        new_centers = result.x.reshape(N, 3)
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
    
    elapsed = time.time() - start_time
    return best_centers, best_radius, elapsed

def _generate_initial_points(method='grid', rng=None):
    """Generate initial center positions."""
    if rng is None:
        rng = np.random.default_rng()
    
    if method == 'grid':
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    elif method == 'random':
        return rng.uniform(-0.45, 0.45, size=(N, 3))
    elif method == 'bcc':
        # Body-centered cubic arrangement
        # Generate a BCC lattice and select N points
        # BCC has 2 atoms per unit cell
        # For 100 spheres, we need about 50 cells
        # Let's use a 5x5x4 grid of BCC cells
        points = []
        for ix in range(5):
            for iy in range(5):
                for iz in range(4):
                    # Corner
                    x = (ix + 0.5) / 5.0 - 0.5
                    y = (iy + 0.5) / 5.0 - 0.5
                    z = (iz + 0.5) / 4.0 - 0.5
                    points.append([x, y, z])
                    # Center
                    xc = (ix + 1.0) / 5.0 - 0.5
                    yc = (iy + 1.0) / 5.0 - 0.5
                    zc = (iz + 1.0) / 4.0 - 0.5
                    points.append([xc, yc, zc])
        points = np.array(points[:N])
        return points
    else:
        return rng.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    """Construct centers of 100 equal spheres in [-0.5, 0.5]^3 to maximize radius."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initializations and local optimizations
    rng = np.random.default_rng(42)
    
    # First, try the simple grid as a baseline
    grid_centers = _generate_initial_points('grid', rng)
    centers, radius, elapsed = _local_optimize(grid_centers, max_time=20.0)
    if radius > best_radius:
        best_radius = radius
        best_centers = centers.copy()
    
    # Try BCC initialization
    if time.time() < deadline - 25.0:
        bcc_centers = _generate_initial_points('bcc', rng)
        centers, radius, elapsed = _local_optimize(bcc_centers, max_time=20.0)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Try random initializations with varying noise scales
    n_random_tries = 6
    for i in range(n_random_tries):
        if time.time() > deadline - 25.0:
            break
        
        # Add some perturbation to grid
        base = _generate_initial_points('grid', rng)
        noise_scale = 0.03 + 0.015 * i
        random_centers = base + rng.normal(0, noise_scale, size=(N, 3))
        # Clip to stay in bounds
        random_centers = np.clip(random_centers, -0.48, 0.48)
        
        remaining = deadline - 25.0 - time.time()
        time_per_try = max(10.0, min(20.0, remaining / max(1, n_random_tries - i)))
        
        centers, radius, elapsed = _local_optimize(random_centers, max_time=time_per_try)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final polish: run one more optimization with high beta for precision
    if time.time() < deadline - 8.0 and best_centers is not None:
        # Use a very high beta for final precision
        beta_final = 300.0
        
        def objective_high_beta(x):
            return _energy_and_grad(x.reshape(N, 3), beta_final)
        
        def grad_high_beta(x):
            return _grad_energy(x.reshape(N, 3), beta_final).flatten()
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        result = minimize(
            objective_high_beta,
            best_centers.flatten(),
            jac=grad_high_beta,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        polished_centers = result.x.reshape(N, 3)
        polished_radius = _compute_radius(polished_centers)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers.copy()
    
    # Ensure validity: clip to bounds and verify
    if best_centers is not None:
        best_centers = np.clip(best_centers, -0.5, 0.5)
        final_radius = _compute_radius(best_centers)
        if final_radius > 0:
            return best_centers
    
    # Fallback to grid if something went wrong
    return _generate_initial_points('grid', rng)
