"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    n = len(centers)
    # Distance to walls
    wall_dist = np.min(np.abs(centers), axis=1)
    wall_radius = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_radius)
    
    # Pairwise distances
    if n > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
    else:
        r_pair = r_wall
    
    return min(r_pair, r_wall)


def _soft_min_objective(latent, beta, n_spheres):
    """
    Soft-min objective for differentiable optimization.
    latent: (n_latent,) parameter vector
    beta: temperature for soft-min approximation
    """
    # Generate centers from latent via MLP
    centers = _mlp_forward(latent, n_spheres)
    
    # Compute distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Soft-min of all constraints
    # Combine pair distances (divided by 2) and wall distances
    pair_radii = dists / 2.0
    all_constraints = np.concatenate([pair_radii.ravel(), wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    min_val = np.min(all_constraints)
    max_val = np.max(all_constraints)
    range_val = max_val - min_val
    if range_val < 1e-10:
        return -min_val
    shifted = all_constraints - min_val
    exp_vals = np.exp(-beta * shifted)
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    return -soft_min


def _mlp_forward(latent, n_spheres):
    """
    Forward pass through a small MLP to generate sphere centers.
    latent: (n_latent,) input
    Returns: (n_spheres, 3) centers in [-0.5, 0.5]^3
    """
    # Fixed random weights (initialized once)
    global _W1, _b1, _W2, _b2, _W3, _b3
    if _W1 is None:
        np.random.seed(42)
        n_latent = len(latent)
        hidden1 = 32
        hidden2 = 16
        _W1 = np.random.randn(n_latent, hidden1) * 0.1
        _b1 = np.zeros(hidden1)
        _W2 = np.random.randn(hidden1, hidden2) * 0.1
        _b2 = np.zeros(hidden2)
        _W3 = np.random.randn(hidden2, n_spheres * 3) * 0.1
        _b3 = np.zeros(n_spheres * 3)
    
    # Forward pass
    h1 = np.tanh(latent @ _W1 + _b1)
    h2 = np.tanh(h1 @ _W2 + _b2)
    out = np.tanh(h2 @ _W3 + _b3)
    
    centers = out.reshape(n_spheres, 3)
    return centers


def _build_bcc_init(n_spheres):
    """Build BCC (body-centered cubic) initialization."""
    # BCC has 2 points per unit cell
    # For 100 spheres, we need about 50 cells
    # Use a grid approach with BCC-like structure
    
    # Start with a grid and add center points
    m = int(np.ceil(np.cbrt(n_spheres / 2) - 1e-9))
    if m < 2:
        m = 2
    
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                centers.append([x, y, z])
                # Body center
                if len(centers) < n_spheres:
                    centers.append([x + 0.5/m, y + 0.5/m, z + 0.5/m])
    
    centers = np.array(centers[:n_spheres])
    return centers


def _build_halton_init(n_spheres):
    """Build Halton sequence initialization for low-discrepancy points."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.zeros((n_spheres, 3))
    for i in range(1, n_spheres + 1):
        centers[i-1, 0] = halton(i, 2) - 0.5
        centers[i-1, 1] = halton(i, 3) - 0.5
        centers[i-1, 2] = halton(i, 5) - 0.5
    
    return centers


def _optimize_from_initial(initial_centers, deadline, beta_start=10.0, beta_end=100.0):
    """
    Optimize from an initial configuration using neural network parameterization.
    """
    n_spheres = len(initial_centers)
    n_latent = 20  # Compact latent representation
    
    # Initialize latent vector randomly
    np.random.seed(0)
    latent0 = np.random.randn(n_latent) * 0.5
    
    best_radius = _compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    # Try multiple beta values
    betas = [beta_start, beta_start * 2, beta_start * 4, beta_end]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _soft_min_objective,
                latent0,
                args=(beta, n_spheres),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_radius * -1:
                new_centers = _mlp_forward(result.x, n_spheres)
                new_radius = _compute_radius(new_centers)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                    latent0 = result.x.copy()
                    
        except Exception:
            continue
    
    # Local refinement with direct coordinate optimization
    if time.time() > deadline - 5:
        return best_centers
    
    # Try a few local perturbations
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        
        perturbed = best_centers + np.random.randn(n_spheres, 3) * 0.01
        # Clip to valid range
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        # Simple local optimization
        def local_obj(x):
            c = x.reshape(n_spheres, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            wall = 0.5 - np.max(np.abs(c), axis=1)
            pair_r = np.min(dists) / 2
            wall_r = np.min(wall)
            return -min(pair_r, wall_r)
        
        try:
            res = minimize(local_obj, perturbed.ravel(), method='L-BFGS-B', 
                         options={'maxiter': 100, 'ftol': 1e-12})
            new_centers = res.x.reshape(n_spheres, 3)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    return best_centers


# Global MLP weights (lazy initialization)
_W1 = None
_b1 = None
_W2 = None
_b2 = None
_W3 = None
_b3 = None


def construct():
    """
    Construct 100 sphere centers in the unit cube using neural network
    parameterization and L-BFGS-B optimization.
    """
    global _W1, _b1, _W2, _b2, _W3, _b3
    _W1 = _b1 = _W2 = _b2 = _W3 = _b3 = None
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Multiple initializations
    inits = [
        ('bcc', _build_bcc_init(N)),
        ('halton', _build_halton_init(N)),
    ]
    
    best_centers = None
    best_radius = -np.inf
    
    for name, init in inits:
        if time.time() > deadline - 30:
            break
        
        centers = _optimize_from_initial(init, deadline)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # If no valid result, fallback to grid
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
