import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    iu = np.triu_indices(N, k=1)
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE with log-sum-exp trick
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid_init():
    """Build a 5x5x4 grid initialization."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    centers += np.random.normal(0, 0.01, centers.shape)
    return np.clip(centers, -0.49, 0.49)

def _build_bcc_init():
    """Build a body-centered cubic lattice initialization."""
    # BCC has 2 points per unit cell
    # We need ~100 points, so use a 4x4x4 grid of BCC cells = 128 points, take first 100
    # Or better: use a grid where we can place exactly 100
    # Let's use a 5x5x4 grid of BCC cells, but that gives 200 points
    # Instead, let's create a BCC-like pattern with 100 points
    
    # Alternative: Use a simple cubic grid with some points shifted to BCC positions
    # For simplicity, let's use a 4x5x5 grid with BCC-like offsets
    
    nx, ny, nz = 4, 5, 5
    spacing_x = 0.9 / (nx - 1) if nx > 1 else 1.0
    spacing_y = 0.9 / (ny - 1) if ny > 1 else 1.0
    spacing_z = 0.9 / (nz - 1) if nz > 1 else 1.0
    
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add BCC-like perturbation: shift some points diagonally
    # For each cell, add a point at the center
    # But we need exactly 100, so let's just use the grid with noise
    centers += np.random.normal(0, 0.02, centers.shape)
    return np.clip(centers, -0.49, 0.49)

def _build_halton_init():
    """Build a Halton sequence initialization for low-discrepancy sampling."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.zeros((N, 3))
    for i in range(N):
        centers[i, 0] = halton(i + 1, 2) - 0.5
        centers[i, 1] = halton(i + 1, 3) - 0.5
        centers[i, 2] = halton(i + 1, 5) - 0.5
    
    # Scale to fit in [-0.45, 0.45]
    centers *= 0.9
    
    # Add small noise
    centers += np.random.normal(0, 0.01, centers.shape)
    return np.clip(centers, -0.49, 0.49)

def _optimize_from_initial(centers, deadline, beta_start=10.0, beta_max=2000.0):
    """Run the soft-min optimization from an initial configuration."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta = beta_start
    
    while time.time() < deadline and beta < beta_max:
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < _soft_min_objective(x0, beta):
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass
        
        beta *= 1.5
        
        if time.time() > deadline:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """High-precision polish pass with tight tolerances."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use very high beta for precise optimization
    beta = 5000.0
    
    try:
        res = minimize(
            _soft_min_objective,
            best_centers.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < _soft_min_objective(best_centers.flatten(), beta):
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
    except Exception:
        pass
    
    return best_centers, best_radius

def _anneal(centers, deadline, T0=0.1, T_end=0.001, steps=500):
    """Simulated annealing to escape local minima."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        T = T0 * (T_end / T0) ** (step / steps)
        
        # Generate a random perturbation
        perturbation = np.random.normal(0, T, current_centers.shape)
        new_centers = current_centers + perturbation
        new_centers = np.clip(new_centers, -0.49, 0.49)
        
        new_radius = _compute_radius(new_centers)
        
        # Accept if better or with probability exp((new - old) / T)
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            prob = np.exp((new_radius - current_radius) / T)
            if np.random.random() < prob:
                current_centers = new_centers
                current_radius = new_radius
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try multiple initializations and keep the best
    initializations = [
        _build_grid_init(),
        _build_bcc_init(),
        _build_halton_init()
    ]
    
    best_centers = None
    best_radius = -np.inf
    
    for init in initializations:
        if time.time() > deadline - 20.0:
            break
        
        # Optimize from this initialization
        centers, radius = _optimize_from_initial(init, deadline - 5.0)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
    
    # Add simulated annealing to escape local minima
    if time.time() < deadline - 10.0 and best_centers is not None:
        centers, radius = _anneal(best_centers, deadline - 5.0)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
    
    # Final high-precision polish
    if time.time() < deadline - 2.0 and best_centers is not None:
        centers, radius = _polish(best_centers, deadline)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
    
    if best_centers is None:
        best_centers = _build_grid_init()
    
    return best_centers

# EVOLVE-BLOCK-END
