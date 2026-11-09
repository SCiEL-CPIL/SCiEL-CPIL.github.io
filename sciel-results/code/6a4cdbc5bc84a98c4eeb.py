"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _radius(c):
    """Calculate the packing radius for a configuration c of shape (N, 2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Check if inside square [-0.5, 0.5]^2
    # min distance to wall
    x = c[:, 0]
    y = c[:, 1]
    min_x = np.min(np.minimum(0.5 - x, x + 0.5))
    min_y = np.min(np.minimum(0.5 - y, y + 0.5))
    min_wall = min(min_x, min_y)
    
    if min_wall < 0:
        return 0.0
        
    # Min distance between pairs
    dists = pdist(c)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair) / 2.0

def _energy(c_flat, N):
    """
    Energy function to minimize.
    c_flat is a vector of length 2N.
    We minimize the sum of inverse distances to encourage separation.
    """
    c = c_flat.reshape(N, 2)
    
    # Pairwise distances
    dists = pdist(c)
    # Avoid division by zero
    dists = np.maximum(dists, 1e-6)
    
    # Repulsive potential
    # Using 1/d^2 is a good balance. 1/d^12 is very steep.
    # Let's try 1/d^4 for smoothness but strong repulsion.
    energy_pairs = np.sum(1.0 / (dists ** 4))
    
    # Wall penalties
    x = c[:, 0]
    y = c[:, 1]
    # Distance to nearest wall
    dx = np.minimum(0.5 - x, x + 0.5)
    dy = np.minimum(0.5 - y, y + 0.5)
    d_wall = np.minimum(dx, dy)
    d_wall = np.maximum(d_wall, 1e-6)
    
    energy_walls = np.sum(1.0 / (d_wall ** 4))
    
    return energy_pairs + energy_walls

def _gradient(c_flat, N):
    """
    Numerical gradient or analytical?
    L-BFGS-B can use numerical if we don't provide jac.
    Providing analytical jac is faster.
    Let's provide numerical first for simplicity, or implement analytical.
    Analytical for 1/d^4:
    d(1/d^4)/dx_i = -4/d^5 * (x_i - x_j)
    """
    c = c_flat.reshape(N, 2)
    grad = np.zeros_like(c)
    
    # Pairwise
    # dists is upper triangle
    # We need to sum gradients for each i
    # For each pair (i, j), i < j
    # dE/dx_i += -4/d^5 * (x_i - x_j)
    # dE/dx_j += -4/d^5 * (x_j - x_i) = 4/d^5 * (x_i - x_j)
    
    # Efficient vectorization is tricky with pdist.
    # Let's use a simple loop for gradient calculation? N=50, 1225 pairs.
    # 1225 * 100 iterations * 100 steps might be slow in Python.
    # Better to use numerical gradient from scipy?
    # Or implement vectorized gradient.
    
    # Let's stick to numerical gradient provided by scipy.optimize.minimize 
    # unless we are sure of the speed.
    # Actually, let's just use numerical. It's 100 variables.
    # Finite difference gradient takes 100 function evaluations per step.
    # 100 steps * 100 evals * 1225 pairs * 10 starts = 12,250,000 pair calcs.
    # pdist is fast. This should be okay within 100s?
    # 12M simple operations in NumPy is fast.
    
    # Let's try to implement analytical gradient to be safe on time.
    
    # Calculate all pairwise distances and differences
    # diff[i,j] = c[i] - c[j]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_masked = np.where(mask, dists, 1e-6)
    
    # Factor for 1/d^4 derivative: -4 * d^-5 * diff
    # E = sum_{i<j} (1/d_ij^4)
    # dE/dc_i = sum_{j != i} -4 * d_ij^-5 * (c_i - c_j)
    # Note: sum over j != i includes both (i,j) and (j,i) in the full matrix sum?
    # No, E is sum over i<j.
    # dE/dc_i = sum_{j>i} -4 d^-5 (c_i - c_j) + sum_{j<i} 4 d^-5 (c_i - c_j)
    # This is equivalent to sum_{j != i} -4 d^-5 (c_i - c_j) / 2 ?
    # Let's check:
    # Term in E for pair (i,j): 1/d^4.
    # d(1/d^4)/dc_i = -4 d^-5 (c_i - c_j)
    # d(1/d^4)/dc_j = -4 d^-5 (c_j - c_i) = 4 d^-5 (c_i - c_j)
    
    # So for each i, grad_i = sum_{j != i} -4 d_ij^-5 (c_i - c_j) * 0.5 ?
    # No.
    # If we sum over all j != i, we count each pair twice in the energy sum if we just sum the matrix.
    # But our energy function `pdist` only sums i<j.
    # So dE/dc_i = sum_{j>i} (-4 d^-5 (c_i - c_j)) + sum_{j<i} (4 d^-5 (c_i - c_j))
    # This is NOT simply sum_{j != i} ...
    
    # Let's just use numerical gradient to avoid bugs.
    # It is safer.
    return None # Indicate to use numerical

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s margin
    
    best_c = None
    best_score = 0.0
    
    # Initializations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Jittered Grid
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.01, size=(N, 2))
    inits.append(grid[:N] + jitter)
    
    # 3. Random
    for _ in range(5):
        random_c = rng.uniform(-0.5, 0.5, size=(N, 2))
        inits.append(random_c)
        
    # 4. Hexagonal Lattice (approx)
    # Generate hex lattice points
    hex_points = []
    s = 0.15 # spacing
    for row in range(-10, 10):
        for col in range(-10, 10):
            x = col * s + (row % 2) * s * 0.5
            y = row * s * 0.866
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                hex_points.append((x, y))
    hex_points = np.array(hex_points)
    if len(hex_points) >= N:
        # Randomly select N points? Or just first N?
        # First N might be biased. Let's shuffle.
        idx = rng.choice(len(hex_points), N, replace=False)
        inits.append(hex_points[idx])
        
    for c_init in inits:
        if time.time() > deadline:
            break
            
        # Flatten for optimizer
        x0 = c_init.flatten()
        
        # Constraints: keep inside box?
        # The energy function penalizes going outside, but it's not a hard constraint.
        # L-BFGS-B supports bounds.
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                _energy, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            c_opt = res.x.reshape(N, 2)
            
            # Calculate score
            score = _radius(c_opt)
            
            if score > best_score:
                best_score = score
                best_c = c_opt
                
        except Exception as e:
            pass
            
    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c

# EVOLVE-BLOCK-END
