import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _exact_radius(c):
    """Calculate the exact maximum radius for a given set of centers."""
    # c is (N, 3)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    # min_k (0.5 - |c_ik|)
    max_coord_abs = np.max(np.abs(c), axis=1)
    min_boundary_dist = 0.5 - np.max(max_coord_abs)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _soft_min_radius_and_grad(c, beta):
    """
    Calculate the negative of the soft-min radius and its gradient.
    We minimize -R_beta(c) to maximize R_beta(c).
    R_beta(c) = -1/beta * log( sum_{i<j} exp(-beta * (d_ij/2)^2) + sum_{i,k} exp(-beta * b_ik^2) )
    """
    N = c.shape[0]
    
    # Pairwise terms
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # We only care about i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Exponent term for pairs: exp(-beta * dist_sq / 4)
    # To avoid underflow/overflow issues with large beta, we can scale, 
    # but for beta up to 10000 and dist_sq ~ 0.01, exponent is ~ -25, which is fine.
    # However, summing many small numbers might lose precision.
    # Let's compute the log-sum-exp properly if needed, but for now direct sum is okay for moderate beta.
    
    exp_pair = np.exp(-beta * dist_sq / 4.0)
    # Zero out diagonal and lower triangle
    exp_pair[~mask] = 0.0
    
    sum_pair = np.sum(exp_pair)
    
    # Boundary terms
    abs_c = np.abs(c)
    dist_bound = 0.5 - abs_c
    # Clip to avoid negative values if center is outside (shouldn't happen with bounds, but safe)
    dist_bound = np.maximum(dist_bound, 1e-12)
    
    exp_bound = np.exp(-beta * dist_bound**2)
    sum_bound = np.sum(exp_bound)
    
    S = sum_pair + sum_bound
    
    # R_beta = -1/beta * log(S)
    # We want to minimize J = -R_beta = 1/beta * log(S)
    J = (1.0 / beta) * np.log(S)
    
    # Gradient of J
    # dJ/dx = 1/beta * (1/S) * dS/dx
    # dS/dx = d(sum_pair)/dx + d(sum_bound)/dx
    
    grad = np.zeros_like(c)
    
    # Gradient of pair term
    # d/d(c_i) exp(-beta * ||c_i - c_j||^2 / 4) = exp(...) * (-beta/4) * 2 * (c_i - c_j) = -beta/2 * (c_i - c_j) * exp(...)
    # Sum over j != i
    # grad_pair_i = sum_{j != i} -beta/2 * (c_i - c_j) * exp_pair[i,j]
    
    # Note: exp_pair is zero for i>=j. We need the full symmetric contribution or just sum over j>i and double?
    # The term in S is sum_{i<j} exp_pair[i,j].
    # d/d(c_i) sum_{j>i} exp_pair[i,j] + d/d(c_i) sum_{j<i} exp_pair[j,i]
    # Since exp_pair[j,i] = exp_pair[i,j], the gradient is symmetric.
    # Let's compute the full matrix gradient contribution.
    # G_ij = -beta/2 * diff[i,j] * exp_pair[i,j]
    # grad_i = sum_{j != i} G_ij
    
    # We have exp_pair with zeros for i>=j. We need the full matrix.
    exp_pair_full = exp_pair + exp_pair.T
    
    # Gradient contribution from pairs
    # weighted_diff = diff * (-beta/2) * exp_pair_full[:,:,None]
    weighted_diff = diff * (-beta / 2.0) * exp_pair_full[:, :, np.newaxis]
    
    # Sum over j for each i
    grad_pair = np.sum(weighted_diff, axis=1)
    
    # Gradient of boundary term
    # d/d(c_ik) exp(-beta * (0.5 - |c_ik|)^2)
    # = exp(...) * (-beta) * 2 * (0.5 - |c_ik|) * d(0.5 - |c_ik|)/d(c_ik)
    # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    # So term = exp(...) * (-beta) * 2 * dist_bound * (-sign(c_ik))
    # = 2 * beta * dist_bound * sign(c_ik) * exp_bound
    
    grad_bound = 2.0 * beta * dist_bound * np.sign(c) * exp_bound
    
    grad = grad_pair + grad_bound
    
    # Scale by 1/(beta * S)
    grad = grad / (beta * S)
    
    return J, grad.flatten()

def _bcc_lattice_init():
    """Generate initial points from a BCC lattice."""
    # BCC: Simple cubic grid + body center
    # Side length 1.
    # We need 50 points.
    # Let's try a grid of size 4x4x4 = 64 points, take 50.
    # Or 3x3x3 = 27 + centers.
    # Let's use a 4x4x4 grid of BCC unit cells? No, just points.
    
    # Create a 4x4x4 grid of simple cubic points
    n = 4
    spacing = 1.0 / n
    centers = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                centers.append((x, y, z))
                # Body center of the cell
                if i < n-1 and j < n-1 and k < n-1:
                    bx = x + spacing/2
                    by = y + spacing/2
                    bz = z + spacing/2
                    centers.append((bx, by, bz))
    
    centers = np.array(centers)
    # Shuffle and take 50
    idx = np.arange(len(centers))
    np.random.shuffle(idx)
    selected = centers[idx[:N]]
    # Add small noise
    noise = np.random.normal(0, 0.005, selected.shape)
    return selected + noise

def _fcc_lattice_init():
    """Generate initial points from an FCC lattice."""
    # FCC: Simple cubic grid + face centers
    n = 4
    spacing = 1.0 / n
    centers = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                centers.append((x, y, z))
                
                # Face centers
                if i < n-1 and j < n-1:
                    centers.append((x + spacing/2, y + spacing/2, z))
                if i < n-1 and k < n-1:
                    centers.append((x + spacing/2, y, z + spacing/2))
                if j < n-1 and k < n-1:
                    centers.append((x, y + spacing/2, z + spacing/2))
    
    centers = np.array(centers)
    idx = np.arange(len(centers))
    np.random.shuffle(idx)
    selected = centers[idx[:N]]
    noise = np.random.normal(0, 0.005, selected.shape)
    return selected + noise

def _cubic_grid_init():
    """Generate initial points from a simple cubic grid with jitter."""
    n = 4
    spacing = 1.0 / n
    centers = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                centers.append((x, y, z))
                
                # Alternate layers with shift?
                if (i + j + k) % 2 == 1:
                     centers.append((x + spacing/2, y + spacing/2, z + spacing/2))
                     
    centers = np.array(centers)
    idx = np.arange(len(centers))
    np.random.shuffle(idx)
    selected = centers[idx[:N]]
    noise = np.random.normal(0, 0.01, selected.shape)
    return selected + noise

def _random_init():
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _optimize_from_init(x0, time_limit):
    """
    Run optimization starting from x0, increasing beta.
    Returns best centers and best radius.
    """
    best_centers = x0.copy()
    best_radius = _exact_radius(x0)
    
    betas = [10, 50, 200, 1000, 5000]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        def objective(x):
            val, grad = _soft_min_radius_and_grad(x.reshape(N, 3), beta)
            return val, grad
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            
            if res.success:
                c_res = res.x.reshape(N, 3)
                r_res = _exact_radius(c_res)
                if r_res > best_radius:
                    best_radius = r_res
                    best_centers = c_res
                x0 = c_res
        except Exception:
            pass
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    best_centers_global = None
    best_radius_global = 0.0
    
    # List of initialization functions
    inits = [
        _bcc_lattice_init,
        _fcc_lattice_init,
        _cubic_grid_init,
        _random_init,
        _random_init,
        _bcc_lattice_init,
        _fcc_lattice_init,
        _cubic_grid_init,
    ]
    
    for i, init_func in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init_func()
        x0 = np.clip(x0, -0.49, 0.49)
        
        c_res, r_res = _optimize_from_init(x0, deadline)
        
        if r_res > best_radius_global:
            best_radius_global = r_res
            best_centers_global = c_res
            
        # Print progress
        print(f"Init {i}: radius {best_radius_global:.6f}")

    if best_centers_global is None:
        # Fallback
        ticks = np.linspace(-0.45, 0.45, 5)
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers_global = grid[:N]
        
    return best_centers_global
