"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _potential(centers_flat, power=6):
    """
    Potential energy function to minimize.
    Minimizing this maximizes the minimum distance.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    dists_sq[dists_sq < 1e-12] = 1e-12
    
    # Sum over i < j
    iu = np.triu_indices(N, k=1)
    # Use a high power to emphasize close pairs
    V_pair = np.sum(1.0 / (dists_sq[iu[0], iu[1]] ** (power / 2)))
    
    # Wall repulsion
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dists = np.min(wall_dists, axis=1)
    min_wall_dists[min_wall_dists < 1e-12] = 1e-12
    V_wall = np.sum(1.0 / (min_wall_dists ** power))
    
    return V_pair + V_wall

def _gradient_potential(centers_flat, power=6):
    """
    Analytical gradient of the potential.
    """
    centers = centers_flat.reshape(-1, 3)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # V_pair = sum_{i<j} d_ij^-p
    # dV/dc_i = sum_{j!=i} -p * d_ij^(-p-2) * (c_i - c_j)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists_sq[dists_sq < 1e-12] = 1e-12
    dists = np.sqrt(dists_sq)
    
    # coeff = -p / d^(p+2) = -p * d^(-p-2)
    # d^2 = dists_sq. d^(p+2) = (d^2)^(p/2 + 1)
    coeff = -power / (dists_sq ** (power / 2.0 + 1.0))
    np.fill_diagonal(coeff, 0.0)
    
    grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pair
    
    # Wall gradient
    # V_wall = sum_i (w_i)^-p where w_i = min_k (0.5 - |c_ik|)
    # We approximate the gradient by taking the derivative of the minimum term.
    d_x_pos = np.maximum(0.5 - centers[:, 0], 1e-12)
    d_x_neg = np.maximum(0.5 + centers[:, 0], 1e-12)
    d_y_pos = np.maximum(0.5 - centers[:, 1], 1e-12)
    d_y_neg = np.maximum(0.5 + centers[:, 1], 1e-12)
    d_z_pos = np.maximum(0.5 - centers[:, 2], 1e-12)
    d_z_neg = np.maximum(0.5 + centers[:, 2], 1e-12)
    
    dists_matrix = np.stack([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg], axis=1)
    min_idx = np.argmin(dists_matrix, axis=1)
    min_dists = np.min(dists_matrix, axis=1)
    
    grad_wall = np.zeros_like(centers)
    
    for i in range(N):
        idx = min_idx[i]
        d = min_dists[i]
        # dV/dw = -p * w^(-p-1)
        # dw/dc = +/- 1
        c = power / (d ** (power + 1))
        
        if idx == 0: # x_pos (0.5 - c_x) -> dw/dc = -1 -> dV/dc = +c
            grad_wall[i, 0] += c
        elif idx == 1: # x_neg (0.5 + c_x) -> dw/dc = +1 -> dV/dc = -c
            grad_wall[i, 0] -= c
        elif idx == 2: # y_pos
            grad_wall[i, 1] += c
        elif idx == 3: # y_neg
            grad_wall[i, 1] -= c
        elif idx == 4: # z_pos
            grad_wall[i, 2] += c
        elif idx == 5: # z_neg
            grad_wall[i, 2] -= c
            
    grad += grad_wall
    
    return grad.flatten()

def _refine_with_nm(centers, max_iter=1000, timeout=10.0):
    """
    Refine the configuration using Nelder-Mead on the negative radius.
    This helps escape the smoothness assumptions of L-BFGS and fine-tune the min distance.
    """
    start_time = time.time()
    
    def neg_radius(c_flat):
        c = c_flat.reshape(-1, 3)
        # Quick bounds check to avoid huge penalties or invalid states
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1000.0
        return -_score(c)
    
    x0 = centers.flatten()
    try:
        res = minimize(neg_radius, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if time.time() - start_time > timeout:
            return centers
        opt_centers = res.x.reshape(-1, 3)
        opt_centers = np.clip(opt_centers, -0.5, 0.5)
        return opt_centers
    except Exception:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Pre-compute indices for speed
    iu = np.triu_indices(N, k=1)
    
    # Strategy 1: Grid Pruning
    # 4x4x4 grid = 64 points. Remove 14.
    # We try multiple random removals.
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    for restart in range(30):
        if time.time() > deadline:
            break
            
        # 1. Generate initial config
        if restart < 10:
            # Grid Pruning: Randomly remove 14 points from 64
            idx = rng.permutation(len(grid))[:N]
            centers = grid[idx].copy()
            # Small jitter to break symmetry
            centers += rng.normal(0, 0.005, centers.shape)
        elif restart < 20:
            # FCC-like initialization
            # FCC basis in unit cell: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
            # Scale to fit in [-0.5, 0.5]
            # Let's generate a 3x3x3 lattice of FCC cells? 
            # Simpler: Just use a dense grid and optimize.
            # Let's try a 5x5x5 grid (125 points) and pick 50 closest to center?
            ticks_fine = np.linspace(-0.45, 0.45, 5)
            grid_fine = np.array([(x, y, z) for z in ticks_fine for y in ticks_fine for x in ticks_fine])
            # Sort by distance to center
            d_center = np.linalg.norm(grid_fine, axis=1)
            idx_sorted = np.argsort(d_center)
            # Take 50 from the center, but add jitter
            idx = idx_sorted[:N]
            centers = grid_fine[idx].copy()
            centers += rng.normal(0, 0.01, centers.shape)
        else:
            # Random initialization
            centers = rng.uniform(-0.45, 0.45, (N, 3))
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # 2. Optimize with L-BFGS-B
        x0 = centers.flatten()
        
        try:
            # Use a higher power for repulsion to focus on minimum distance
            res = minimize(_potential, x0, method='L-BFGS-B', jac=lambda x: _gradient_potential(x, power=8), 
                           bounds=[(-0.5, 0.5)] * (N * 3), options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            opt_centers = res.x.reshape(-1, 3)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            
            score = _score(opt_centers)
            
            # 3. Refine with Nelder-Mead if time permits
            if score > best_score and (deadline - time.time()) > 15:
                refined_centers = _refine_with_nm(opt_centers, timeout=10)
                refined_score = _score(refined_centers)
                if refined_score > score:
                    opt_centers = refined_centers
                    score = refined_score
            
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
                
        except Exception as e:
            pass

    # If time remains, do a final polish on the best candidate
    if best_centers is not None and (deadline - time.time()) > 20:
        # Run L-BFGS again on the best candidate with tighter tolerance
        x0 = best_centers.flatten()
        try:
            res = minimize(_potential, x0, method='L-BFGS-B', jac=lambda x: _gradient_potential(x, power=10), 
                           bounds=[(-0.5, 0.5)] * (N * 3), options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            opt_centers = res.x.reshape(-1, 3)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            score = _score(opt_centers)
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
        except Exception:
            pass
        
        # Final Nelder-Mead polish
        if (deadline - time.time()) > 10:
            refined_centers = _refine_with_nm(best_centers, timeout=10)
            refined_score = _score(refined_centers)
            if refined_score > best_score:
                best_score = refined_score
                best_centers = refined_centers.copy()

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
