"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Minimum distance to boundary (scaled by 2 for consistency with pair distances in soft-min?)
    # Score r = min( min_pair_dist/2, min_wall_dist )
    # min_wall_dist = 0.5 - max(|c_ik|)
    min_boundary = 0.5 - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    return min(min_boundary, min_pair)


def _soft_min_energy(centers, beta):
    """
    Compute the negative soft-min of pairwise distances and wall distances.
    Minimizing this energy maximizes the minimum distance (radius).
    """
    n = len(centers)
    
    # Wall distances: 2 * (0.5 - |c_ik|) is not quite right for the min definition.
    # r = min( min_pair/2, min_wall )
    # Let y_pair = d_ij and y_wall = 2 * (0.5 - |c_ik|)? No.
    # If we define y_wall = 2 * (0.5 - |c_ik|), then r = min( min_pair/2, min_wall/2 ) = 0.5 * min( min_pair, min_wall ).
    # So maximizing r is equivalent to maximizing min( min_pair, min_wall ).
    # Let's use x_pair = d_ij and x_wall = 2 * (0.5 - |c_ik|).
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    # We need sqrt for distances, but exp(-beta * dist) is hard to grad easily with sqrt?
    # Actually, we can just use distances.
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    
    # Wall distances
    # dist_to_wall = 0.5 - |c_ik|
    # We want to minimize -softmin(dists, 2*dist_to_wall)
    # Let's define the set of "critical distances" Z.
    # Z contains d_ij for i<j and 2*(0.5 - |c_ik|) for all i,k.
    
    # To avoid constructing a huge vector, we can sum the exponentials.
    # Sum_pairs = sum_{i<j} exp(-beta * d_ij)
    # Sum_walls = sum_{i,k} exp(-beta * 2 * (0.5 - |c_ik|))
    
    # Gradient calculation requires care.
    # Let's use a simpler potential that is smooth:
    # E = sum_{i<j} 1/d_ij^2 + sum_{i,k} 1/(0.5 - |c_ik|)^2
    # This is a standard repulsion potential. It effectively maximizes the minimum distance.
    
    # Let's implement the repulsion potential.
    # E = sum_{i<j} (d_ij)^-2 + sum_i sum_k (0.5 - |c_ik|)^-2
    
    # However, this blows up if d_ij -> 0 or wall -> 0.
    # We need to ensure spheres don't overlap or go outside.
    # The optimizer will try to increase these distances.
    
    # Let's use the soft-min approach with explicit gradients.
    # Energy = -1/beta * log( Sum_pairs + Sum_walls )
    # Grad = -1/beta * ( 1/(Sum) * ( Grad_Sum ) )
    
    # Sum_pairs = sum_{i<j} exp(-beta * d_ij)
    # d_ij = sqrt((ci-cj)^2)
    # d/dci exp(-beta * d_ij) = exp(-beta * d_ij) * (-beta) * (ci-cj)/d_ij
    
    # This is computationally intensive for N=100 (4950 pairs).
    # Vectorized NumPy should handle it.
    
    # Calculate pairwise distances and their gradients
    # diffs shape (N, N, 3)
    # dists shape (N, N)
    
    # To save memory, we can process in chunks or just use the full matrix since N=100 is small.
    
    exp_pairs = np.exp(-beta * dists)
    np.fill_diagonal(exp_pairs, 0.0)
    sum_pairs = np.sum(exp_pairs)
    
    # Wall part
    # wall_dist = 0.5 - abs(centers)
    # term = 2 * wall_dist
    # exp_walls = exp(-beta * 2 * wall_dist)
    wall_dist = 0.5 - np.abs(centers)
    # If wall_dist is negative, the sphere is outside. We want to penalize this heavily.
    # The soft-min will be dominated by the smallest value. If any is negative, the min is negative.
    # We want to maximize the min.
    # If we use exp(-beta * x), and x is negative, exp is large.
    # This works.
    
    term_walls = 2.0 * wall_dist
    exp_walls = np.exp(-beta * term_walls)
    sum_walls = np.sum(exp_walls)
    
    total_sum = sum_pairs + sum_walls
    
    if total_sum < 1e-100:
        return 0.0, np.zeros_like(centers)
    
    energy = -1.0 / beta * np.log(total_sum)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # dE/dci = -1/beta * 1/total_sum * d(sum_pairs)/dci
    # d(sum_pairs)/dci = sum_{j != i} d(exp(-beta * d_ij))/dci
    # = sum_{j != i} exp(-beta * d_ij) * (-beta) * (ci - cj) / d_ij
    
    # We have exp_pairs[i, j] = exp(-beta * d_ij)
    # We need (ci - cj) / d_ij
    # diffs[i, j] = ci - cj
    # dists[i, j] = d_ij
    
    # To avoid division by zero, we already set diagonal to inf/0.
    # dists has inf on diagonal, so exp is 0.
    # But (ci-cj)/d_ij is 0/inf = 0? No, it's undefined.
    # We should mask it.
    
    # grad_pairs = sum_j exp_pairs[i,j] * (-beta) * diffs[i,j] / dists[i,j]
    
    # Note: dists[i,j] is inf on diagonal, so diffs[i,j]/dists[i,j] is 0/inf = 0 in numpy?
    # Actually 0/inf is 0. But if dists is 0, it's 0/0 = nan.
    # We set diagonal dists to inf, so it should be fine.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diffs / dists[:, :, np.newaxis]
        # Set diagonal to 0
        unit_vecs[np.eye(n, dtype=bool), :] = 0.0
        
        # grad_pairs[i] = sum_j exp_pairs[i,j] * (-beta) * unit_vecs[i,j]
        # Multiply exp_pairs by -beta
        weighted_exp = exp_pairs * (-beta)
        
        # Sum over j (axis 1)
        grad_pairs = np.sum(weighted_exp[:, :, np.newaxis] * unit_vecs, axis=1)
        
    # Wall gradient
    # sum_walls = sum_{i,k} exp(-beta * 2 * (0.5 - |c_ik|))
    # d/dc_ik exp(-beta * 2 * (0.5 - |c_ik|))
    # = exp(...) * (-beta * 2) * d(0.5 - |c_ik|)/dc_ik
    # = exp(...) * (-2*beta) * (-sign(c_ik))
    # = exp(...) * (2*beta) * sign(c_ik)
    
    grad_walls = exp_walls * (2.0 * beta) * np.sign(centers)
    
    # Total gradient
    grad = grad_pairs + grad_walls
    
    # Scale by -1/beta * 1/total_sum
    grad *= (-1.0 / beta) / total_sum
    
    return energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin

    best_centers = None
    best_radius = -np.inf

    np.random.seed(42)

    # Try multiple random restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialize randomly within the cube, with some padding
        # Padding ensures we don't start too close to walls
        pad = 0.05
        centers = np.random.uniform(-0.5 + pad, 0.5 - pad, (N, 3))
        
        # Quick local optimization to resolve initial overlaps
        # Use a high beta to push spheres apart
        current_beta = 50.0
        
        # Optimize
        try:
            x0 = centers.flatten()
            
            def objective(x):
                c = x.reshape(N, 3)
                e, _ = _soft_min_energy(c, current_beta)
                return e
                
            def gradient(x):
                c = x.reshape(N, 3)
                _, g = _soft_min_energy(c, current_beta)
                return g.flatten()
            
            # Bounds
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            # Run optimization with increasing beta
            for beta_step in [10.0, 50.0, 200.0, 1000.0]:
                if time.time() > deadline - 5:
                    break
                    
                current_beta = beta_step
                
                # Re-define closures to capture current_beta
                def obj_curr(x, b=current_beta):
                    c = x.reshape(N, 3)
                    e, _ = _soft_min_energy(c, b)
                    return e
                    
                def grad_curr(x, b=current_beta):
                    c = x.reshape(N, 3)
                    _, g = _soft_min_energy(c, b)
                    return g.flatten()
                
                result = minimize(obj_curr, x0, jac=grad_curr, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                x0 = result.x
                centers = result.x.reshape(N, 3)
                
        except Exception as e:
            continue

        # Evaluate final radius
        current_radius = _compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()

    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        best_radius = _compute_radius(best_centers)

    return best_centers
# EVOLVE-BLOCK-END
