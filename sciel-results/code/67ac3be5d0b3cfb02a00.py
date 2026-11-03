"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _generate_bcc_initial():
    """
    Generate an initial configuration based on a Body-Centered Cubic (BCC) lattice.
    This is denser than simple cubic and provides a better isotropic start.
    We scale the lattice to fit within the cube [-0.5, 0.5]^3.
    """
    # BCC lattice points: corners + center of the cube.
    # In a unit cell of side 1, points are at (0,0,0) and (0.5, 0.5, 0.5).
    # For a grid of size LxLxL, we have L^3 cells.
    # Total points = 2 * L^3.
    # We need ~50 points.
    # If L=3, points = 2*27 = 54. This is close to 50.
    # If L=2, points = 2*8 = 16. Too few.
    
    L = 3
    points = []
    
    # Generate BCC points in [0, 1]^3
    for i in range(L):
        for j in range(L):
            for k in range(L):
                # Corner point
                points.append((i, j, k))
                # Center point
                points.append((i + 0.5, j + 0.5, k + 0.5))
    
    points = np.array(points)
    
    # We have 54 points. We need 50.
    # We can randomly remove 4, or just take the first 50.
    # Taking the first 50 might break symmetry. 
    # Let's try to keep it somewhat symmetric or just pick a subset.
    # Actually, for optimization, the starting point just needs to be valid and diverse.
    # Let's scale these points to fit in [-0.5, 0.5].
    # Current range is [0, L-1] for corners, [0.5, L-0.5] for centers.
    # Max coordinate is L-1 = 2. Min is 0.
    # Let's map [0, 2] to [-0.45, 0.45] to stay safely inside.
    
    # Normalize to [0, 1]
    points = points / (L - 1)
    
    # Map to [-0.5, 0.5] with a small margin
    margin = 0.05
    points = points * (1.0 - 2*margin) - 0.5 + margin
    
    # Shuffle to avoid bias in which points are kept if we truncate
    np.random.seed(42)
    idx = np.random.permutation(len(points))
    points = points[idx]
    
    # Take first N
    return points[:N].copy()

def _objective_and_grad(x, N, p=8.0):
    """
    Objective function: minimize the smooth minimum of pairwise distances.
    We use the power-sum trick: min(d_i) ≈ (sum(d_i^(-p)))^(-1/p) for large p.
    Actually, to minimize the radius, we want to maximize the min distance.
    So we minimize -min_distance.
    Using the power-sum for the minimum: 
    min(d) ≈ (sum(d^{-p}))^{-1/p}
    We want to minimize this value? No, we want to maximize it.
    So we minimize - (sum(d^{-p}))^{-1/p}.
    
    Alternatively, we can minimize the "energy" which is high when spheres are close.
    E = sum(1/d^2) is a common repulsive potential. Minimizing E pushes spheres apart.
    However, minimizing sum(1/d^2) doesn't strictly guarantee maximizing the minimum distance,
    but it's a good proxy.
    
    A better proxy for "maximize min distance" is to minimize the maximum overlap.
    But that's non-smooth.
    
    Let's stick to minimizing the negative of the power-sum minimum.
    f(x) = - (sum_{i<j} d_ij^{-p} + sum_i wall_dist_i^{-p})^{-1/p}
    
    Wait, if we use p=8, it's a smooth approximation of the minimum.
    Let's compute the gradient numerically or analytically?
    Analytical is better for L-BFGS-B speed.
    
    Let d_ij be distance between i and j.
    Let w_i be distance to wall for i.
    Let S = sum_{i<j} d_ij^{-p} + sum_i w_i^{-p}
    We want to minimize -S^{-1/p} = -S^{-(1/p)}
    
    d/dx (-S^{-1/p}) = -(-1/p) S^{-1/p - 1} * dS/dx = (1/p) S^{-1/p - 1} * dS/dx
    
    dS/dx_k = sum_{j!=k} d(k,j)^{-p} * d/dx_k (d(k,j)^{-p}) + d/dx_k (w_k^{-p})
    
    d/dx_k (d(k,j)^{-p}) = -p * d(k,j)^{-p-1} * d/dx_k (d(k,j))
    d/dx_k (d(k,j)) = (x_k - x_j) / d(k,j)  (vector)
    
    So term = -p * d(k,j)^{-p-1} * (x_k - x_j) / d(k,j) = -p * d(k,j)^{-p-2} * (x_k - x_j)
    
    Similarly for walls.
    Wall distance for sphere k:
    w_k = min(0.5 - x_k, x_k + 0.5, 0.5 - y_k, y_k + 0.5, 0.5 - z_k, z_k + 0.5)
    This is non-differentiable at the center. But we are likely not at the exact center.
    We can approximate the gradient of the minimum wall distance.
    Actually, it's safer to just use the sum of all wall distances? No, that's not the minimum.
    
    Let's simplify:
    Use a repulsive potential: E = sum_{i<j} 1/d_ij^2 + sum_i 1/w_i^2
    Minimizing this encourages all distances to be large.
    It is smooth everywhere (except d=0 or w=0).
    Gradient:
    dE/dx_k = sum_{j!=k} -2 * d(k,j)^{-3} * (x_k - x_j) / d(k,j) ... wait
    d/dx (1/d^2) = -2/d^3 * (x_i - x_j)/d = -2 (x_i - x_j) / d^4
    
    For walls:
    w_k = min(0.5 - x_k, x_k + 0.5, ...)
    If we are near the boundary, the derivative is clear.
    Let's use a soft wall potential: 1/(0.5 - x_k)^2 + 1/(x_k + 0.5)^2 + ...
    This penalizes being close to ANY wall, effectively keeping it centered or pushed to the nearest wall if forced.
    But we want to maximize the distance to the NEAREST wall.
    The sum of inverses of squares of distances to ALL walls is a good proxy for "distance to nearest wall" in the sense that the term with the smallest distance dominates.
    
    So Objective:
    E(x) = sum_{i<j} 1/d_ij^4 + sum_i sum_{axis} 1/d_{i,axis}^4
    (Using power 4 for stronger repulsion at close range)
    
    Gradient:
    dE/dx_k = sum_{j!=k} -4 (x_k - x_j) / d_ij^6  (for each component)
    dE/dx_k (wall term) = sum_{axis} 4 (x_k - wall) / dist^6  (sign depends on which wall)
    
    Let's implement this.
    """
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero
    
    # E_pair = sum 1/d^4 = sum 1/dists_sq^2
    E_pair = np.sum(1.0 / (dists_sq ** 2))
    
    # Gradient of E_pair
    # d/dx_k (1/d^4) = -4 (x_k - x_j) / d^6
    # Note: dists_sq = d^2. So 1/d^6 = 1/dists_sq^3.
    # grad_pair = -4 * sum_j (x_k - x_j) / dists_sq^3
    
    # We need to compute this efficiently.
    # Let A_kj = (x_k - x_j) / dists_sq^3
    # grad_k = -4 * sum_j A_kj
    
    # dists_sq is (N, N).
    # We want to sum over j.
    # (x_k - x_j) is (N, N, 3).
    # dists_sq^3 is (N, N).
    
    # Avoid division by zero for diagonal (though we set it to 1)
    inv_d6 = 1.0 / (dists_sq ** 3)
    np.fill_diagonal(inv_d6, 0.0)
    
    # grad_pair = -4 * sum_j (x_k - x_j) * inv_d6[k,j]
    # (x_k - x_j) * inv_d6[:, :, np.newaxis]
    weighted_diff = diff * inv_d6[:, :, np.newaxis]
    grad_pair = -4.0 * np.sum(weighted_diff, axis=1)
    
    # Wall distances
    # For each sphere i and axis a:
    # dist to -0.5: x_i_a + 0.5
    # dist to 0.5: 0.5 - x_i_a
    
    # E_wall = sum_i sum_a 1/(x_i_a + 0.5)^4 + 1/(0.5 - x_i_a)^4
    
    # Let's compute this and its gradient.
    E_wall = 0.0
    grad_wall = np.zeros_like(x)
    
    for i in range(N):
        for a in range(3):
            d1 = x[i, a] + 0.5
            d2 = 0.5 - x[i, a]
            
            # Add to energy
            E_wall += 1.0 / (d1 ** 4) + 1.0 / (d2 ** 4)
            
            # Gradient
            # d/dx (1/d1^4) = -4/d1^5 * 1
            # d/dx (1/d2^4) = -4/d2^5 * (-1) = 4/d2^5
            
            grad_wall[i, a] += -4.0 / (d1 ** 5) + 4.0 / (d2 ** 5)
    
    total_E = E_pair + E_wall
    total_grad = grad_pair + grad_wall
    
    return total_E, total_grad

def _compute_radius(centers):
    """Compute the common radius for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Generate initial BCC configuration
    centers = _generate_bcc_initial()
    
    # 2. Optimize using L-BFGS-B
    # We minimize the repulsive energy E.
    # This pushes spheres apart.
    
    # Flatten centers for scipy
    x0 = centers.flatten()
    
    # Bounds: keep inside cube with small margin to avoid singularities
    # Actually, if we hit the boundary, the gradient becomes huge, which is good.
    # But we must stay within [-0.5, 0.5].
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def fun(x):
        val, grad = _objective_and_grad(x, N)
        return val
    
    def fprime(x):
        val, grad = _objective_and_grad(x, N)
        return grad
    
    # Use L-BFGS-B
    # maxiter: limit iterations to stay within time
    # maxfun: limit function evaluations
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    try:
        res = minimize(
            fun, 
            x0, 
            method='L-BFGS-B', 
            jac=fprime,
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < 1e10: # Check for valid result
            new_centers = res.x.reshape(N, 3)
            
            # Clamp to ensure validity (L-BFGS-B should respect bounds, but just in case)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
                
    except Exception as e:
        pass # If optimization fails, return initial
    
    # 3. Local refinement (optional, if time permits)
    # Simple random perturbations to escape any remaining local minima or boundary issues
    if time.time() - start_time < deadline - 5:
        step_size = 0.01
        np.random.seed(43)
        for _ in range(100):
            if time.time() - start_time > deadline - 2:
                break
            delta = np.random.randn(N, 3) * step_size
            new_centers = best_centers + delta
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
            else:
                step_size *= 0.9

    return best_centers
# EVOLVE-BLOCK-END
