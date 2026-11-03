"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Minimum distance to boundary
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


def _energy_and_gradient(centers, soft_beta=20.0):
    """
    Compute a smooth surrogate energy and its gradient.
    Energy encourages large minimum radius by penalizing small pairwise distances
    and boundary violations using a soft-min (log-sum-exp) formulation.
    We minimize -r where r is approximated by a smooth function of distances.
    """
    n = len(centers)
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)  # shape (n, 3)
    
    # We want to maximize the minimum of all pairwise distances and boundary distances.
    # Let d_min be the minimum distance. We use a soft-min approximation.
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize soft_min, so we minimize -soft_min.
    # Let's define the set of all "distances" to consider:
    # All pairwise distances (n*(n-1)/2 of them) and all boundary distances (3*n of them).
    
    # For efficiency, let's work with the log-sum-exp of the negative distances.
    # Z = sum(exp(-beta * d_i))
    # soft_min = -1/beta * log(Z)
    # We want to minimize E = -soft_min = 1/beta * log(Z)
    
    # To avoid overflow/underflow, we can use the log-sum-exp trick.
    # But since we are minimizing E, and E is positive, we can just compute it.
    
    # Pairwise part
    # dists is (n, n) with inf on diagonal.
    # We only need upper triangle for unique pairs, but exp(-beta * inf) = 0, so we can use all.
    # However, 100x100 matrix is 10000 elements, which is fine.
    # Let's flatten the upper triangle to save a bit of computation? No, 10k is small.
    
    # Let's compute log-sum-exp for pairwise distances
    # To handle inf, we can mask them.
    mask_pair = np.isfinite(dists)
    # We also need to exclude the diagonal which is inf, but mask handles it.
    # Actually, dists[i,i] is inf, so exp(-beta*inf)=0.
    
    # For numerical stability, let's find the min distance to shift.
    # d_min_pair = min(dists[mask_pair])
    # But we want a smooth function.
    
    # Let's just compute:
    # Z_pair = sum(exp(-beta * dists))
    # Z_boundary = sum(exp(-beta * boundary_dists))
    # Z_total = Z_pair + Z_boundary
    # E = 1/beta * log(Z_total)
    
    # Gradient of E w.r.t. centers:
    # dE/dc_i = 1/beta * 1/Z_total * dZ_total/dc_i
    # dZ_pair/dc_i = sum_j exp(-beta * d_ij) * (-beta) * d(d_ij)/dc_i
    # d(d_ij)/dc_i = (c_i - c_j) / d_ij
    
    # This is getting complex. Let's try a simpler energy:
    # E = sum(1/d_ij^2) + sum(1/boundary_dist_k^2)
    # This is smooth and encourages separation.
    # Gradient: dE/dc_i = -2/d_ij^3 * (c_i - c_j) for each j != i
    #             -2/boundary_dist_k^3 * sign(c_ik) for each boundary k
    
    # Let's use this simpler energy. It's not a direct soft-min, but it works well for packing.
    
    # Pairwise energy
    # dists_sq is (n, n)
    # We want sum_{i!=j} 1/dists_sq[i,j]
    # But 1/0 is inf. We need to handle close points.
    # Let's add a small epsilon to avoid division by zero.
    eps = 1e-8
    dists_sq_safe = np.maximum(dists_sq, eps)
    # Zero out diagonal
    np.fill_diagonal(dists_sq_safe, np.inf)
    
    energy_pair = np.sum(1.0 / dists_sq_safe)
    
    # Boundary energy
    boundary_dists_safe = np.maximum(boundary_dists, eps)
    energy_boundary = np.sum(1.0 / boundary_dists_safe ** 2)
    
    energy = energy_pair + energy_boundary
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # dE/dc_i = -2 * sum_{j!=i} (c_i - c_j) / dists_sq[i,j]^2
    # Let's compute this efficiently.
    # dists_sq is (n, n)
    # We need to compute for each i, the sum over j of (c_i - c_j) / dists_sq[i,j]^2
    # Let's create a matrix W where W[i,j] = 1 / dists_sq[i,j]^2
    W = 1.0 / dists_sq_safe ** 2
    np.fill_diagonal(W, 0.0)
    
    # grad_pair[i] = -2 * sum_j W[i,j] * (c_i - c_j)
    #                = -2 * (sum_j W[i,j] * c_i - sum_j W[i,j] * c_j)
    #                = -2 * (c_i * sum_j W[i,j] - sum_j W[i,j] * c_j)
    
    sum_W = np.sum(W, axis=1)  # (n,)
    W_c = W @ centers  # (n, 3)
    
    grad_pair = -2.0 * (centers * sum_W[:, np.newaxis] - W_c)
    
    # Boundary gradient
    # dE/dc_ik = -2 * sign(c_ik) / boundary_dist_k^3
    # boundary_dists[i,k] = 0.5 - |c_ik|
    # d(boundary_dist)/dc_ik = -sign(c_ik)
    # d(1/boundary_dist^2)/dc_ik = -2/boundary_dist^3 * (-sign(c_ik)) = 2*sign(c_ik)/boundary_dist^3
    # Wait, energy_boundary = sum(1/boundary_dist^2)
    # dE/dc_ik = -2/boundary_dist^3 * d(boundary_dist)/dc_ik
    # d(boundary_dist)/dc_ik = -sign(c_ik)
    # So dE/dc_ik = -2/boundary_dist^3 * (-sign(c_ik)) = 2*sign(c_ik)/boundary_dist^3
    
    # Let's verify: if c_ik > 0, boundary_dist = 0.5 - c_ik.
    # dE/dc_ik = d/dc_ik (1/(0.5-c_ik)^2) = -2/(0.5-c_ik)^3 * (-1) = 2/(0.5-c_ik)^3.
    # sign(c_ik) = 1. So 2*1/boundary_dist^3. Correct.
    
    signs = np.sign(centers)
    # Handle zero: sign(0) = 0, but derivative is undefined. We can use 0 or arbitrary.
    # Let's use 0 for simplicity, or np.where(centers == 0, 1, np.sign(centers))
    # Actually, if c_ik = 0, boundary_dist = 0.5. The derivative of |x| at 0 is undefined,
    # but the subgradient is [-1, 1]. Let's pick 0.
    
    grad_boundary = 2.0 * signs / boundary_dists_safe ** 3
    
    grad = grad_pair + grad_boundary
    
    return energy, grad


def _optimize_for_radius(centers, time_budget=10.0):
    """
    Use L-BFGS-B to minimize the energy function.
    """
    start_time = time.time()
    
    # Define the objective function for scipy
    def obj(x):
        c = x.reshape(-1, 3)
        # Clamp to cube
        c = np.clip(c, -0.5, 0.5)
        e, g = _energy_and_gradient(c)
        # The gradient needs to be flattened
        return e, g.flatten()
    
    # Initial point
    x0 = centers.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use L-BFGS-B
    # maxiter and ftol can be adjusted based on time
    # We'll run until time budget is exhausted
    # But scipy doesn't support time-based stopping directly.
    # We'll set a high maxiter and hope it converges or we break early.
    # Alternatively, we can use a callback to check time.
    
    def callback(xk):
        if time.time() - start_time > time_budget - 1.0:
            return True  # Stop
        return False
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=True, bounds=bounds, 
                       options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8,
                                'callback': callback, 'disp': False})
        new_centers = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
    except Exception as e:
        # If optimization fails, return original
        new_centers = centers.copy()
    
    return new_centers


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin

    np.random.seed(42)

    # Start with a grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()

    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)

    # Multiple restarts with random perturbations + L-BFGS-B optimization
    n_restarts = 0
    while time.time() < deadline - 15.0:  # Leave 15s for final polish
        # Generate a new random starting point
        # Mix of grid and random
        if n_restarts % 3 == 0:
            # Perturbed grid
            c_init = best_centers.copy()
            c_init += np.random.normal(0, 0.05, c_init.shape)
        elif n_restarts % 3 == 1:
            # Random points
            c_init = np.random.uniform(-0.45, 0.45, (N, 3))
        else:
            # Clustered random
            c_init = np.random.uniform(-0.45, 0.45, (N, 3))
            # Add some structure
            for i in range(N):
                if np.random.random() < 0.5:
                    c_init[i] = (np.random.uniform(-0.45, 0.45), 
                                 np.random.uniform(-0.45, 0.45), 
                                 np.random.uniform(-0.45, 0.45))
        
        c_init = np.clip(c_init, -0.5, 0.5)
        
        # Optimize
        time_for_opt = min(10.0, deadline - 15.0 - time.time())
        if time_for_opt < 1.0:
            break
            
        c_opt = _optimize_for_radius(c_init, time_budget=time_for_opt)
        
        r_opt = _compute_radius(c_opt)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
        
        n_restarts += 1

    # Final polish: try small local moves around best solution
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        polish_centers = best_centers.copy()
        polish_radius = best_radius
        T_polish = 0.005
        n_polish = int(min(5000, remaining_time * 100))
        for _ in range(n_polish):
            if time.time() > deadline - 1:
                break
            idx = np.random.randint(N)
            delta = np.random.normal(0, T_polish, 3)
            new_centers = polish_centers.copy()
            new_centers[idx] += delta
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_radius = _compute_radius(new_centers)
            if new_radius > polish_radius:
                polish_centers = new_centers
                polish_radius = new_radius
                T_polish *= 0.99
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers.copy()

    return best_centers
# EVOLVE-BLOCK-END
