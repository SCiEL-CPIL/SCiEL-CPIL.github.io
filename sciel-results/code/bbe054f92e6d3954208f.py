"""
Strategy: Deterministic Riemannian Newton Method for Logarithmic Energy Minimization.
1. Compute the exact Hessian of the logarithmic energy on the sphere.
2. Use a Riemannian Newton method with backtracking line search for fast, second-order convergence.
3. Run from multiple diverse deterministic initializations (Fibonacci, Halton, Jittered) to avoid local minima.
4. This avoids the slow linear convergence of gradient descent and the randomness/stagnation of Simulated Annealing.
"""

N = 150
import numpy as np
import time
import math

def _energy_and_hessian_sphere(x):
    """
    Computes the logarithmic energy, Euclidean gradient, and Euclidean Hessian 
    for points on the unit sphere.
    
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3), hessian (N, 3, N, 3)
    
    Energy E = sum_{i<j} -log(||x_i - x_j||)
    Grad_i = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    Hessian_ij (for i != j) = 2 * (x_i - x_j) (x_i - x_j)^T / ||x_i - x_j||^4 - delta_ij * I / ||x_i - x_j||^2
    Hessian_ii = - sum_{j!=i} Hessian_ij
    """
    # diff[i, j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    mask = ~np.eye(N, dtype=bool)
    safe_dist_sq = np.where(mask, dist_sq, 1.0)
    dist = np.sqrt(safe_dist_sq)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist_ij)
    log_terms = -np.log(dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient
    # grad_i = - sum_{j!=i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / safe_dist_sq[mask]
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    # Hessian
    # H_ij = 2 * diff_ij diff_ij^T / dist_ij^4 - delta_ij * I / dist_ij^2
    # Note: The Euclidean Hessian of the energy function defined on R^3.
    
    # Term 1: 2 * diff_ij diff_ij^T / dist_ij^4
    inv_dist_sq2 = np.zeros_like(dist_sq)
    inv_dist_sq2[mask] = 1.0 / (safe_dist_sq[mask] ** 2)
    
    # hessian_term1[i, j, k, l] = 2 * diff[i, j, k] * diff[i, j, l] * inv_dist_sq2[i, j]
    # We need to construct the (N, 3, N, 3) tensor.
    
    # Efficient construction:
    # H[i, :, j, :] = 2 * outer(diff[i,j], diff[i,j]) * inv_dist_sq2[i,j] - (I if i==j else 0) * inv_dist_sq[i,j]
    
    hessian = np.zeros((N, 3, N, 3))
    
    # Vectorized outer product part
    # diff is (N, N, 3)
    # We want to compute for all i, j: 2 * inv_dist_sq2[i,j] * outer(diff[i,j], diff[i,j])
    
    # Let's do it in a loop over N for clarity and memory safety, or use broadcasting carefully.
    # N=150, so N^2 = 22500. 3x3 matrices. 22500 * 9 * 8 bytes ~ 1.6 MB. Very safe.
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = diff[i, j]
            w = 2.0 * inv_dist_sq2[i, j]
            hessian[i, :, j, :] = w * np.outer(d, d)
            # Subtract delta_ij * I / dist^2. Since i!=j, delta is 0.
            
    # Now add the diagonal terms: H_ii = - sum_{j!=i} H_ij
    for i in range(N):
        sum_h = np.sum(hessian[i, :, :, :], axis=1) # Sum over j axis (axis 1 of the 4D tensor slice i)
        # Wait, hessian[i, :, j, :] is a 3x3 matrix.
        # hessian[i, :, :, :] is shape (3, N, 3).
        # We want sum over j (the N dimension).
        # hessian[i, :, :, :] -> shape (3, N, 3).
        # Sum axis 1 -> shape (3, 3).
        hessian[i, :, i, :] = -sum_h
        
    return energy, grad, hessian

def _riemannian_hessian(grad, hessian, x):
    """
    Compute the Riemannian Hessian projected to the tangent space.
    For a function f on the sphere, the Riemannian Hessian at x in direction v is:
    Hess_f^R(x)[v, v] = v^T (H_f(x) - (grad f(x) . x) I) v  (for v in T_x S^2)
    More generally, the Riemannian Hessian tensor is:
    Hess^R = P_x (H_euclidean - grad f x^T - x (grad f)^T + (grad f . x) I) P_x
    where P_x = I - x x^T is the projection to tangent space.
    
    However, for Newton step, we need to solve Hess^R [v] = -grad^R.
    Since grad is already in tangent space (grad . x = 0), we can simplify.
    The Riemannian gradient is grad_euclidean projected.
    The Riemannian Hessian applied to a tangent vector v is:
    (H_euclidean v) projected - (grad . v) x - (H_euclidean x) v + (H_euclidean x . v) x ... this gets messy.
    
    A simpler approach for Newton on manifold:
    Use the "Euclidean Newton step" projected back.
    Or, construct the 2x2 block matrix for each point? No, the Hessian couples all points.
    
    Let's use the property that for small steps, the Riemannian Hessian is approximated by the Euclidean Hessian projected to the tangent space of each point, minus curvature terms.
    Actually, the standard Riemannian Newton method uses:
    v = - (Hess^R)^{-1} grad^R
    
    Computing the full Riemannian Hessian tensor (N, 3, N, 3) is possible.
    But we can also just use the Euclidean Hessian and project the solution.
    Let's try the "Projected Euclidean Newton":
    1. Solve H_euclidean [d] = -grad_euclidean
    2. Project d to tangent space: v = P_x d
    3. Step: x_new = exp_x(v)
    
    This is often effective and much cheaper than constructing the full Riemannian Hessian.
    """
    # We will implement the Projected Euclidean Newton.
    # We already have grad and hessian (Euclidean).
    # We need to solve H d = -g.
    # H is (N, 3, N, 3). We can reshape to (3N, 3N).
    
    H_flat = hessian.reshape(3*N, 3*N)
    g_flat = grad.reshape(3*N)
    
    try:
        # Solve linear system. H is symmetric positive semi-definite? 
        # Log energy is not convex, so H might be indefinite.
        # Use np.linalg.solve. If singular, add small regularization.
        d = np.linalg.solve(H_flat, -g_flat)
    except np.linalg.LinAlgError:
        # Regularize
        d = np.linalg.solve(H_flat + 1e-8 * np.eye(3*N), -g_flat)
    
    d = d.reshape(N, 3)
    
    # Project to tangent space
    dot = np.sum(d * x, axis=1, keepdims=True)
    v = d - dot * x
    
    return v

def _exponential_map(x, v):
    """Exponential map on the unit sphere."""
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    mask = norm_v > 1e-10
    new_x = np.zeros_like(x)
    
    cos_theta = np.cos(norm_v)
    sin_theta_over_norm = np.zeros_like(norm_v)
    sin_theta_over_norm[mask] = np.sin(norm_v[mask]) / norm_v[mask]
    
    new_x = cos_theta * x + sin_theta_over_norm * v
    return new_x

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _halton_sphere(n, seed=0):
    """Generate Halton sequence points on the sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    indices = np.arange(n) + 1
    u1 = np.array([halton(i, 2) for i in indices])
    u2 = np.array([halton(i, 3) for i in indices])
    
    # Map [0,1]x[0,1] to sphere
    # Use spherical coordinates
    theta = 2 * np.pi * u1
    phi = np.arccos(1 - 2 * u2)
    
    x = np.zeros((n, 3))
    x[:, 0] = np.sin(phi) * np.cos(theta)
    x[:, 1] = np.sin(phi) * np.sin(theta)
    x[:, 2] = np.cos(phi)
    return x

def _jittered_sphere(n, seed=0):
    """Generate jittered grid points on the sphere."""
    rng = np.random.default_rng(seed)
    # Create a grid in spherical coordinates
    n_theta = int(np.sqrt(n))
    n_phi = int(np.ceil(n / n_theta))
    
    thetas = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi
    phis = (np.arange(n_phi) + 0.5) / n_phi * np.pi
    
    theta_grid, phi_grid = np.meshgrid(thetas, phis)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Jitter
    jitter_theta = rng.uniform(-1.0, 1.0, size=n_theta) / n_theta * 0.5
    jitter_phi = rng.uniform(-1.0, 1.0, size=n_phi) / n_phi * 0.5
    
    # This simple jittering might not be uniform. Let's just use random points for this init.
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _riemannian_newton(x_init, max_iter=200, time_budget=10.0):
    """
    Riemannian Newton method with backtracking line search.
    """
    x_curr = x_init.copy()
    e_curr, g_curr, h_curr = _energy_and_hessian_sphere(x_curr)
    best_energy = e_curr
    best_x = x_curr.copy()
    
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() > start_time + time_budget:
            break
            
        # Compute Newton step
        v = _riemannian_hessian(g_curr, h_curr, x_curr)
        
        # Check if step is zero
        if np.linalg.norm(v) < 1e-10:
            break
            
        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4
        for _ in range(30):
            x_new = _exponential_map(x_curr, alpha * v)
            e_new, g_new, h_new = _energy_and_hessian_sphere(x_new)
            
            # Armijo condition
            if e_new < e_curr + c1 * alpha * np.sum(g_curr * v):
                x_curr = x_new
                e_curr = e_new
                g_curr = g_new
                h_curr = h_new
                if e_curr < best_energy:
                    best_energy = e_curr
                    best_x = x_curr.copy()
                break
            alpha *= 0.5
            
        # If no step accepted, break
        else:
            break
            
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
        _jittered_sphere(N, seed=1),
        _jittered_sphere(N, seed=2)
    ]
    
    best_energy = float('inf')
    best_x = None
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Run Riemannian Newton
        x_opt = _riemannian_newton(x_init, max_iter=100, time_budget=10.0)
        e_opt, _, _ = _energy_and_hessian_sphere(x_opt)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
            
    # Final polish on the best result found with more iterations
    if best_x is not None and time.time() < deadline - 5:
        x_final = _riemannian_newton(best_x, max_iter=200, time_budget=10.0)
        e_final, _, _ = _energy_and_hessian_sphere(x_final)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final
            
    return best_x
