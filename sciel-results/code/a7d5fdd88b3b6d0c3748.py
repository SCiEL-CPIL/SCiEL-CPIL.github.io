"""Combined strategy: Diverse Initialization + Riemannian Conjugate Gradient + L-BFGS Polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _energy_and_grad(X):
    """
    Compute logarithmic energy and Riemannian gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Riemannian Gradient w.r.t. x_i = (I - x_i x_i^T) * (sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2)
    """
    n = X.shape[0]
    # Compute all pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    
    # Squared distances
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # Avoid division by zero
    
    # Distances
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    # Clip distances to avoid log(0)
    d_iu = dist[iu]
    d_iu = np.maximum(d_iu, 1e-15)
    energy = -np.sum(np.log(d_iu))
    
    # Euclidean Gradient
    # grad_eucl[i] = sum_{j!=i} (x_j - x_i) / dist[i,j]^2
    # diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # We need sum_j (-diff[i,j]) / dist2[i,j]
    # Note: dist2[i,i] is 1, but we should ignore diagonal in sum.
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # Weighted sum: sum_j inv_dist2[i,j] * (x_j - x_i)
    # = sum_j inv_dist2[i,j] * (-diff[i,j])
    # = - sum_j inv_dist2[i,j] * diff[i,j]
    # diff is (n, n, 3). inv_dist2 is (n, n).
    # We want result (n, 3).
    # term_ij = inv_dist2[i,j] * diff[i,j]
    # sum_j term_ij
    eucl_grad = -np.sum(inv_dist2[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space
    # grad_riem[i] = eucl_grad[i] - (eucl_grad[i] . X[i]) X[i]
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _exp_map(X, V):
    """
    Exponential map on the sphere.
    X: (N, 3) base points
    V: (N, 3) tangent vectors
    Returns: (N, 3) points on sphere
    """
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    # Handle zero vectors
    mask = norms > 1e-15
    X_new = X.copy()
    if np.any(mask):
        cos_theta = np.cos(norms[mask])
        sin_theta = np.sin(norms[mask])
        V_hat = V[mask] / norms[mask]
        X_new[mask] = X[mask] * cos_theta + V_hat * sin_theta
    # Ensure they are on sphere (numerical stability)
    return _project_to_sphere(X_new)

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_init(n, num_dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        points[i - 1] = [halton(i, 2), halton(i, 3), halton(i, 5)]
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    # Add some jitter to break symmetry
    rng = np.random.RandomState(42)
    points += rng.normal(0, 0.01, size=points.shape)
    return _project_to_sphere(points)

def _farthest_point_init(n, rng_seed=42):
    """Farthest-point sampling initialization for better diversity."""
    rng = np.random.RandomState(rng_seed)
    points = np.zeros((n, 3))
    
    # Start with a random point
    points[0] = rng.randn(3)
    points[0] /= np.linalg.norm(points[0])
    
    for i in range(1, n):
        best_idx = None
        best_min_dist = -1
        
        # Try a few random candidates
        for _ in range(50):
            candidate = rng.randn(3)
            candidate /= np.linalg.norm(candidate)
            
            # Compute minimum distance to existing points
            dists = np.linalg.norm(points[:i] - candidate, axis=1)
            min_dist = np.min(dists)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = candidate
        
        points[i] = best_idx
    
    return points

def _riemannian_cg(X0, max_iter=500, tol=1e-10, time_limit=10.0):
    """
    Riemannian Conjugate Gradient with line search on the sphere.
    """
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    # Initial direction: negative gradient
    d = -grad.copy()
    
    # Store previous gradient for beta calculation
    g_prev = grad.copy()
    g_prev_norm_sq = np.sum(g_prev * g_prev)
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Line search
        # We want to minimize f(exp_x(alpha * d))
        # Start with alpha = 1.0
        alpha = 1.0
        best_alpha = 0.0
        best_energy = energy
        
        for ls_iter in range(30):
            # Compute new point
            X_new = _exp_map(X, alpha * d)
            
            # Compute new energy
            e_new, _ = _energy_and_grad(X_new)
            
            if e_new < best_energy:
                best_energy = e_new
                best_alpha = alpha
            
            # Armijo condition check
            if e_new <= energy - 1e-4 * alpha * grad_norm**2:
                break
                
            alpha *= 0.5
            
            if alpha < 1e-10:
                break
        
        if best_alpha == 0.0:
            # If no improvement, break
            break
            
        # Update X
        X = _exp_map(X, best_alpha * d)
        
        # Compute new gradient
        energy, grad = _energy_and_grad(X)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Polak-Ribiere beta
        # beta = (g_new^T (g_new - g_prev)) / ||g_prev||^2
        # Note: gradients are in tangent space
        dot_diff = np.sum(grad * (grad - g_prev))
        beta = dot_diff / g_prev_norm_sq if g_prev_norm_sq > 1e-30 else 0.0
        beta = max(0.0, beta)
        
        # Update direction
        d = -grad + beta * d
        
        # Project d to tangent space
        dot_prod = np.sum(d * X, axis=1, keepdims=True)
        d -= dot_prod * X
        
        # Update previous gradient
        g_prev = grad
        g_prev_norm_sq = np.sum(g_prev * g_prev)
        
        # Restart if beta is large or direction is not descent
        if np.sum(d * grad) > 0:
            d = -grad.copy()
            # Re-project
            dot_prod = np.sum(d * X, axis=1, keepdims=True)
            d -= dot_prod * X
            
    return X, energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (limit is 150s)
    
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_init(N))
    
    # 2. Halton sequence
    initializations.append(_halton_init(N))
    
    # 3. Farthest-point sampling
    initializations.append(_farthest_point_init(N, rng_seed=42))
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(123)
    rand_pts = rng.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts))
    
    # 5. Another random seed
    rng2 = np.random.RandomState(456)
    rand_pts2 = rng2.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts2))
    
    # 6. Fibonacci with perturbation
    fib_perturbed = _fibonacci_init(N).copy()
    rng4 = np.random.RandomState(321)
    fib_perturbed += rng4.normal(0, 0.05, size=fib_perturbed.shape)
    initializations.append(_project_to_sphere(fib_perturbed))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: Riemannian CG from multiple starts
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 40.0:
            break
            
        # Time budget per run
        remaining_runs = len(initializations) - init_idx
        time_budget = (deadline - 40.0 - time.time()) / max(1, remaining_runs)
        time_budget = max(time_budget, 5.0)
        
        try:
            # Run RCG
            X_opt, e_opt = _riemannian_cg(X0, max_iter=500, tol=1e-8, time_limit=time_budget)
            
            # Polish with L-BFGS-B
            if time.time() < deadline - 25.0:
                def objective(x_flat):
                    X = x_flat.reshape(N, 3)
                    X = _project_to_sphere(X)
                    e, _ = _energy_and_grad(X)
                    return e
                
                def gradient(x_flat):
                    X = x_flat.reshape(N, 3)
                    X = _project_to_sphere(X)
                    e, g = _energy_and_grad(X)
                    return g.flatten()
                    
                result = minimize(
                    objective,
                    X_opt.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'm': 15}
                )
                X_polished = _project_to_sphere(result.x.reshape(N, 3))
                e_polished, _ = _energy_and_grad(X_polished)
                
                if e_polished < e_opt:
                    X_opt = X_polished
                    e_opt = e_polished
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt.copy()
                
        except Exception:
            pass

    # Phase 2: Final high-precision polish on the best result
    if time.time() < deadline - 10.0:
        try:
            X0 = best_points.copy()
            
            def objective(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, _ = _energy_and_grad(X)
                return e
            
            def gradient(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, g = _energy_and_grad(X)
                return g.flatten()
                
            # Tighter L-BFGS
            result = minimize(
                objective,
                X0.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'm': 20}
            )
            
            X_opt = _project_to_sphere(result.x.reshape(N, 3))
            e_opt, _ = _energy_and_grad(X_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
            
            # Second pass if time allows
            if time.time() < deadline - 5.0:
                result2 = minimize(
                    objective,
                    best_points.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15, 'm': 20}
                )
                X_opt2 = _project_to_sphere(result2.x.reshape(N, 3))
                e_opt2, _ = _energy_and_grad(X_opt2)
                
                if e_opt2 < best_energy:
                    best_energy = e_opt2
                    best_points = X_opt2
                    
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        # Fall back to Fibonacci initialization
        return _fibonacci_init(N)
    
    return best_points
