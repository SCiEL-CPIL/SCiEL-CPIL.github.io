"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: for point x_i, grad = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    N = X.shape[0]
    # Pairwise differences
    # X: (N, 3), diff[i,j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of -log(dists[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (X[i] - X[j]) / dists[i,j]^2
    # diff[i,j] = X[i] - X[j], so grad_i = sum_j diff[i,j] / dists[i,j]^2
    # dists[i,j]^2
    dists_sq = dists ** 2
    np.fill_diagonal(dists_sq, 1.0)
    # weight[i,j] = 1/dists_sq[i,j]
    weights = 1.0 / dists_sq
    np.fill_diagonal(weights, 0.0)
    # grad[i] = sum_j weights[i,j] * diff[i,j]
    grad = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _riemannian_cg(X, max_iter=2000, tol=1e-10, time_limit=None, start_time=None):
    """Riemannian conjugate gradient on the sphere."""
    if start_time is None:
        start_time = time.time()
    
    N = X.shape[0]
    X = _project_to_sphere(X.copy())
    
    energy, grad = _energy_and_grad(X)
    
    # Initial search direction: negative gradient projected to tangent space
    grad_tan = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    p = -grad_tan
    
    # Fletcher-Reeves beta
    beta = 0.0
    
    for k in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit - 5:
            break
        
        # Check convergence
        if np.linalg.norm(grad_tan) < tol:
            break
        
        # Line search: minimize along geodesic
        # X(t) = X * cos(t) + p * sin(t) / ||p|| (for small t, approximate)
        # For simplicity, use exponential map: X_new = X * cos(t) + (p/||p||) * sin(t)
        
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-15:
            break
        p_unit = p / p_norm
        
        # Backtracking line search
        t = 1.0
        best_t = 0.0
        best_energy = energy
        best_X = X.copy()
        
        for ls_iter in range(30):
            # Exponential map
            c = np.cos(t)
            s = np.sin(t)
            X_new = X * c + p_unit * s
            
            # Project to ensure exact unit sphere (numerical stability)
            X_new = _project_to_sphere(X_new)
            
            E_new, _ = _energy_and_grad(X_new)
            
            if E_new < best_energy:
                best_energy = E_new
                best_X = X_new
                best_t = t
            
            # Armijo condition
            if E_new < energy - 1e-4 * t * np.sum(grad_tan * p):
                break
            
            t *= 0.5
        
        if best_t == 0.0:
            # No improvement, try smaller steps
            for t_try in [0.01, 0.001, 0.0001]:
                c = np.cos(t_try)
                s = np.sin(t_try)
                X_new = _project_to_sphere(X * c + p_unit * s)
                E_new, _ = _energy_and_grad(X_new)
                if E_new < energy:
                    best_X = X_new
                    best_energy = E_new
                    best_t = t_try
                    break
        
        if best_t == 0.0:
            break
        
        X = best_X
        energy = best_energy
        
        _, grad_new = _energy_and_grad(X)
        grad_tan_new = grad_new - np.sum(grad_new * X, axis=1, keepdims=True) * X
        
        # Fletcher-Reeves beta
        beta = np.sum(grad_tan_new * grad_tan_new) / (np.sum(grad_tan * grad_tan) + 1e-15)
        # Damping
        beta = min(beta, 0.5)
        
        grad_tan = grad_tan_new
        p = -grad_tan + beta * p
        
        # Ensure p is in tangent space
        p = p - np.sum(p * X, axis=1, keepdims=True) * X
    
    return X, energy


def _perturb_and_optimize(X, rng, time_limit, start_time):
    """Perturb points and re-optimize."""
    # Random perturbation in tangent space
    noise_scale = 0.05
    noise = rng.standard_normal(X.shape)
    # Project noise to tangent space
    noise_tan = noise - np.sum(noise * X, axis=1, keepdims=True) * X
    X_perturbed = X + noise_scale * noise_tan
    X_perturbed = _project_to_sphere(X_perturbed)
    
    X_opt, E = _riemannian_cg(X_perturbed, max_iter=500, tol=1e-10, 
                               time_limit=time_limit, start_time=start_time)
    return X_opt, E


def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse initial configurations
    best_X = None
    best_E = np.inf
    
    # Configuration 1: Golden angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X1 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Configuration 2: Random points
    X2 = rng.standard_normal((N, 3))
    X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    
    # Configuration 3: Two golden spirals with offset
    i3 = np.arange(N) + 0.5
    z3 = 1.0 - 2.0 * i3 / N
    phi3 = np.pi * (1.0 + np.sqrt(5.0)) * i3 + np.pi / 3
    r3 = np.sqrt(1.0 - z3 * z3)
    X3 = np.column_stack([r3 * np.cos(phi3), r3 * np.sin(phi3), z3])
    
    # Configuration 4: Fibonacci sphere with different seed
    i4 = np.arange(N) + 0.5
    z4 = 1.0 - 2.0 * i4 / N
    phi4 = np.pi * (1.0 + np.sqrt(5.0)) * i4 + 0.7
    r4 = np.sqrt(1.0 - z4 * z4)
    X4 = np.column_stack([r4 * np.cos(phi4), r4 * np.sin(phi4), z4])
    
    configs = [X1, X2, X3, X4]
    
    for idx, X_init in enumerate(configs):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Optimize from this configuration
        X_opt, E = _riemannian_cg(X_init, max_iter=2000, tol=1e-10,
                                   time_limit=time_limit, start_time=start_time)
        
        if E < best_E:
            best_E = E
            best_X = X_opt
        
        # Try perturbations from this local minimum
        if time.time() - start_time > time_limit - 15:
            break
        
        for p_iter in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            X_pert, E_pert = _perturb_and_optimize(X_opt, rng, time_limit, start_time)
            if E_pert < best_E:
                best_E = E_pert
                best_X = X_pert
    
    # Final check
    if best_X is None:
        best_X = X1
    
    # Ensure all points are on unit sphere and well-separated
    best_X = _project_to_sphere(best_X)
    
    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < min_dist:
                min_dist = d
    
    return best_X

# EVOLVE-BLOCK-END
