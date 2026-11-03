"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def _compute_energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    X: (N, 3) array of unit vectors.
    Returns (energy, grad) where grad is (N, 3).
    """
    # Compute all pairwise distances
    # X[:, None, :] - X[None, :, :] gives (N, N, 3)
    diff = X[:, None, :] - X[None, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # We need to be careful not to double count
    # Create a mask for upper triangle
    iu = np.triu_indices(N, k=1)
    d_upper = dist[iu]  # distances for i<j
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: d/dX_i sum_{j != i} log(1/|X_i - X_j|) = d/dX_i (-sum_{j != i} log|X_i - X_j|)
    # = -sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
    # But we need to project onto the sphere tangent space.
    # Actually, for points constrained to the sphere, the gradient of the energy
    # in ambient space projected to the tangent space is what we need.
    # 
    # The ambient gradient for point i is:
    # g_i = -sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
    # 
    # The projected gradient (tangent to sphere) is:
    # g_i_proj = g_i - (g_i . X_i) * X_i
    
    # Compute g_i for all i
    # For each pair (i, j), contribution to grad[i] is -(X_i - X_j)/dist(i,j)^2
    # Contribution to grad[j] is +(X_i - X_j)/dist(i,j)^2 = -(X_j - X_i)/dist(i,j)^2
    
    # Vectorized computation:
    # diff[i,j] = X_i - X_j
    # contrib[i,j] = -diff[i,j] / dist_sq[i,j]  for i != j
    # grad[i] = sum_j contrib[i,j]
    
    # Safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # contrib[i,j] = -diff[i,j] * inv_dist_sq[i,j]
    contrib = -diff * inv_dist_sq[:, :, None]  # (N, N, 3)
    grad = np.sum(contrib, axis=1)  # (N, 3)
    
    # Project gradient onto tangent space: g_proj = g - (g . X) * X
    grad_dot_X = np.sum(grad * X, axis=1)  # (N,)
    grad_proj = grad - grad_dot_X[:, None] * X  # (N, 3)
    
    return energy, grad_proj


def _project_to_sphere(X):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = _project_to_sphere(X)
    
    best_X = X.copy()
    best_energy, _ = _compute_energy_and_grad(X)
    
    # Multi-start with gradient-based optimization using L-BFGS-B
    # We'll use scipy's L-BFGS-B but parameterize points as (theta, phi) to avoid
    # the sphere constraint, or use the projected gradient approach.
    
    # Actually, let's use a cleaner approach: parameterize each point by (theta, phi)
    # and use L-BFGS-B on these parameters.
    
    from scipy.optimize import minimize
    
    def params_to_points(params):
        """Convert (theta, phi) pairs to 3D points on sphere.
        theta in [0, pi], phi in [0, 2*pi)
        """
        theta = params[0::2]  # (N,)
        phi = params[1::2]    # (N,)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def points_to_params(X):
        """Convert 3D points to (theta, phi) parameters."""
        z = X[:, 2]
        r_xy = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(X[:, 1], X[:, 0])
        # Ensure phi in [0, 2*pi)
        phi = np.mod(phi, 2 * np.pi)
        params = np.zeros(2 * N)
        params[0::2] = theta
        params[1::2] = phi
        return params
    
    def energy_from_params(params):
        """Compute energy from (theta, phi) parameters."""
        X = params_to_points(params)
        e, _ = _compute_energy_and_grad(X)
        return e
    
    def grad_from_params(params):
        """Compute gradient of energy w.r.t. (theta, phi) parameters."""
        X = params_to_points(params)
        e, g_proj = _compute_energy_and_grad(X)
        
        # Convert projected gradient (in 3D tangent space) to (theta, phi) gradient
        # X = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
        # dX/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
        # dX/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
        
        theta = params[0::2]
        phi = params[1::2]
        
        dX_dtheta = np.column_stack([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta)
        ])  # (N, 3)
        
        dX_dphi = np.column_stack([
            -np.sin(theta) * np.sin(phi),
            np.sin(theta) * np.cos(phi),
            np.zeros(N)
        ])  # (N, 3)
        
        # grad_theta[i] = g_proj[i] . dX_dtheta[i]
        # grad_phi[i] = g_proj[i] . dX_dphi[i]
        
        grad_theta = np.sum(g_proj * dX_dtheta, axis=1)  # (N,)
        grad_phi = np.sum(g_proj * dX_dphi, axis=1)      # (N,)
        
        grad_params = np.zeros(2 * N)
        grad_params[0::2] = grad_theta
        grad_params[1::2] = grad_phi
        
        return e, grad_params
    
    # Run multiple restarts with different initial configurations
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            # Use the Fibonacci spiral as the first start
            X_init = X.copy()
        else:
            # Generate random initial configurations with some structure
            rng = np.random.RandomState(restart * 12345 + 67890)
            
            if restart % 2 == 1:
                # Random points on sphere
                X_init = rng.randn(N, 3)
                X_init = _project_to_sphere(X_init)
            else:
                # Perturbed Fibonacci spiral
                X_init = X.copy()
                # Add random perturbations
                noise = rng.randn(N, 3) * 0.1
                X_init = X_init + noise
                X_init = _project_to_sphere(X_init)
        
        params_init = points_to_params(X_init)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                grad_from_params,
                params_init,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 50,
                    'maxfun': 10000,
                    'disp': False,
                }
            )
            
            if result.success or result.fun < best_energy:
                X_opt = params_to_points(result.x)
                X_opt = _project_to_sphere(X_opt)
                e_opt, _ = _compute_energy_and_grad(X_opt)
                
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_X = X_opt.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 60.0:
            break
    
    # Final polish with tighter tolerances if time permits
    if time.time() < deadline - 30.0:
        params_best = points_to_params(best_X)
        try:
            result = minimize(
                grad_from_params,
                params_best,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 10000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'maxfun': 20000,
                    'disp': False,
                }
            )
            if result.fun < best_energy:
                X_opt = params_to_points(result.x)
                X_opt = _project_to_sphere(X_opt)
                e_opt, _ = _compute_energy_and_grad(X_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_X = X_opt.copy()
        except Exception:
            pass
    
    # Ensure all points are valid (nonzero, on unit sphere, finite)
    best_X = _project_to_sphere(best_X)
    
    # Final validation
    if not np.all(np.isfinite(best_X)):
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_X = _project_to_sphere(best_X)
    
    return best_X
# EVOLVE-BLOCK-END
