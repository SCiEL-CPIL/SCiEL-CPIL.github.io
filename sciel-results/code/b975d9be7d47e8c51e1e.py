"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based(n):
    """Generate n points based on icosahedron vertices with subdivision."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Start with icosahedron, then add points by finding gaps
    points = [verts]
    total = len(verts)
    
    while total < n:
        # Find the largest gap
        best_gap = 0
        best_idx = 0
        best_point = None
        for i, v in enumerate(verts):
            # Find minimum distance to other vertices
            min_d = min(np.linalg.norm(v - u) for u in verts if u is not v)
            if min_d > best_gap:
                best_gap = min_d
                best_idx = i
                best_point = v
        
        # Add a new point in the direction away from neighbors
        # Simple approach: add points at random positions and keep best
        for _ in range(n - total):
            # Random point on sphere
            u = np.random.randn(3)
            u = u / np.linalg.norm(u)
            # Check minimum distance to existing points
            min_d = min(np.linalg.norm(u - p) for p in verts)
            if min_d > 0.05:  # Minimum separation
                verts = np.vstack([verts, u])
                total += 1
                if total >= n:
                    break
        
        if total >= n:
            break
    
    return verts[:n]


def _energy_and_grad(flat_x):
    """Compute energy and gradient for L-BFGS-B.
    flat_x is a 1D array of shape (3*N,) representing N points on sphere.
    Returns (energy, gradient) where gradient is with respect to flat_x.
    """
    x = flat_x.reshape(N, 3)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid log(0) on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(N, k=1)
    energy = -math.fsum(np.log(dists[iu[0], iu[1]]).tolist())
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad.flatten()


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _lbfgs_optimize(x0, maxiter=500, ftol=1e-12, gtol=1e-10, deadline=None):
    """Run L-BFGS-B optimization on the sphere using reparameterization."""
    # Reparameterize: use unconstrained variables and project to sphere
    # We'll use the fact that we can optimize in R^3 and project
    
    def objective_and_grad(flat_x):
        x = flat_x.reshape(N, 3)
        # Project to sphere
        x = _project_to_sphere(x)
        e, g = _energy_and_grad(x.flatten())
        # Project gradient to tangent space
        grad_t = g.reshape(N, 3) - np.sum(g.reshape(N, 3) * x, axis=1, keepdims=True) * x
        return e, grad_t.flatten()
    
    result = minimize(
        objective_and_grad,
        x0.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    
    x_opt = result.x.reshape(N, 3)
    x_opt = _project_to_sphere(x_opt)
    e_opt, _ = _energy_and_grad(x_opt.flatten())
    
    return x_opt, e_opt


def _gradient_descent_polish(x, max_iter=2000, tol=1e-14, deadline=None):
    """Polish with gradient descent on the sphere using adaptive step size."""
    x = x.copy()
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e, _ = _energy_and_grad(x.flatten())
    
    # Compute initial gradient
    _, g = _energy_and_grad(x.flatten())
    g = g.reshape(N, 3)
    # Project to tangent space
    grad_t = g - np.sum(g * x, axis=1, keepdims=True) * x
    
    # Adaptive step size
    alpha = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try step
        x_trial = _project_to_sphere(x + alpha * grad_t)
        e_trial, g_trial = _energy_and_grad(x_trial.flatten())
        g_trial = g_trial.reshape(N, 3)
        grad_t_trial = g_trial - np.sum(g_trial * x_trial, axis=1, keepdims=True) * x_trial
        
        if e_trial < best_e - 1e-15:
            # Accept step
            x = x_trial
            best_x = x.copy()
            best_e = e_trial
            grad_t = grad_t_trial
            # Increase step size if we made good progress
            if it % 10 == 0:
                alpha = min(alpha * 1.1, 0.1)
        else:
            # Decrease step size
            alpha *= 0.5
            if alpha < 1e-15:
                break
        
        # Check convergence
        if np.linalg.norm(grad_t) < tol:
            break
    
    return best_x, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    x_fib = _fibonacci_sphere(N)
    x_fib = _project_to_sphere(x_fib)
    initializations.append(x_fib)
    
    # 2. Icosahedron-based
    x_ico = _icosahedron_based(N)
    x_ico = _project_to_sphere(x_ico)
    initializations.append(x_ico)
    
    # 3. Random with multiple seeds
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        x_rand = rng.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(x_rand)
    
    best_x = initializations[0].copy()
    best_e = _energy_and_grad(best_x.flatten())[0]
    
    # Optimize each initialization
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # L-BFGS-B optimization
        try:
            x_opt, e_opt = _lbfgs_optimize(x_init, maxiter=300, ftol=1e-12, gtol=1e-10, deadline=deadline - 10)
            if e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
        
        # Gradient descent polish
        if time.time() > deadline - 15:
            break
        try:
            x_pol, e_pol = _gradient_descent_polish(best_x, max_iter=1000, tol=1e-14, deadline=deadline - 5)
            if e_pol < best_e:
                best_x = x_pol
                best_e = e_pol
        except Exception:
            pass
    
    # Final tight polish if time remains
    if time.time() < deadline - 5:
        try:
            x_final, e_final = _gradient_descent_polish(best_x, max_iter=5000, tol=1e-15, deadline=deadline - 1)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
        except Exception:
            pass
    
    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    min_dist = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                # Nudge point j slightly
                rng = np.random.RandomState(12345 + i * 100 + j)
                best_x[j] += 1e-10 * rng.randn(3)
                best_x = _project_to_sphere(best_x)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
