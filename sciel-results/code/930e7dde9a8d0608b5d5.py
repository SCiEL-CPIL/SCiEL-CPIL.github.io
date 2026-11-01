"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150

def _energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_and_grad(X):
    """Compute logarithmic energy and Riemannian gradient for points on the unit sphere."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Euclidean gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    eucl_grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _icosahedral_init(n, rng_seed=42):
    """
    Initialize n points using icosahedral symmetry.
    We place orbits under the icosahedral group.
    For n=150, we can use orbits of sizes 12, 20, 30, 60.
    150 = 12 + 20 + 30 + 60 + 28 (not clean)
    Let's use: 2 orbits of 60 + 1 orbit of 30 = 150
    Or: 1 orbit of 60 + 1 orbit of 30 + 1 orbit of 20 + 1 orbit of 12 + 1 orbit of 28 (no)
    
    Better approach: Use a set of "generator" points and apply the icosahedral rotation group.
    The icosahedral group has 60 elements.
    Orbit sizes can be 60 (generic), 30 (on a 2-fold axis), 20 (on a 5-fold axis), 12 (on a 3-fold axis).
    
    150 = 60 + 60 + 30
    So we can have two generic orbits and one 2-fold orbit.
    
    Alternatively, just use a good random init with some symmetry breaking.
    Let's try a simpler approach: start with icosahedron vertices (12 points) and then
    use a Fibonacci spiral for the rest, or use a more structured approach.
    
    Actually, for n=150, a common good configuration is related to spherical codes.
    Let's use a combination: 12 icosahedron vertices + 138 Fibonacci points,
    or better, use the fact that 150 is close to 144 (12*12) or 160.
    
    Let's try: Start with icosahedron vertices, then add points via a spiral,
    but perturb to break symmetry if needed.
    """
    # Use icosahedron as base
    base = _icosahedron_vertices()  # 12 points
    
    if n == 12:
        return base
    
    # Add more points using a spiral that avoids the base points
    # This is a heuristic init; the optimizer will refine it
    remaining = n - 12
    if remaining <= 0:
        return base[:n]
    
    # Generate remaining points via Fibonacci spiral
    fib_pts = _fibonacci_init(remaining)
    
    # Combine
    points = np.vstack([base, fib_pts])
    return _project_to_sphere(points)

def _riemannian_gd(X0, max_iter=1000, tol=1e-10, time_limit=10.0):
    """Riemannian Gradient Descent with momentum and backtracking line search."""
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    alpha = 0.5
    c = 1e-4
    momentum = 0.0
    velocity = np.zeros_like(X)
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        if k >= 20:
            momentum = 0.3
        elif k >= 10:
            momentum = 0.15
        
        effective_grad = grad + momentum * velocity
        
        alpha_trial = alpha
        step = alpha_trial * effective_grad
        X_new = X - step
        X_new = _project_to_sphere(X_new)
        
        energy_new, _ = _energy_and_grad(X_new)
        
        count = 0
        while energy_new > energy - c * alpha_trial * grad_norm**2:
            alpha_trial *= 0.5
            step = alpha_trial * effective_grad
            X_new = X - step
            X_new = _project_to_sphere(X_new)
            energy_new, _ = _energy_and_grad(X_new)
            count += 1
            if alpha_trial < 1e-15 or count > 30:
                break
        
        if alpha_trial < 1e-15:
            alpha = max(alpha * 0.5, 1e-10)
            continue
            
        velocity = momentum * velocity - alpha_trial * grad
        
        X = X_new
        energy = energy_new
        _, grad = _energy_and_grad(X)
        grad_norm = np.linalg.norm(grad)
        
        if count == 0:
            alpha = min(alpha * 1.2, 1.0)
        else:
            alpha = max(alpha * 0.8, 1e-10)
        
        if grad_norm < tol:
            break
            
    return X, energy

def _lbfgs_polish(X0, max_iter=1000, time_limit=20.0):
    """Polish using L-BFGS-B in R^3 with projection."""
    start_time = time.time()
    
    def objective(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        return _energy(X)
    
    def gradient(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        _, g = _energy_and_grad(X)
        return g.flatten()
    
    try:
        result = minimize(
            objective,
            X0.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'm': 20}
        )
        X_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _energy(X_opt)
        return X_opt, e_opt
    except Exception:
        return X0, _energy(X0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_init(N))
    
    # 2. Icosahedral-based init
    initializations.append(_icosahedral_init(N))
    
    # 3. Random with fixed seed
    rng = np.random.RandomState(123)
    rand_pts = rng.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts))
    
    # 4. Another random seed
    rng2 = np.random.RandomState(456)
    rand_pts2 = rng2.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts2))
    
    # 5. Fibonacci with perturbation
    fib_perturbed = _fibonacci_init(N).copy()
    rng4 = np.random.RandomState(321)
    fib_perturbed += rng4.normal(0, 0.05, size=fib_perturbed.shape)
    initializations.append(_project_to_sphere(fib_perturbed))
    
    # 6. Icosahedron with perturbation
    ico_perturbed = _icosahedral_init(N).copy()
    rng5 = np.random.RandomState(789)
    ico_perturbed += rng5.normal(0, 0.02, size=ico_perturbed.shape)
    initializations.append(_project_to_sphere(ico_perturbed))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: Riemannian GD from multiple starts
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 40.0:
            break
            
        time_budget = (deadline - 40.0 - time.time()) / max(1, len(initializations) - init_idx)
        time_budget = max(time_budget, 5.0)
        
        try:
            X_opt, e_opt = _riemannian_gd(X0, max_iter=300, tol=1e-8, time_limit=time_budget)
            
            if time.time() < deadline - 20.0:
                X_polished, e_polished = _lbfgs_polish(X_opt, max_iter=500, time_limit=15.0)
                
                if e_polished < e_opt:
                    X_opt = X_polished
                    e_opt = e_polished
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt.copy()
                
        except Exception:
            pass

    # Phase 2: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            X_opt, e_opt = _lbfgs_polish(best_points, max_iter=2000, time_limit=30.0)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
            
            if time.time() < deadline - 3.0:
                X_opt2, e_opt2 = _lbfgs_polish(best_points, max_iter=1000, time_limit=10.0)
                if e_opt2 < best_energy:
                    best_energy = e_opt2
                    best_points = X_opt2
        except Exception:
            pass
    
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        return _fibonacci_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
