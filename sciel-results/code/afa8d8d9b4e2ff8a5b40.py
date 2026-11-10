"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise differences: (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Pairwise squared distances: (n, n)
    dist_sq = np.sum(diff * diff, axis=2)

    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist_sq, 1.0)

    # Energy: sum over i < j of -log(dist) = -0.5 * sum log(dist_sq)
    triu_indices = np.triu_indices(n, k=1)
    dist_sq_triu = dist_sq[triu_indices]
    energy = -0.5 * np.sum(np.log(dist_sq_triu))

    # Gradient calculation
    # grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Let W_ij = 1 / |x_i - x_j|^2 for i != j, and 0 for i == j
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)

    # grad[i] = -sum_j diff[i, j] * inv_dist_sq[i, j]
    # diff[i, j] is (x_i - x_j)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

    return energy, grad

def _project_to_tangent(pts, grad):
    """
    Project the Euclidean gradient to the tangent space of the unit sphere.
    T_p S^2 = { v in R^3 | v . p = 0 }
    Projection: g_tan = g - (g . p) p
    """
    # pts: (n, 3), grad: (n, 3)
    # Dot product: (n,)
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    grad_tan = grad - dot * pts
    return grad_tan

def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _make_fibonacci_sphere(n):
    """Generate points on the sphere using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z)) # Clamp to avoid sqrt of negative due to FP
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    # Generate random vectors and normalize
    pts = rng.normal(size=(n, 3))
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # List of initial configurations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_make_fibonacci_sphere(N))
    
    # 2. Random points (several seeds)
    for seed in [1, 2, 3, 42, 123]:
        initializations.append(_random_sphere(N, seed))
        
    # 3. Perturbed Fibonacci
    rng = np.random.default_rng(99)
    fib = _make_fibonacci_sphere(N)
    for scale in [0.05, 0.1, 0.2]:
        perturbed = fib + rng.normal(0, scale, size=fib.shape)
        initializations.append(_normalize(perturbed))

    # Optimization function wrapper
    def obj_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        # Normalize to ensure we stay on the manifold during intermediate steps
        # Note: L-BFGS-B steps in R^3, so we must normalize before calculating energy/grad
        # However, the gradient of the normalized energy is not simply the projected gradient of the unnormalized energy.
        # Standard approach: minimize f(p) where p is on sphere.
        # If we use L-BFGS-B on R^3, we are effectively doing:
        # p_new = p - alpha * g_proj
        # Then normalize p_new.
        # This is a heuristic. A more rigorous method is Riemannian optimization.
        # Given the time constraint, this heuristic usually works well for this problem.
        
        e, g = _energy_and_grad(pts)
        g_tan = _project_to_tangent(pts, g)
        return e, g_tan.ravel()

    # Run optimization for each initialization
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
            
        current_energy, _ = _energy_and_grad(init_pts)
        
        # If this initialization is already worse than the best found, skip if we have enough restarts
        if current_energy > best_energy and i > 2:
             # Allow a small tolerance for exploration
             if current_energy - best_energy > 5.0:
                 continue

        x0 = init_pts.ravel()
        
        try:
            # Phase 1: L-BFGS-B with moderate tolerances
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            res_pts = _normalize(result.x.reshape(N, 3))
            res_energy, _ = _energy_and_grad(res_pts)
            
            # Phase 2: Tighter polish
            if res_energy < best_energy and time.time() < deadline - 10.0:
                result2 = minimize(
                    obj_grad,
                    res_pts.ravel(),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'maxls': 100,
                        'disp': False
                    }
                )
                res_pts = _normalize(result2.x.reshape(N, 3))
                res_energy, _ = _energy_and_grad(res_pts)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
                
        except Exception as e:
            # If optimization fails, keep the initial points if they are better
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = init_pts.copy()
                
    # Final high-precision polish on the best configuration
    if best_pts is not None and time.time() < deadline - 3.0:
        try:
            x0 = best_pts.ravel()
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 200,
                    'disp': False
                }
            )
            res_pts = _normalize(result.x.reshape(N, 3))
            res_energy, _ = _energy_and_grad(res_pts)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except:
            pass

    # Coordinate-descent polish with line search
    if best_pts is not None and time.time() < deadline - 1.0:
        pts = best_pts.copy()
        cur_e, _ = _energy_and_grad(pts)
        for _cd_round in range(5):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                p = pts[i]
                # Generate multiple random tangent directions
                for _trial in range(6):
                    rng_cd = np.random.default_rng(hash((i, _trial, _cd_round)) % (2**31))
                    v = rng_cd.normal(size=3)
                    v_tan = v - np.dot(v, p) * p
                    nv = np.linalg.norm(v_tan)
                    if nv < 1e-15:
                        continue
                    v_tan /= nv
                    # Try multiple scales with backtracking
                    best_scale = 0.0
                    best_new_e = cur_e
                    for scale in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
                        new_p = p + scale * v_tan
                        new_p = new_p / np.linalg.norm(new_p)
                        old_p = pts[i].copy()
                        pts[i] = new_p
                        new_e, _ = _energy_and_grad(pts)
                        if new_e < best_new_e - 1e-16:
                            best_new_e = new_e
                            best_scale = scale
                        pts[i] = old_p
                    if best_scale > 0:
                        pts[i] = p + best_scale * v_tan
                        pts[i] = pts[i] / np.linalg.norm(pts[i])
                        cur_e = best_new_e
                        improved = True
            if not improved:
                break
        res_energy, _ = _energy_and_grad(pts)
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = pts.copy()

    # Final validity checks
    if best_pts is None:
        best_pts = _make_fibonacci_sphere(N)
        
    # Ensure no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _make_fibonacci_sphere(N)
        
    if not np.all(np.isfinite(best_pts)):
        best_pts = _make_fibonacci_sphere(N)

    return best_pts

# EVOLVE-BLOCK-END
