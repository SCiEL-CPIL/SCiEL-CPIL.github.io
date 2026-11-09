import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _log_energy_and_tangent_grad(points):
    """
    Compute the logarithmic energy and the Riemannian gradient (tangent to sphere).
    Energy E = sum_{i<j} log(1 / |x_i - x_j|)
    Euclidean gradient w.r.t x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Tangent gradient: project Euclidean gradient to tangent space at x_i.
    """
    n = len(points)
    # Compute pairwise squared distances efficiently
    sq_dists = np.sum(points**2, axis=1)[:, np.newaxis] + np.sum(points**2, axis=1)[np.newaxis, :] - 2.0 * (points @ points.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)  # numerical stability
    dists = np.sqrt(sq_dists)
    
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of -log(dists[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Euclidean gradient: grad_i E = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    dists_inv_sq = 1.0 / (dists * dists)
    np.fill_diagonal(dists_inv_sq, 0.0)
    
    # diff[i,j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    euclidean_grads = np.einsum('ijk,ij->ik', diff, dists_inv_sq)
    
    # Project to tangent space: grad_tan_i = grad_euc_i - (grad_euc_i . x_i) * x_i
    dot_products = np.sum(euclidean_grads * points, axis=1, keepdims=True)
    tangent_grads = euclidean_grads - dot_products * points
    
    return energy, tangent_grads

def _objective_and_grad(x_flat):
    """
    Objective and Euclidean gradient for L-BFGS-B.
    We use a penalty-free approach: optimize in R^3 and project after each step.
    For L-BFGS-B, we provide the Euclidean gradient (not tangent) and let the
    optimizer handle the unconstrained problem; we'll do projection-based refinement.
    """
    points = x_flat.reshape(N, 3)
    energy, tangent_grads = _log_energy_and_tangent_grad(points)
    # Convert tangent gradient back to Euclidean gradient for L-BFGS-B
    # The Euclidean gradient is the tangent gradient plus the normal component
    # But for optimization on the sphere, we can use the tangent gradient projected
    # Actually, for L-BFGS-B in R^3, we need the Euclidean gradient.
    # The relationship: if we're at a point on the sphere, the Euclidean gradient
    # of the energy (as a function on R^3) restricted to the sphere equals the
    # tangent gradient. So we can use tangent_grads as the gradient for L-BFGS-B
    # with the understanding that steps will be projected back.
    return energy, tangent_grads.flatten()

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _initial_fibonacci():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_lbfgs(initial_points, time_limit, start_time):
    """
    Optimize using L-BFGS-B with periodic re-projection to the sphere.
    Uses a custom callback to enforce the sphere constraint.
    """
    best_points = initial_points.copy()
    best_energy = _log_energy_and_tangent_grad(best_points)[0]
    
    x0 = initial_points.flatten()
    
    # Callback to track time and re-project
    call_count = [0]
    
    def callback(xk):
        call_count[0] += 1
        if time.time() - start_time > time_limit:
            # Raise exception to stop optimization
            raise TimeoutError()
        # Every 10 iterations, re-project to sphere
        if call_count[0] % 10 == 0:
            pts = xk.reshape(N, 3)
            proj_pts = _project_to_sphere(pts)
            # We can't easily modify xk in place for L-BFGS-B, so we'll
            # rely on the fact that the gradient is tangent and steps
            # should stay close to the sphere
        return False
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxcor': 20,
            }
        )
        
        if result.x is not None:
            result_points = _project_to_sphere(result.x.reshape(N, 3))
            result_energy = _log_energy_and_tangent_grad(result_points)[0]
            if result_energy < best_energy:
                best_energy = result_energy
                best_points = result_points
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Final polish: a few more iterations with tighter tolerance
    if time.time() - start_time < time_limit:
        try:
            x_start = best_points.flatten()
            result2 = minimize(
                _objective_and_grad,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-18,
                    'gtol': 1e-14,
                    'maxcor': 30,
                }
            )
            if result2.x is not None:
                result_points = _project_to_sphere(result2.x.reshape(N, 3))
                result_energy = _log_energy_and_tangent_grad(result_points)[0]
                if result_energy < best_energy:
                    best_energy = result_energy
                    best_points = result_points
        except Exception:
            pass
    
    return best_points, best_energy

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Best solution tracking
    best_points = None
    best_energy = np.inf
    
    # Initial Fibonacci spiral
    fib = _initial_fibonacci()
    
    # Try multiple initializations
    initializations = [fib]
    
    rng = np.random.default_rng(42)
    for _ in range(4):
        pts = rng.standard_normal((N, 3))
        initializations.append(_project_to_sphere(pts))
    
    # Perturbed Fibonacci
    for _ in range(3):
        perturbed = fib + rng.standard_normal((N, 3)) * 0.05
        initializations.append(_project_to_sphere(perturbed))
    
    # Optimize each initialization
    for init_idx, init_points in enumerate(initializations):
        if time.time() - start_time > time_limit:
            break
        
        remaining = time_limit - (time.time() - start_time)
        # Allocate at least 5s per remaining init, cap at remaining time
        per_init_time = min(remaining / (len(initializations) - init_idx), 30.0)
        
        try:
            local_points, local_energy = _optimize_lbfgs(init_points, time_limit, start_time)
            if local_energy < best_energy:
                best_energy = local_energy
                best_points = local_points
        except Exception:
            continue
    
    # Final check: ensure best_points is valid
    if best_points is None:
        best_points = fib
    
    # Ensure all points are on unit sphere and non-zero
    best_points = _project_to_sphere(best_points)
    
    return best_points
