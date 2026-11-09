"""
Riemannian optimization on the sphere using log-sum-exp surrogate of covering radius.
Combines smooth pairwise repulsion with facet-aware penalties for better convergence.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    facets = hull.simplices
    if len(facets) == 0:
        return np.pi
    
    min_cos = 1.0
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_val = p @ a
        if cos_val < min_cos:
            min_cos = cos_val
    
    return np.arccos(np.clip(min_cos, -1.0, 1.0))


def _soft_max_repulsion(points, temperature=0.01):
    """
    Compute a smooth approximation to the max pairwise dot product using
    log-sum-exp. Returns (value, gradient) where value approximates
    max_{i,j} dot(x_i, x_j) and gradient is w.r.t. points.
    """
    n = points.shape[0]
    dot = points @ points.T
    dot_no_diag = dot.copy()
    np.fill_diagonal(dot_no_diag, -1e10)
    
    # Soft-max over all pairs (symmetric, so we can use full matrix)
    max_dot = np.max(dot_no_diag, axis=1, keepdims=True)
    exp_dot = np.exp((dot_no_diag - max_dot) / temperature)
    log_sum_exp = np.log(np.sum(exp_dot, axis=1, keepdims=True)) + max_dot / temperature
    
    # Use the max of soft-max values as surrogate for max pairwise dot
    value = np.max(log_sum_exp)
    
    softmax_weights = exp_dot / np.sum(exp_dot, axis=1, keepdims=True)  # (n, n)
    
    # Gradient of sum_i soft_max_i w.r.t. x_k:
    # d/dx_k [sum_i T*log(sum_j exp(x_i·x_j/T))] 
    # = sum_i softmax[i,k] * x_i  (from i-th term where j=k)
    #   + sum_j softmax[k,j] * x_j  (from k-th term where i=k)
    # By symmetry of softmax, these are related but let's compute properly.
    
    # For the objective f = sum_i T * log(sum_j exp(x_i·x_j / T)):
    # df/dx_k = sum_i (1/T) * (exp(x_i·x_k/T) / sum_j exp(x_i·x_j/T)) * x_i
    #           + sum_j (1/T) * (exp(x_k·x_j/T) / sum_i exp(x_k·x_i/T)) * x_j
    #         = sum_i softmax[i,k] * x_i + sum_j softmax[k,j] * x_j
    
    grad_from_rows = softmax_weights.T @ points  # sum_i softmax[i,k] * x_i for each k
    grad_from_cols = softmax_weights @ points    # sum_j softmax[k,j] * x_j for each k
    grad = grad_from_rows + grad_from_cols
    
    return value, grad


def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return points / norms


def _soft_max_optimize(points, deadline, max_iters=5000, base_lr=0.01, 
                        temperature_start=0.05, temperature_end=0.005):
    """
    Optimize using Riemannian gradient descent on the soft-max repulsion objective.
    """
    points = points.copy().astype(np.float64)
    best_value = _covering_radius(points)
    best_points = points.copy()
    
    lr = base_lr
    patience = 0
    max_patience = 150
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Anneal temperature with exponential decay
        progress = it / max(max_iters - 1, 1)
        temperature = temperature_start * (temperature_end / temperature_start) ** progress
        
        # Compute soft-max objective and gradient
        value, grad = _soft_max_repulsion(points, temperature=temperature)
        
        # Project gradient onto tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Adaptive step size based on gradient norm
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm > 1e-10:
            step_lr = lr * min(1.0, 0.1 / grad_norm)
        else:
            step_lr = lr
        
        # Gradient descent step
        new_points = points - step_lr * grad_tangent
        new_points = _project_to_sphere(new_points)
        
        # Evaluate actual covering radius (expensive, do less frequently)
        if it % 30 == 0 or it == max_iters - 1:
            cr = _covering_radius(new_points)
            if cr < best_value:
                best_value = cr
                best_points = new_points.copy()
                patience = 0
                lr = min(lr * 1.08, 0.03)
            else:
                patience += 1
                lr *= 0.92
            
            if patience > max_patience:
                # Restart with perturbation
                rng = np.random.default_rng(it * 7 + 13)
                perturbation = 0.15 * rng.standard_normal((N, 3))
                new_points = best_points + perturbation
                new_points = _project_to_sphere(new_points)
                patience = 0
                lr = base_lr
        
        points = new_points
    
    return best_points, best_value


def _facet_aware_optimize(points, deadline, max_iters=2000, base_lr=0.003,
                           temperature=0.02):
    """
    Facet-aware optimization: identify worst facets from convex hull and
    apply targeted repulsion from facet circumcenters.
    """
    points = points.copy().astype(np.float64)
    best_value = _covering_radius(points)
    best_points = points.copy()
    
    lr = base_lr
    patience = 0
    max_patience = 100
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Compute pairwise repulsion (smooth)
        value, grad = _soft_max_repulsion(points, temperature=temperature)
        
        # Add facet-aware forces: push points away from worst facet circumcenters
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            
            # Compute circumcenters and find worst facets
            facet_centers = []
            facet_radii = []
            for facet in facets:
                a = points[facet[0]]
                b = points[facet[1]]
                c = points[facet[2]]
                v1 = a - b
                v2 = b - c
                p = np.cross(v1, v2)
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-12:
                    continue
                p = p / p_norm
                if p @ a < 0:
                    p = -p
                cos_val = p @ a
                if cos_val > -1.0:
                    radius = np.arccos(np.clip(cos_val, -1.0, 1.0))
                    facet_centers.append(p)
                    facet_radii.append(radius)
            
            if facet_centers:
                facet_centers = np.array(facet_centers)
                facet_radii = np.array(facet_radii)
                
                # Focus on worst 20% of facets
                n_worst = max(1, len(facet_radii) // 5)
                worst_idx = np.argsort(-facet_radii)[:n_worst]
                
                # For each point, compute force away from nearby worst facet centers
                forces = np.zeros_like(points)
                for i in range(N):
                    # Find facet centers that are close to this point
                    dots = points[i] @ facet_centers[worst_idx].T
                    # Weight by proximity and facet badness
                    weights = np.exp(-dots * 3.0) * np.exp(facet_radii[worst_idx] * 2.0)
                    if np.sum(weights) > 1e-10:
                        # Force direction: away from weighted average of facet centers
                        avg_center = np.average(facet_centers[worst_idx], axis=0, weights=weights)
                        avg_center = avg_center / np.linalg.norm(avg_center)
                        # Tangent direction away from center
                        tangent = -avg_center + np.dot(avg_center, points[i]) * points[i]
                        norm_t = np.linalg.norm(tangent)
                        if norm_t > 1e-10:
                            forces[i] += 0.5 * tangent / norm_t
                
                # Combine with soft-max gradient
                grad = grad + forces
        except Exception:
            pass
        
        # Project gradient onto tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Adaptive step
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm > 1e-10:
            step_lr = lr * min(1.0, 0.05 / grad_norm)
        else:
            step_lr = lr
        
        new_points = points - step_lr * grad_tangent
        new_points = _project_to_sphere(new_points)
        
        # Evaluate
        if it % 20 == 0 or it == max_iters - 1:
            cr = _covering_radius(new_points)
            if cr < best_value:
                best_value = cr
                best_points = new_points.copy()
                patience = 0
                lr = min(lr * 1.05, 0.015)
            else:
                patience += 1
                lr *= 0.93
            
            if patience > max_patience:
                rng = np.random.default_rng(it * 11 + 7)
                perturbation = 0.12 * rng.standard_normal((N, 3))
                new_points = best_points + perturbation
                new_points = _project_to_sphere(new_points)
                patience = 0
                lr = base_lr
        
        points = new_points
    
    return best_points, best_value


def _fibonacci_spiral(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Phase 1: Start with Fibonacci spiral and optimize with soft-max
    fib_points = _fibonacci_spiral(N)
    points, radius = _soft_max_optimize(fib_points, deadline, max_iters=2500, 
                                          base_lr=0.006, temperature_start=0.04,
                                          temperature_end=0.003)
    best_points = points
    best_radius = radius
    
    # Phase 2: Facet-aware optimization from best
    if time.time() < deadline - 35.0:
        points, radius = _facet_aware_optimize(best_points, deadline, max_iters=1500,
                                                 base_lr=0.003, temperature=0.02)
        if radius < best_radius:
            best_radius = radius
            best_points = points
    
    # Phase 3: Random restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        # Alternate between perturbing best, Fibonacci, and random init
        if restart % 3 == 0:
            perturbation = 0.10 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = _project_to_sphere(points)
        elif restart % 3 == 1:
            points = _fibonacci_spiral(N)
            perturbation = 0.05 * rng.standard_normal((N, 3))
            points = points + perturbation
            points = _project_to_sphere(points)
        else:
            points = rng.standard_normal((N, 3))
            points = _project_to_sphere(points)
        
        # Use soft-max optimization
        max_iters = min(1500, int(remaining * 12))
        optimized, radius = _soft_max_optimize(points, deadline, max_iters=max_iters,
                                                 base_lr=0.005, temperature_start=0.03,
                                                 temperature_end=0.002)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Phase 4: Final facet-aware polish
    if best_points is not None and time.time() < deadline - 15.0:
        points, radius = _facet_aware_optimize(best_points, deadline, max_iters=1000,
                                                 base_lr=0.002, temperature=0.015)
        if radius < best_radius:
            best_radius = radius
            best_points = points
    
    # Phase 5: Very fine polish with soft-max
    if best_points is not None and time.time() < deadline - 8.0:
        points, radius = _soft_max_optimize(best_points, deadline, max_iters=800,
                                              base_lr=0.001, temperature_start=0.008,
                                              temperature_end=0.001)
        if radius < best_radius:
            best_radius = radius
            best_points = points
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
