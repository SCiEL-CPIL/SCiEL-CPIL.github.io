"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient.
    Energy E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient w.r.t x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    # Pairwise differences
    # diff[i, j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Check for proximity
    if np.any(dist_sq[mask] < 1e-18):
        return float('inf'), np.zeros_like(pts)
        
    dists = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(d_ij)
    # Since dists is symmetric, sum over i<j is 0.5 * sum_{i!=j}
    # We only need upper triangle for sum
    i, j = np.triu_indices(n, k=1)
    d_upper = dists[i, j]
    
    if np.any(d_upper < 1e-9):
        return float('inf'), np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    # Gradient calculation
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: log(1/d) = -log(d). d = ||x_i - x_j||.
    # d/dx_i (-log(||x_i - x_j||)) = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
    # Wait, let's re-derive.
    # f(x) = -log(||x_i - x_j||)
    # grad_xi f = -1/||x_i - x_j|| * (x_i - x_j)/||x_i - x_j|| = -(x_i - x_j)/||x_i - x_j||^2
    # So grad_i = sum_{j != i} -(x_i - x_j) / ||x_i - x_j||^2
    #            = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad = sum_j (x_j - x_i) / d^2
    # (x_j - x_i) is -diff[i,j]
    # term_ij = -diff[i,j] * inv_dist_sq[i,j]
    # grad_i = sum_j term_ij
    
    # diff[i,j] = x_i - x_j
    # We want sum_j (x_j - x_i) / d^2 = - sum_j diff[i,j] / d^2
    # Let's compute matrix G where G[i,j] = diff[i,j] * inv_dist_sq[i,j]
    # Then grad_i = - sum_j G[i,j]
    
    G = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(G, axis=1)
    
    return energy, grad

def _local_minimize(start_pts, deadline):
    """
    Run L-BFGS-B optimization on the sphere using a retraction or penalty method.
    Since L-BFGS-B works in Euclidean space, we can use a penalty method or
    project back to the sphere after each step. 
    A more robust way for spheres is to use a parameterization or just 
    minimize in R^3 and project. However, simple projection can break gradient consistency.
    
    Alternative: Use `scipy.optimize.minimize` with method='L-BFGS-B' on the 
    unconstrained space, but add a large penalty for ||x||^2 - 1.
    Or, better, use the fact that for small steps, we can move on the tangent space.
    
    Given time constraints and simplicity, let's use a standard trick:
    Minimize the energy in R^3, but at each iteration, project the points to the sphere.
    This is not strictly gradient descent on the manifold, but often works well for 
    energy minimization on spheres if the initial points are close to the sphere.
    
    However, a cleaner approach for SciPy is to use `minimize` with bounds or 
    just trust the projection if the energy function is smooth.
    
    Let's try a different approach: Randomize the starting points significantly, 
    run a few steps of gradient descent with projection, then use L-BFGS-B 
    on the projected points.
    
    Actually, the most reliable "different" strategy is Basin Hopping.
    But Basin Hopping requires many local minimizations.
    
    Let's implement a robust local minimizer using L-BFGS-B with a penalty term 
    for the sphere constraint.
    """
    def penalty_energy_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        # Project to sphere for energy calculation? 
        # No, L-BFGS-B works on the vector x. 
        # If we just minimize E(pts) in R^3, the points will fly off to infinity or collapse.
        # We need to constrain ||x_i|| = 1.
        
        # Let's use a penalty method:
        # E_pen = E(pts) + lambda * sum(||x_i||^2 - 1)^2
        energy, grad = _energy_and_grad(pts)
        if not np.isfinite(energy):
            return 1e10, np.zeros_like(x_flat)
            
        norms_sq = np.sum(pts * pts, axis=1)
        penalty = 1e6 * np.sum((norms_sq - 1.0)**2)
        
        # Gradient of penalty:
        # d/dx_i (||x_i||^2 - 1)^2 = 2(||x_i||^2 - 1) * 2 x_i = 4 x_i (||x_i||^2 - 1)
        pen_grad = 1e6 * 4.0 * pts * (norms_sq - 1.0)[:, np.newaxis]
        
        total_grad = grad.flatten() + pen_grad.flatten()
        
        return energy + penalty, total_grad

    # We need a good starting point.
    # If start_pts is already on the sphere, we are good.
    # We will run L-BFGS-B for a limited number of iterations.
    
    # To save time, we check deadline inside the callback? 
    # SciPy minimize doesn't easily support early stopping based on time in the callback 
    # without breaking the optimization. 
    # Instead, we will limit the number of function evaluations (maxfun).
    
    # Estimate time per function eval: ~0.1 ms? 
    # 100 points, 4950 pairs. 
    # Let's allow 200 function evaluations per restart.
    
    try:
        res = minimize(
            penalty_energy_and_grad, 
            start_pts.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if res.success:
            opt_pts = res.x.reshape(N, 3)
            # Normalize to ensure they are on the sphere
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            opt_pts = opt_pts / norms
            return opt_pts
        else:
            return start_pts
    except Exception:
        return start_pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # Strategy: Parallel (sequential) Basin Hopping / Multi-start Local Optimization
    # 1. Generate many diverse random starts.
    # 2. For each start, run a local minimizer (L-BFGS-B with penalty).
    # 3. Keep the best result.
    
    best_pts = None
    best_energy = float('inf')
    
    # Fibonacci spiral as a good initial guess (often close to optimal)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # First, try optimizing the Fibonacci spiral
    opt_fib = _local_minimize(fib_pts, deadline)
    e_fib, _ = _energy_and_grad(opt_fib)
    if e_fib < best_energy:
        best_energy = e_fib
        best_pts = opt_fib.copy()
        
    if time.time() > deadline:
        return best_pts

    # Now, try random restarts
    # How many restarts can we afford?
    # Each restart takes ~0.1-0.5 seconds depending on convergence.
    # Let's try 50 restarts.
    
    rng = np.random.default_rng(42)
    
    for k in range(50):
        if time.time() > deadline:
            break
            
        # Generate random points on the sphere
        # Uniform distribution
        pts = rng.normal(size=(N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        # To make it a "Basin Hopping" style, we could perturb the best solution,
        # but pure random restarts are also a valid "different" strategy from SA.
        # Let's mix: 50% random, 50% perturbation of best.
        
        if k % 2 == 0 and best_pts is not None:
            # Perturb best solution
            # Add small noise
            noise = rng.normal(scale=0.5, size=(N, 3))
            pts = best_pts + noise
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            pts = pts / norms
            
        opt_pts = _local_minimize(pts, deadline)
        
        e_opt, _ = _energy_and_grad(opt_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = opt_pts.copy()
            
    # Final check
    if best_pts is None:
        # Fallback to Fibonacci if all optimizations failed
        best_pts = fib_pts
    else:
        # Ensure final normalization
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_pts = best_pts / norms
        
    return best_pts
# EVOLVE-BLOCK-END
