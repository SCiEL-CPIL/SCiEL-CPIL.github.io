import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())

def _energy_grad_vec(points):
    """Vectorized energy (np.sum, fast) and gradient on the sphere.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy (fast, for comparison during optimization)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    # Project gradient to tangent space
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    
    return energy, g_tangent

def _project(points):
    """Project to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-8,
                     patience=3000):
    """Riemannian gradient descent with backtracking line search.
    Two-phase: fast descent then slow refinement."""
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0
    
    # Phase 1: Fast descent (first 70% of budget)
    fast_budget = time_budget * 0.7
    while time.time() - start_time < fast_budget:
        energy, grad = _energy_grad_vec(best_points)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        alpha = lr
        improved = False
        for _ in range(20):
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            new_energy, _ = _energy_grad_vec(new_points)
            
            if new_energy < energy - 1e-12:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
            # Decay learning rate slowly during fast phase
            lr *= 0.9995
        else:
            no_improve += 1
            lr *= 0.85
            if no_improve > patience:
                break
    
    # Phase 2: Slow refinement (remaining 30% of budget)
    lr = max(lr, 1e-5)
    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_vec(best_points)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        alpha = lr
        improved = False
        for _ in range(25):
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            new_energy, _ = _energy_grad_vec(new_points)
            
            if new_energy < energy - 1e-14:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
            lr *= 0.999
        else:
            no_improve += 1
            lr *= 0.8
            if no_improve > patience:
                break
    
    return best_points

def _polish_l_bfgs(points, max_iter=20000, ftol=1e-16, gtol=1e-14):
    """L-BFGS-B polish with very tight tolerances."""
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': ftol, 
                                'gtol': gtol, 'maxls': 100, 'iprint': -1})
        if hasattr(res, 'x') and res.x is not None:
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points

def _polish_newton(points, max_iter=100):
    """Newton-like refinement using Hessian approximation via finite differences.
    For small steps, this can achieve higher precision than L-BFGS-B."""
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    
    for iteration in range(max_iter):
        energy, grad = _energy_grad_vec(best_points)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Approximate Hessian using finite differences
        h = 1e-6
        hess_approx = np.zeros_like(grad)
        for i in range(N):
            for j in range(3):
                perturbed = best_points.copy()
                perturbed[i, j] += h
                perturbed = _project(perturbed)
                _, grad_plus = _energy_grad_vec(perturbed)
                
                perturbed = best_points.copy()
                perturbed[i, j] -= h
                perturbed = _project(perturbed)
                _, grad_minus = _energy_grad_vec(perturbed)
                
                hess_approx[i, j] = (grad_plus[i, j] - grad_minus[i, j]) / (2 * h)
        
        # Solve for Newton step in tangent space
        # Use simple diagonal approximation for stability
        diag = np.abs(hess_approx) + 1e-8
        newton_step = grad / diag
        
        # Scale step to ensure it's not too large
        step_norm = np.linalg.norm(newton_step)
        if step_norm > 1e-3:
            newton_step *= 1e-3 / step_norm
        
        new_points = best_points - newton_step
        new_points = _project(new_points)
        new_energy, _ = _energy_grad_vec(new_points)
        
        if new_energy < best_energy - 1e-15:
            best_points = new_points
            best_energy = new_energy
        else:
            # Try smaller step
            for scale in [0.1, 0.01, 0.001]:
                new_points = best_points - scale * newton_step
                new_points = _project(new_points)
                new_energy, _ = _energy_grad_vec(new_points)
                if new_energy < best_energy - 1e-15:
                    best_points = new_points
                    best_energy = new_energy
                    break
    
    return best_points

def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)

def _random_points(n, seed):
    """Generate random points on sphere with given seed."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)

def _perturb_points(points, scale=0.01, seed=None):
    """Create perturbed versions of points for escape from local minima."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    perturbed = points.copy()
    for i in range(len(perturbed)):
        noise = rng.randn(3) * scale
        perturbed[i] = _project(perturbed[i] + noise)
    return perturbed

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _time_left():
        return max(0.0, deadline - time.time())
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Riemannian flow from Fibonacci spiral (most promising start)
    points_fib = _fibonacci_spiral(N)
    if _time_left() > 10.0:
        budget1 = min(_time_left() * 0.2, 15.0)
        refined = _riemannian_flow(points_fib, budget1)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 2: Multiple random restarts with short budgets
    restart_start = time.time()
    max_restart_time = _time_left() * 0.4
    
    for seed in range(1, 500):
        if time.time() - restart_start > max_restart_time:
            break
        if _time_left() < 15.0:
            break
        
        random_points = _random_points(N, seed)
        
        # Quick Riemannian flow with limited budget
        budget_per = min(2.0, _time_left() * 0.05)
        if budget_per < 0.3:
            break
        
        refined = _riemannian_flow(random_points, budget_per, initial_lr=0.1, patience=1000)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 3: Try perturbations of the best solution
    if best_points is not None and _time_left() > 10.0:
        for seed in range(1, 50):
            if _time_left() < 10.0:
                break
            
            perturbed = _perturb_points(best_points, scale=0.005, seed=seed)
            budget_per = min(1.5, _time_left() * 0.03)
            if budget_per < 0.2:
                break
            
            refined = _riemannian_flow(perturbed, budget_per, initial_lr=0.05, patience=500)
            e, _ = _energy_grad_vec(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Phase 4: Multi-stage polishing
    if best_points is not None and _time_left() > 5.0:
        # Stage 4a: L-BFGS-B with moderate tolerance
        if _time_left() > 10.0:
            polished = _polish_l_bfgs(best_points, max_iter=5000, ftol=1e-14, gtol=1e-12)
            e, _ = _energy_grad_vec(polished)
            if e < best_energy:
                best_energy = e
                best_points = polished
        
        # Stage 4b: L-BFGS-B with tight tolerance
        if _time_left() > 5.0:
            polished = _polish_l_bfgs(best_points, max_iter=10000, ftol=1e-16, gtol=1e-14)
            e, _ = _energy_grad_vec(polished)
            if e < best_energy:
                best_energy = e
                best_points = polished
        
        # Stage 4c: Newton-like refinement for final precision
        if _time_left() > 3.0:
            polished = _polish_newton(best_points, max_iter=50)
            e, _ = _energy_grad_vec(polished)
            if e < best_energy:
                best_energy = e
                best_points = polished
    
    # Final validation: ensure no points are too close
    if best_points is not None:
        best_points = _project(best_points)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < 1e-9:
                    noise = np.random.randn(3) * 1e-5
                    best_points[i] = _project(best_points[i] + noise)
        
        return best_points
    
    # Fallback
    return _fibonacci_spiral(N)
