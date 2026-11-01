import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 145.0  # Hard wall-clock deadline with 5s safety margin


def _time_left():
    return max(0.0, DEADLINE - time.time())


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


def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-7,
                     lr_decay=0.999, patience=2000):
    """Riemannian gradient descent with backtracking line search and adaptive learning rate.
    Operates directly on the sphere manifold.
    """
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        if _time_left() < 5.0:
            break
        
        energy, grad = _energy_grad_vec(best_points)
        
        # Check if gradient is small enough
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            # Riemannian step: move in direction -grad, then re-project
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            
            new_energy, _ = _energy_grad_vec(new_points)
            
            if new_energy < energy - 1e-10:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
            # Slightly increase learning rate if improving well
            if alpha > lr:
                lr = min(alpha * 1.1, 0.5)
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
        
        # Decay learning rate slowly
        lr = max(lr * lr_decay, min_lr)
    
    return best_points


def _polish_lbfgs(points, max_iter=5000):
    """High-precision polish using L-BFGS-B with tight tolerances.
    Uses the ambient-space formulation with projection."""
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


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


def _sobol_points(n):
    """Generate points using quasi-random (Sobol-like) distribution."""
    # Use a simple deterministic low-discrepancy sequence
    rng = np.random.RandomState(12345)
    # Generate using Halton sequence approximation
    points = np.zeros((n, 3))
    for i in range(n):
        # Use golden ratio conjugate for 1D, extend to 3D
        t1 = (i + 0.5) / n
        t2 = ((i + 0.5) * 0.618033988749895) % 1.0
        t3 = ((i + 0.5) * 0.381966011250105) % 1.0
        
        # Map to sphere using spherical coordinates
        theta = 2 * math.pi * t1
        cos_phi = 1 - 2 * t2
        sin_phi = math.sqrt(1 - cos_phi**2)
        # Add a third component for better distribution
        angle = 2 * math.pi * t3
        
        x = sin_phi * math.cos(angle)
        y = sin_phi * math.sin(angle)
        z = cos_phi
        
        points[i] = [x, y, z]
    
    return _project(points)


def _spectral_smooth(points, num_steps=50, alpha=0.5):
    """Apply spectral Laplacian smoothing using Delaunay neighbors.
    This acts as a low-pass filter on the point distribution,
    enforcing global geometric consistency.
    """
    from scipy.spatial import Delaunay, cKDTree
    
    try:
        # Create Delaunay triangulation (project to 2D for simplicity)
        # Use a simple neighbor-based smoothing instead
        tree = cKDTree(points)
        
        for step in range(num_steps):
            if _time_left() < 5.0:
                break
            
            new_points = points.copy()
            for i in range(N):
                # Find k nearest neighbors
                distances, indices = tree.query(points[i], k=10)
                # Exclude self
                mask = indices != i
                neighbors = indices[mask]
                
                if len(neighbors) == 0:
                    continue
                
                # Compute weighted average of neighbors
                avg_neighbor = np.mean(points[neighbors], axis=0)
                
                # Smooth: move point toward average of neighbors
                new_points[i] = (1 - alpha) * points[i] + alpha * avg_neighbor
            
            # Project back to sphere
            new_points = _project(new_points)
            
            # Check if energy improved
            e_old, _ = _energy_grad_vec(points)
            e_new, _ = _energy_grad_vec(new_points)
            
            if e_new < e_old:
                points = new_points
            else:
                # Reduce smoothing strength
                alpha *= 0.8
                if alpha < 0.01:
                    break
    
    except Exception:
        pass
    
    return points


def construct():
    global DEADLINE
    DEADLINE = time.time() + 145.0
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Riemannian flow from multiple initial configurations
    initial_configs = [
        ("fibonacci", _fibonacci_spiral(N)),
        ("random_42", _random_points(N, 42)),
        ("sobol", _sobol_points(N)),
    ]
    
    for name, points in initial_configs:
        if _time_left() < 10.0:
            break
        
        budget = min(_time_left() * 0.2, 15.0)
        if budget < 2.0:
            break
        
        refined = _riemannian_flow(points, budget, initial_lr=0.05)
        e, _ = _energy_grad_vec(refined)
        
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 2: Spectral smoothing to escape local minima
    if best_points is not None and _time_left() > 10.0:
        smoothed = _spectral_smooth(best_points, num_steps=100, alpha=0.3)
        e, _ = _energy_grad_vec(smoothed)
        if e < best_energy:
            best_energy = e
            best_points = smoothed
        
        # Re-optimize after smoothing
        if _time_left() > 8.0:
            budget = min(_time_left() * 0.15, 10.0)
            if budget > 1.0:
                refined = _riemannian_flow(smoothed, budget, initial_lr=0.02)
                e, _ = _energy_grad_vec(refined)
                if e < best_energy:
                    best_energy = e
                    best_points = refined
    
    # Phase 3: Multiple random restarts with Riemannian flow
    restart_start = time.time()
    restart_time_limit = _time_left() * 0.3
    
    for seed in range(1, 200):
        if time.time() - restart_start > restart_time_limit:
            break
        if _time_left() < 8.0:
            break
        
        random_points = _random_points(N, seed)
        
        # Quick Riemannian flow with limited budget
        budget_per = min(2.0, _time_left() * 0.05)
        if budget_per < 0.5:
            break
        
        refined = _riemannian_flow(random_points, budget_per, initial_lr=0.1, patience=500)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 4: Polish the best with L-BFGS-B
    if best_points is not None and _time_left() > 5.0:
        polish_budget = min(_time_left() * 0.5, 25.0)
        if polish_budget > 3.0:
            polished = _polish_lbfgs(best_points, max_iter=3000)
            e, _ = _energy_grad_vec(polished)
            if e < best_energy:
                best_energy = e
                best_points = polished
    
    # Phase 5: Final high-precision polish
    if best_points is not None and _time_left() > 3.0:
        final_polish_budget = min(_time_left() * 0.3, 10.0)
        if final_polish_budget > 2.0:
            final = _polish_lbfgs(best_points, max_iter=5000)
            e, _ = _energy_grad_vec(final)
            if e < best_energy:
                best_energy = e
                best_points = final
    
    # Final validation: ensure no points are too close
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    best_points = _project(best_points)
    
    # Check for points that are too close and perturb them
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                noise = np.random.randn(3) * 1e-5
                best_points[i] = _project(best_points[i] + noise)
    
    return best_points
