"""Stronger optimizer for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 138.0  # safety margin before 150s kill


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * math.fsum(np.log(d_sq).tolist())


def _energy_grad_fast(points):
    """Compute energy and gradient of logarithmic energy using vectorized ops.
    
    E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0)
    
    # inv_dist_sq[i,j] = 1/||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq_safe
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = -sum_j inv_dist_sq[i,j] * diff[i,j]
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy: sum over i<j of -log(dist_ij) = -0.5 * sum log(dist_sq_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * math.fsum(np.log(d_sq).tolist())
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.sqrt(np.sum(points * points, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _sphere_fun_grad(x):
    """Objective and gradient for scipy minimize in flat R^(3N) space.
    
    We use the ambient gradient from the sphere-embedded problem.
    The projection to sphere happens implicitly; we return the tangent-space
    gradient mapped back to ambient coordinates.
    """
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    # Project gradient to tangent space: g_tangent = g - (g·p)p
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _spherical_design_init(seed=42):
    """Generate initial points using a quasi-random spherical design."""
    rng = np.random.RandomState(seed)
    # Start with Fibonacci and add small random perturbations
    pts = _fibonacci_spiral()
    noise = rng.randn(N, 3) * 0.01
    pts = pts + noise
    return _project_to_sphere(pts)


def _icosahedral_init():
    """Generate icosahedral-based initial points (12 vertices expanded)."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = _project_to_sphere(verts)
    
    # Expand to N points by adding points near each vertex
    pts = []
    per_vertex = N // 12 + 1
    for v in verts:
        for k in range(per_vertex):
            if len(pts) >= N:
                break
            if k == 0:
                pts.append(v.copy())
            else:
                # Add small perturbation
                noise = np.random.randn(3) * 0.05
                p = v + noise
                p = p / np.linalg.norm(p)
                pts.append(p)
    
    # Pad if needed
    while len(pts) < N:
        pts.append(_random_points(1)[0].copy())
    
    return np.array(pts[:N])


def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _steepest_descent_sphere(points, max_iter=2000, tol=1e-10, time_budget=10.0):
    """Steepest descent on the sphere with backtracking line search."""
    start_time = time.time()
    p = points.copy()
    e, g = _energy_grad_fast(p)
    best_e = e
    best_p = p.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Project gradient to tangent space
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        
        # Check convergence
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        step_size = 0.1
        for _ in range(30):
            # Move in negative gradient direction and project back to sphere
            p_new = p - step_size * g_tan
            p_new = _project_to_sphere(p_new)
            
            e_new, g_new = _energy_grad_fast(p_new)
            if e_new < e - 1e-4 * step_size * grad_norm * grad_norm:
                break
            step_size *= 0.5
        
        else:
            # If no step accepted, try smaller
            step_size = 1e-6
            p_new = _project_to_sphere(p - step_size * g_tan)
            e_new, g_new = _energy_grad_fast(p_new)
        
        p = p_new
        e = e_new
        g = g_new
        
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    return best_p, best_e


def _riemannian_flow(points, max_iter=5000, tol=1e-12, time_budget=15.0):
    """Riemannian gradient flow with adaptive step size on the sphere."""
    start_time = time.time()
    p = points.copy()
    e, g = _energy_grad_fast(p)
    best_e = e
    best_p = p.copy()
    
    # Adaptive step size based on gradient magnitude
    step_size = 0.05
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Project gradient to tangent space
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        grad_norm = np.linalg.norm(g_tan)
        
        if grad_norm < tol:
            break
        
        # Adaptive step size: smaller when gradient is large
        target_step = min(0.1, 0.5 / (grad_norm + 1e-8))
        step_size = 0.8 * step_size + 0.2 * target_step
        
        # Try a few step sizes and pick the best
        best_e_new = e
        best_p_new = p
        best_g_new = g
        
        for alpha in [step_size, step_size * 0.5, step_size * 0.25, step_size * 2.0]:
            p_new = p - alpha * g_tan
            p_new = _project_to_sphere(p_new)
            e_new, g_new = _energy_grad_fast(p_new)
            if e_new < best_e_new:
                best_e_new = e_new
                best_p_new = p_new
                best_g_new = g_new
        
        if best_e_new >= e:
            # No improvement, reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                break
            continue
        
        p = best_p_new
        e = best_e_new
        g = best_g_new
        
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    return best_p, best_e


def _lbfgs_optimize(points, max_iter=5000, time_budget=20.0):
    """L-BFGS-B optimization from given initial points."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            new_energy = _energy(new_points)
            return new_points, new_energy
    except Exception:
        pass
    
    return points, _energy(points)


def _simulated_annealing_refine(points, time_budget=15.0):
    """Simulated annealing refinement for logarithmic energy with adaptive moves."""
    start_time = time.time()
    p = points.copy()
    e = _energy(p)
    best_p = p.copy()
    best_e = e
    
    # Initial temperature
    T = 0.3
    T_min = 1e-7
    cooling = 0.999
    
    # Track move sizes
    move_scale = 0.05
    
    # Precompute for fast energy updates
    # We'll use full recomputation for simplicity but with early termination
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random point to move
        idx = np.random.randint(N)
        original = p[idx].copy()
        
        # Adaptive perturbation: mix of small and larger moves
        if np.random.random() < 0.1:
            # Occasionally make a larger move to escape local minima
            delta = np.random.randn(3) * (move_scale * 3)
        else:
            delta = np.random.randn(3) * move_scale
        
        p[idx] = original + delta
        norm = np.linalg.norm(p[idx])
        if norm < 1e-10:
            p[idx] = original
            continue
        p[idx] = p[idx] / norm
        
        e_new = _energy(p)
        delta_e = e_new - e
        
        if delta_e < 0 or (T > 0 and np.random.random() < math.exp(-delta_e / T)):
            e = e_new
            if e < best_e:
                best_e = e
                best_p = p.copy()
            # Increase move scale on success
            move_scale = min(0.3, move_scale * 1.02)
        else:
            p[idx] = original
            # Decrease move scale on rejection
            move_scale = max(0.005, move_scale * 0.98)
        
        T *= cooling
    
    return best_p, best_e


def construct():
    start_time = time.time()
    
    # Start with multiple initializations and pick the best
    inits = [
        _fibonacci_spiral(),
        _spherical_design_init(42),
        _spherical_design_init(123),
        _icosahedral_init(),
    ]
    
    best_points = None
    best_energy = float('inf')
    
    for init_pts in inits:
        e = _energy(init_pts)
        if e < best_energy:
            best_energy = e
            best_points = init_pts.copy()
    
    # Phase 1: Riemannian flow from best initial point (more robust than L-BFGS)
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.2:
        t_budget = min(TIME_LIMIT * 0.2 - elapsed, 20.0)
        pts, e = _riemannian_flow(best_points, max_iter=10000, tol=1e-12, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 2: L-BFGS refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.35:
        t_budget = min(TIME_LIMIT * 0.35 - elapsed, 15.0)
        pts, e = _lbfgs_optimize(best_points, max_iter=5000, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 3: Steepest descent refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.45:
        t_budget = min(TIME_LIMIT * 0.45 - elapsed, 10.0)
        pts, e = _steepest_descent_sphere(best_points, max_iter=5000, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 4: Multiple random restarts with Riemannian flow
    np.random.seed(12345)
    elapsed = time.time() - start_time
    restart_count = 0
    while elapsed < TIME_LIMIT * 0.7:
        restart_count += 1
        # Use structured perturbations of current best
        if restart_count % 3 == 0:
            # Perturb current best
            noise = np.random.randn(N, 3) * 0.05
            random_pts = _project_to_sphere(best_points + noise)
        elif restart_count % 3 == 1:
            # Moderate perturbation
            noise = np.random.randn(N, 3) * 0.15
            random_pts = _project_to_sphere(best_points + noise)
        else:
            # Pure random
            random_pts = _random_points(seed=1000 + restart_count)
        
        # Quick Riemannian flow from random start
        t_budget = min(TIME_LIMIT * 0.7 - elapsed, 8.0)
        pts, e = _riemannian_flow(random_pts, max_iter=5000, tol=1e-10, time_budget=t_budget)
        
        if e < best_energy:
            best_energy = e
            best_points = pts
            # Refine with L-BFGS
            elapsed = time.time() - start_time
            if elapsed < TIME_LIMIT * 0.7:
                t_budget2 = min(TIME_LIMIT * 0.7 - elapsed, 5.0)
                pts2, e2 = _lbfgs_optimize(best_points, max_iter=3000, time_budget=t_budget2)
                if e2 < best_energy:
                    best_energy = e2
                    best_points = pts2
        
        elapsed = time.time() - start_time
    
    # Phase 5: Simulated annealing refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.85:
        t_budget = min(TIME_LIMIT * 0.85 - elapsed, 10.0)
        pts, e = _simulated_annealing_refine(best_points, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 6: Final Riemannian flow polish with tight tolerance
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.93:
        t_budget = min(TIME_LIMIT * 0.93 - elapsed, 5.0)
        pts, e = _riemannian_flow(best_points, max_iter=10000, tol=1e-13, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 7: Final steepest descent polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.97:
        t_budget = min(TIME_LIMIT * 0.97 - elapsed, 2.0)
        pts, e = _steepest_descent_sphere(best_points, max_iter=5000, tol=1e-13, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d_sq = np.sum((best_points[i] - best_points[j]) ** 2)
            if d_sq < 1e-18:  # d < 1e-9
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng.randn(3) * 1e-6
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    return best_points
