import numpy as np
import time
import math

N = 150

def _energy_and_grad(points):
    # points: (N, 3)
    # dist: (N, N)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero for i=j
    dist_sq[np.diag_indices_from(dist_sq)] = 1.0
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    # We want to minimize E, so we maximize sum log(dist)
    # Let's compute the gradient of E directly.
    # grad_i E = sum_{j!=i} (x_j - x_i) / ||x_i - x_j||^2
    
    # diff is x_i - x_j. We need x_j - x_i which is -diff.
    # term = (x_j - x_i) / dist_sq = -diff / dist_sq
    # grad_i = sum_{j!=i} -diff[i,j] / dist_sq[i,j]
    
    # Create a mask to exclude i=j
    mask = ~np.eye(N, dtype=bool)
    
    # Compute gradient
    # diff: (N, N, 3)
    # dist_sq: (N, N)
    # We want to sum over j for each i.
    # grad[i] = sum_j ( -diff[i,j] / dist_sq[i,j] )
    
    # diff / dist_sq[..., np.newaxis]
    inv_dist_sq = 1.0 / dist_sq
    inv_dist_sq[~mask] = 0.0 # Zero out diagonal
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy
    # E = - sum_{i<j} log(dist)
    # sum_{i<j} log(dist) = 0.5 * (sum_{i!=j} log(dist) - sum_{i=j} log(1))
    # log(dist) matrix:
    log_dist = np.log(dist)
    log_dist[~mask] = 0.0
    total_log_dist = np.sum(log_dist) / 2.0
    energy = -total_log_dist
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * points, axis=1)
    grad -= dot_prod[:, np.newaxis] * points
    
    return energy, grad

def _exp_map(x, v):
    # x: (N, 3)
    # v: (N, 3)
    # exp_x(v) = x cos(||v||) + (v/||v||) sin(||v||)
    v_norm = np.linalg.norm(v, axis=1)
    # Avoid division by zero
    v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1.0)
    cos_v = np.cos(v_norm_safe)
    sin_v = np.sin(v_norm_safe)
    
    # Handle v=0 case
    # If v_norm is 0, exp_x(0) = x
    # x * cos(0) + 0 * sin(0) / 0 -> x * 1 + 0 = x. Correct.
    # But v/v_norm is 0/0.
    v_scaled = v / v_norm_safe[:, np.newaxis]
    v_scaled[v_norm < 1e-12] = 0.0
    
    return x * cos_v[:, np.newaxis] + v_scaled * sin_v[:, np.newaxis]

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    return points / norms_safe[:, np.newaxis]

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _lbgfs_sphere(pts0, max_iter=2000, m=10, time_limit=None, start_time=None, tol=1e-12):
    """L-BFGS optimization on the sphere with projected gradients."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # L-BFGS memory
    s_list = []
    y_list = []
    
    step_size = 0.5
    min_step = 1e-14
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # L-BFGS two-loop recursion
        if len(s_list) == 0:
            direction = -grad
        else:
            q = grad.copy()
            rhos = []
            alphas = []
            
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                if abs(si_yi) < 1e-16:
                    rho = 0.0
                else:
                    rho = 1.0 / si_yi
                rhos.append(rho)
                ai = np.sum(si * q) * rho
                alphas.append(ai)
                q -= ai * yi
            
            if len(s_list) > 0:
                last_s = s_list[-1]
                last_y = y_list[-1]
                sy = np.sum(last_s * last_y)
                if abs(sy) > 1e-16:
                    gamma = np.sum(last_s * last_s) / sy
                    if gamma < 1e-10 or gamma > 1e10:
                        gamma = 1.0
                else:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = q * gamma
            
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Line search with backtracking
        alpha = step_size
        accepted = False
        for _ in range(50):
            new_pts = _project_to_sphere(pts + alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * direction):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            new_pts = _project_to_sphere(pts + alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not accepted:
            break
        
        # Update L-BFGS memory
        s = alpha * direction
        y = new_grad - grad
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)
    
    return pts, energy

def _cg_sphere(pts0, max_iter=2000, time_limit=None, start_time=None, tol=1e-12):
    """Conjugate gradient optimization on the sphere using exponential maps."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # CG state
    p = -grad  # initial search direction
    p = p - np.sum(p * pts, axis=1, keepdims=True) * pts  # project to tangent
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search along p using exponential map
        alpha = 0.5
        accepted = False
        for _ in range(50):
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * p):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try smaller step
            alpha = 0.01
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                accepted = True
            else:
                break
        
        if not accepted:
            break
        
        # Update CG direction
        g_old = grad
        grad = new_grad
        pts = new_pts
        energy = new_energy
        
        # Fletcher-Reeves restart
        beta = np.sum(grad * grad) / (np.sum(g_old * g_old) + 1e-16)
        p_new = -grad + beta * p
        # Project to tangent space
        p_new = p_new - np.sum(p_new * pts, axis=1, keepdims=True) * pts
        
        # Check for restart condition
        if np.sum(grad * p) > 0:
            p = -grad
            p = p - np.sum(p * pts, axis=1, keepdims=True) * pts
        else:
            p = p_new
    
    return pts, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_energy = float('inf')
    best_points = None
    
    # Initialization 1: Fibonacci
    points = _fibonacci_sphere(N)
    
    # Run L-BFGS first
    pts_lbfgs, energy_lbfgs = _lbgfs_sphere(points, max_iter=1000, time_limit=deadline - start_time, start_time=start_time)
    if energy_lbfgs < best_energy:
        best_energy = energy_lbfgs
        best_points = pts_lbfgs.copy()
    
    # If time permits, run CG from the L-BFGS result
    if time.time() < deadline - 10.0:
        pts_cg, energy_cg = _cg_sphere(pts_lbfgs, max_iter=1000, time_limit=deadline - start_time, start_time=start_time)
        if energy_cg < best_energy:
            best_energy = energy_cg
            best_points = pts_cg.copy()
    
    # Try a second initialization with random points
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(42)
        points2 = rng.normal(size=(N, 3))
        points2 = _project_to_sphere(points2)
        
        pts_lbfgs2, energy_lbfgs2 = _lbgfs_sphere(points2, max_iter=1000, time_limit=deadline - start_time, start_time=start_time)
        if energy_lbfgs2 < best_energy:
            best_energy = energy_lbfgs2
            best_points = pts_lbfgs2.copy()
        
        # If time permits, run CG from the second L-BFGS result
        if time.time() < deadline - 10.0:
            pts_cg2, energy_cg2 = _cg_sphere(pts_lbfgs2, max_iter=1000, time_limit=deadline - start_time, start_time=start_time)
            if energy_cg2 < best_energy:
                best_energy = energy_cg2
                best_points = pts_cg2.copy()
    
    # Final polish: run a few more L-BFGS iterations with tighter tolerance
    if best_points is not None and time.time() < deadline - 5.0:
        pts_polish, energy_polish = _lbgfs_sphere(best_points, max_iter=500, time_limit=deadline - start_time, start_time=start_time, tol=1e-14)
        if energy_polish < best_energy:
            best_energy = energy_polish
            best_points = pts_polish.copy()
    
    if best_points is None:
        return points  # Fallback to initial
    
    return best_points
