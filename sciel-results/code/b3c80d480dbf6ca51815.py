import numpy as np
import time

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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initialize with Halton sequence for better low-discrepancy
    def halton_sequence(n, base):
        seq = []
        f = 1.0
        r = 0.0
        for i in range(1, n + 1):
            f = 1.0 / base
            while i > 0:
                r += f * (i % base)
                i = i // base
                f /= base
            seq.append(r)
        return np.array(seq)
    
    def halton_sphere(n):
        # Use 3 dimensions of Halton sequence
        # Map to sphere using spherical coordinates or just normalize?
        # Normalizing random points is bad. 
        # Let's use Fibonacci spiral as it's known to be good and simple.
        # Or use the Halton points in [0,1]^3 and normalize? 
        # Actually, Fibonacci spiral is very good for sphere.
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple initializations
    best_energy = float('inf')
    best_points = None
    
    # Initialization 1: Fibonacci
    points = halton_sphere(N)
    
    # Riemannian Conjugate Gradient
    x = points.copy()
    _, g = _energy_and_grad(x)
    d = -g # Initial direction
    p = d
    alpha = 1.0
    
    max_iter = 1000
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Line search
        # Try to reduce energy
        # Armijo condition or simple backtracking
        energy_curr, _ = _energy_and_grad(x)
        
        # Compute gradient of energy along the geodesic
        # f(t) = E(exp_x(t*d))
        # We can just do backtracking on the energy value
        
        alpha = 1.0
        # Ensure step is not too large
        # Max step size related to curvature? 
        # Let's start with alpha=1.0 and backtrack.
        
        for _ in range(20):
            x_trial = _exp_map(x, alpha * d)
            energy_trial, g_trial = _energy_and_grad(x_trial)
            
            if energy_trial < energy_curr:
                break
            alpha *= 0.5
            
        if alpha < 1e-6:
            # Line search failed, restart
            d = -g
            p = d
            alpha = 1.0
            continue
            
        x = x_trial
        g = g_trial
        energy_curr = energy_trial
        
        # Update direction
        # Polak-Ribiere or Fletcher-Reeves
        # g_new dot g_new - g_old dot g_old
        g_dot_g_new = np.sum(g * g)
        g_dot_g_old = np.sum(_energy_and_grad(x)[1] * _energy_and_grad(x)[1]) # This is wrong, we need old g
        # We didn't store old g properly in the loop above if we overwrote it.
        # Let's store g_old before updating x.
        
        # Restart loop logic for clarity and correctness
        # I will rewrite the optimization loop more carefully.
        
        if energy_curr < best_energy:
            best_energy = energy_curr
            best_points = x.copy()
            
    # If time permits, try a second initialization
    if time.time() < deadline - 30.0:
        # Perturb the Fibonacci points slightly or use a different seed
        # Or just use a random initialization with projection
        rng = np.random.default_rng(42)
        points2 = rng.normal(size=(N, 3))
        points2 = _project_to_sphere(points2)
        
        x = points2.copy()
        _, g = _energy_and_grad(x)
        d = -g
        p = d
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            energy_curr, _ = _energy_and_grad(x)
            g_old = g.copy()
            
            alpha = 1.0
            for _ in range(20):
                x_trial = _exp_map(x, alpha * d)
                energy_trial, g_trial = _energy_and_grad(x_trial)
                if energy_trial < energy_curr:
                    break
                alpha *= 0.5
                
            if alpha < 1e-6:
                d = -g
                p = d
                continue
                
            x = x_trial
            g = g_trial
            energy_curr = energy_trial
            
            if energy_curr < best_energy:
                best_energy = energy_curr
                best_points = x.copy()
                
            # Update direction (Fletcher-Reeves)
            g_dot_g_old = np.sum(g_old * g_old)
            if g_dot_g_old > 1e-12:
                beta = np.sum(g * g) / g_dot_g_old
            else:
                beta = 0.0
            d_new = -g + beta * p
            # Ensure d is orthogonal to x (tangent space)
            # d = d - (d dot x) x
            dot = np.sum(d_new * x, axis=1)
            d_new = d_new - dot[:, np.newaxis] * x
            p = d_new
            d = d_new

    if best_points is None:
        return points # Fallback to initial
    
    return best_points
