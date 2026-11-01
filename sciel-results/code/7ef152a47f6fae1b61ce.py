"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_energy(x):
    """
    Compute logarithmic energy efficiently.
    E = sum_{i<j} log(1/||x_i - x_j||)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    return energy

def _compute_grad(x):
    """
    Compute gradient of logarithmic energy.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    grad = np.sum(weighted_diff, axis=1)
    
    return grad

def _exp_map(x, p, v):
    """
    Exponential map on the sphere.
    x: current position (N, 3)
    p: point index
    v: tangent vector (3,)
    Returns new point at x[p] + v
    """
    xp = x[p]
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return xp
    v_hat = v / theta
    # Rodrigues rotation formula
    # v is tangent to xp, so xp . v = 0
    # new_p = xp * cos(theta) + v_hat * sin(theta)
    new_p = xp * np.cos(theta) + v_hat * np.sin(theta)
    # Normalize for numerical stability
    new_p /= np.linalg.norm(new_p)
    return new_p

def _riemannian_lbfgs(x, max_iter=200, time_limit=15.0):
    """
    Riemannian L-BFGS method with 2-point memory.
    """
    start_time = time.time()
    m = 10  # L-BFGS memory
    
    x_new = x.copy()
    f_new = _compute_energy(x_new)
    g_new = _compute_grad(x_new)
    g_new = g_new - np.sum(g_new * x_new, axis=1, keepdims=True) * x_new
    
    # Store history
    s_hist = []  # steps
    y_hist = []  # gradient differences
    f_hist = [f_new]
    g_hist = [g_new]
    
    # Initial direction: negative gradient
    d = -g_new
    d = d - np.sum(d * x_new, axis=1, keepdims=True) * x_new
    
    for k in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        
        # Line search with backtracking
        alpha = 1.0
        df0 = np.sum(g_new * d)
        
        if df0 >= 0:
            # Restart
            d = -g_new
            d = d - np.sum(d * x_new, axis=1, keepdims=True) * x_new
            continue
        
        for ls_iter in range(30):
            theta = np.linalg.norm(d, axis=1)
            theta_safe = np.where(theta < 1e-12, 1.0, theta)
            d_hat = d / theta_safe[:, np.newaxis]
            
            x_trial = x_new * np.cos(alpha * theta_safe)[:, np.newaxis] + d_hat * np.sin(alpha * theta_safe)[:, np.newaxis]
            x_trial /= np.linalg.norm(x_trial, axis=1, keepdims=True)
            
            f_trial = _compute_energy(x_trial)
            
            if f_trial < f_new + 1e-4 * alpha * df0:
                break
            else:
                alpha *= 0.5
        else:
            break
        
        # Update
        s_k = x_trial - x_new
        y_k = _compute_grad(x_trial) - g_new
        y_k = y_k - np.sum(y_k * x_trial, axis=1, keepdims=True) * x_trial
        
        x_new = x_trial
        f_new = f_trial
        g_new = _compute_grad(x_new)
        g_new = g_new - np.sum(g_new * x_new, axis=1, keepdims=True) * x_new
        
        # Check convergence
        if np.linalg.norm(g_new) < 1e-10:
            break
        
        # Update history
        s_hist.append(s_k)
        y_hist.append(y_k)
        f_hist.append(f_new)
        g_hist.append(g_new)
        
        # Keep only last m
        if len(s_hist) > m:
            s_hist.pop(0)
            y_hist.pop(0)
            f_hist.pop(0)
            g_hist.pop(0)
        
        # Two-loop recursion for L-BFGS
        gamma = np.sum(s_k * y_k)
        if gamma < 1e-12:
            beta = 0
        else:
            beta = np.sum(s_k * y_k) / gamma
        
        # Simple: use Polak-Ribiere beta
        num = np.sum((g_new - g_hist[-2]) * s_k) if len(g_hist) >= 2 else 0
        den = np.sum(s_k * s_k)
        if den < 1e-12:
            beta = 0
        else:
            beta = num / den
            if beta < 0:
                beta = 0
        
        d = -g_new + beta * d
        d = d - np.sum(d * x_new, axis=1, keepdims=True) * x_new
        
        # Check descent
        if np.sum(g_new * d) >= 0:
            d = -g_new
            d = d - np.sum(d * x_new, axis=1, keepdims=True) * x_new
    
    return x_new

def _rotate_points(x, i, j, angle):
    """
    Rotate points i and j around the axis defined by their difference vector.
    """
    x_new = x.copy()
    v = x[i] - x[j]
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return x_new
    
    axis = v / norm_v
    
    c = np.cos(angle)
    s = np.sin(angle)
    k = axis
    v1 = x[i]
    v2 = x[j]
    
    v1_rot = v1 * c + np.cross(k, v1) * s + k * np.dot(k, v1) * (1 - c)
    v2_rot = v2 * c + np.cross(k, v2) * s + k * np.dot(k, v2) * (1 - c)
    
    v1_rot /= np.linalg.norm(v1_rot)
    v2_rot /= np.linalg.norm(v2_rot)
    
    x_new[i] = v1_rot
    x_new[j] = v2_rot
    
    return x_new

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initial configuration: Fibonacci sphere
    x = _fibonacci_sphere(N)
    best_x = x.copy()
    best_energy = _compute_energy(x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Riemannian L-BFGS
    x = _riemannian_lbfgs(x, max_iter=200, time_limit=15.0)
    e = _compute_energy(x)
    if e < best_energy:
        best_energy = e
        best_x = x.copy()
        
    # Phase 2: Basin Hopping with L-BFGS restarts
    T = 5.0
    T_end = 0.05
    n_hops = 3000  # Will be limited by time
    
    current_x = best_x.copy()
    current_e = best_energy
    
    for hop in range(n_hops):
        if time.time() > deadline - 10.0:  # Leave 10s for final polish
            break
            
        # Perturb: random rotation of a few points
        n_perturb = rng.integers(3, 8)
        indices = rng.choice(N, n_perturb, replace=False)
        x_perturb = current_x.copy()
        for idx in indices:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(-0.15, 0.15)
            c = np.cos(angle)
            s = np.sin(angle)
            v = x_perturb[idx]
            v_rot = v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
            v_rot /= np.linalg.norm(v_rot)
            x_perturb[idx] = v_rot
            
        # Optimize
        x_opt = _riemannian_lbfgs(x_perturb, max_iter=30, time_limit=1.5)
        e_opt = _compute_energy(x_opt)
        
        # Accept
        delta_e = e_opt - current_e
        if delta_e < 0:
            accept = True
        else:
            if T < 1e-10:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta_e / T)
                
        if accept:
            current_x = x_opt
            current_e = e_opt
            if current_e < best_energy:
                best_energy = current_e
                best_x = current_x.copy()
                
        # Cool
        T *= 0.995
        
    # Phase 3: Final high-precision polish
    # Use many small L-BFGS steps from slightly perturbed states
    x = best_x.copy()
    e = best_energy
    
    # First, do a very careful L-BFGS with tight tolerance
    x = _riemannian_lbfgs(x, max_iter=500, time_limit=8.0)
    e = _compute_energy(x)
    if e < best_energy:
        best_energy = e
        best_x = x.copy()
    
    # Then, try a few more perturbations + optimizations
    for trial in range(20):
        if time.time() > deadline - 2.0:
            break
        
        # Small random perturbation
        x_trial = best_x.copy()
        n_pert = rng.integers(2, 5)
        idxs = rng.choice(N, n_pert, replace=False)
        for idx in idxs:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(-0.05, 0.05)
            c = np.cos(angle)
            s = np.sin(angle)
            v = x_trial[idx]
            v_rot = v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
            v_rot /= np.linalg.norm(v_rot)
            x_trial[idx] = v_rot
        
        x_opt = _riemannian_lbfgs(x_trial, max_iter=50, time_limit=0.5)
        e_opt = _compute_energy(x_opt)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
    
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

# EVOLVE-BLOCK-END
