import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 145.0
EPS = 1e-12

def _energy_and_ambient_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, ambient_grad) where ambient_grad is (N, 3).
    """
    dot = pts @ pts.T
    dist2 = 2.0 - 2.0 * dot
    np.fill_diagonal(dist2, 0.0)
    dist2 = np.maximum(dist2, EPS)
    
    log_dist2 = np.log(dist2)
    np.fill_diagonal(log_dist2, 0.0)
    energy = -0.25 * np.sum(log_dist2)
    
    # grad_i = sum_{j != i} x_j / (2 * (1 - x_i . x_j))
    # 1 - x_i . x_j = dist2_ij / 2
    # So term is x_j / dist2_ij
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad_i = sum_j (x_j / dist2_ij)
    ambient_grad = inv_dist2 @ pts
    
    return energy, ambient_grad

def _theta_phi_to_pts(theta, phi):
    """Convert theta (N,) and phi (N,) to pts (N, 3)."""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    
    pts = np.column_stack([x, y, z])
    # Normalize just in case of numerical drift
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    pts = pts / norms
    return pts

def _pts_to_theta_phi(pts):
    """Convert pts (N, 3) to theta (N,) and phi (N,)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi

def _energy_and_grad_theta_phi(params):
    """
    params: (2N,) array [theta_0...theta_{N-1}, phi_0...phi_{N-1}].
    Returns (energy, grad_params).
    """
    theta = params[:N]
    phi = params[N:]
    
    pts = _theta_phi_to_pts(theta, phi)
    energy, ambient_grad = _energy_and_ambient_grad(pts)
    
    # Compute Jacobian of pts w.r.t theta, phi
    # dpts/dtheta_i = (cos(theta_i)cos(phi_i), cos(theta_i)sin(phi_i), -sin(theta_i))
    # dpts/dphi_i = (-sin(theta_i)sin(phi_i), sin(theta_i)cos(phi_i), 0)
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    dpts_dtheta = np.column_stack([
        cos_t * cos_p,
        cos_t * sin_p,
        -sin_t
    ])
    
    dpts_dphi = np.column_stack([
        -sin_t * sin_p,
        sin_t * cos_p,
        0.0
    ])
    
    # grad_theta_i = dot(ambient_grad_i, dpts_dtheta_i)
    grad_theta = np.sum(ambient_grad * dpts_dtheta, axis=1)
    grad_phi = np.sum(ambient_grad * dpts_dphi, axis=1)
    
    grad_params = np.concatenate([grad_theta, grad_phi])
    
    return energy, grad_params

def _optimize_from_start(theta, phi, t_start=None, time_limit=20.0):
    """
    Optimize starting from theta, phi.
    """
    if t_start is None:
        t_start = time.time()
        
    x0 = np.concatenate([theta, phi])
    
    bounds = [((0, math.pi), (0, 2*math.pi))] * N
    
    # L-BFGS-B is good for this
    try:
        res = minimize(_energy_and_grad_theta_phi, x0, jac=True, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0})
        
        if res.success or np.isfinite(res.fun):
            x_opt = res.x
            theta_opt = x_opt[:N]
            phi_opt = x_opt[N:]
            pts_opt = _theta_phi_to_pts(theta_opt, phi_opt)
            energy_opt, _ = _energy_and_ambient_grad(pts_opt)
            return pts_opt, energy_opt
    except Exception:
        pass
        
    # Fallback: return initial
    pts_init = _theta_phi_to_pts(theta, phi)
    energy_init, _ = _energy_and_ambient_grad(pts_init)
    return pts_init, energy_init

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def construct():
    t_start = time.time()
    
    best_pts = None
    best_energy = np.inf
    
    # List of initializations
    inits = []
    
    # 1. Fibonacci spiral
    fib_pts = _fibonacci_spiral(N)
    fib_theta, fib_phi = _pts_to_theta_phi(fib_pts)
    inits.append((fib_theta, fib_phi))
    
    # 2. Random starts
    for seed in range(10):
        rnd_pts = _random_points(N, seed=seed)
        rnd_theta, rnd_phi = _pts_to_theta_phi(rnd_pts)
        inits.append((rnd_theta, rnd_phi))
        
    # Optimize each
    for idx, (theta, phi) in enumerate(inits):
        elapsed = time.time() - t_start
        remaining = DEADLINE_SECONDS - elapsed
        if remaining < 5:
            break
            
        # Allocate time
        # We have 11 inits. Let's give ~10s each if time permits.
        # But L-BFGS-B is fast.
        time_limit = min(15.0, remaining * 0.8)
        
        pts_opt, energy_opt = _optimize_from_start(theta, phi, t_start=t_start, time_limit=time_limit)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
            
    # Final polish on the best result
    if best_pts is not None and time.time() - t_start < DEADLINE_SECONDS - 10:
        best_theta, best_phi = _pts_to_theta_phi(best_pts)
        best_pts, best_energy = _optimize_from_start(best_theta, best_phi, t_start=t_start, time_limit=DEADLINE_SECONDS - time.time() - 5)
        
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
