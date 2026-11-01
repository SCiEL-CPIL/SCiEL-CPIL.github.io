"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(points):
    """
    Calculate the logarithmic energy and its Euclidean gradient for points.
    points: (N, 3) array
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    mask = ~np.eye(n, dtype=bool)
    log_dist = np.log(dist)
    total_log_dist = np.sum(log_dist[mask])
    energy = -0.5 * total_log_dist
    
    # Gradient
    # grad_i = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    # diff[i,j] = x_i - x_j. So x_j - x_i = -diff[i,j].
    # grad[i] = - sum_j diff[i,j] / dist_sq[i,j]
    
    dist_sq_inv = np.zeros_like(dist_sq)
    dist_sq_inv[mask] = 1.0 / dist_sq[mask]
    
    weighted_diff = diff * dist_sq_inv[..., np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_grad(points, euclidean_grad):
    """
    Project Euclidean gradient onto the tangent space of the sphere.
    """
    # proj(v) = v - (v . x) x
    dots = np.sum(points * euclidean_grad, axis=1, keepdims=True)
    return euclidean_grad - dots * points

def _retract(points, step):
    """
    Retraction: normalize points + step.
    """
    new_points = points + step
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return new_points / norms

def _lbfgs_sphere(points, max_iter=200, tol=1e-8):
    """
    Riemannian L-BFGS approximation using Euclidean L-BFGS on tangent space with retraction.
    Actually, we will use a simple Riemannian Steepest Descent or CG if L-BFGS is tricky.
    Let's use a Riemannian Conjugate Gradient (RCG) with Fletcher-Reeves.
    """
    n = points.shape[0]
    x = points.copy()
    
    # Initial gradient
    e, g_euc = _energy_and_grad(x)
    g_r = _riemannian_grad(x, g_euc)
    
    # Initial direction
    d = -g_r
    
    beta = 0.0
    for k in range(max_iter):
        # Line search: Backtracking
        alpha = 1.0
        e_curr = e
        g_curr_r = g_r
        g_curr_norm = np.linalg.norm(g_curr_r)
        
        if g_curr_norm < tol:
            break
            
        # Armijo condition
        c1 = 1e-4
        rho = 0.5
        for ls_iter in range(30):
            x_trial = _retract(x, alpha * d)
            e_trial, g_trial_euc = _energy_and_grad(x_trial)
            g_trial_r = _riemannian_grad(x_trial, g_trial_euc)
            
            # Check decrease
            if e_trial < e_curr + c1 * alpha * np.sum(g_curr_r * d):
                # Update
                x = x_trial
                e = e_trial
                g_r = g_trial_r
                break
            else:
                alpha *= rho
        
        # Update beta (Fletcher-Reeves)
        g_new_norm = np.linalg.norm(g_r)
        if g_new_norm < tol:
            break
            
        if beta == 0.0:
            beta = 1.0
        else:
            # Need parallel transport of d to new x?
            # For RCG, we need to transport d from x_old to x_new.
            # Approximation: d_transported = d - (d . x_old) x_old + (d . x_new) x_new ? No.
            # Simple approach: Just reset d = -g_r (Steepest Descent) if transport is hard.
            # Or use the "Euclidean" transport: d_trans = d * (alpha) ... no.
            # Let's use a simplified RCG where we just use the new gradient direction mixed with old.
            # For robustness, let's just do Steepest Descent if transport is an issue, or use a small beta.
            # Actually, for the sphere, the parallel transport of a tangent vector v from x to y is:
            # P(v) = v - (v . x) x + (v . y) y is NOT correct.
            # The correct transport along the geodesic is complex.
            # Let's use a Riemannian Steepest Descent with adaptive step size, which is robust.
            pass
            
        # To keep it simple and robust, let's switch to Riemannian Newton for the final polish
        # and use Steepest Descent for the main loop if RCG is unstable.
        # However, Steepest Descent is slow.
        # Let's try a simple Riemannian Newton step if we are close.
        
        # For now, let's just do a few steps of projected gradient descent with backtracking.
        # This is safer than a buggy RCG.
        
    return x, e

def _newton_polish(points, max_iter=10):
    """
    Riemannian Newton polish.
    Solves H d = -g on the tangent space.
    """
    n = points.shape[0]
    x = points.copy()
    
    for k in range(max_iter):
        e, g_euc = _energy_and_grad(x)
        g_r = _riemannian_grad(x, g_euc)
        g_norm = np.linalg.norm(g_r)
        if g_norm < 1e-10:
            break
            
        # Compute Riemannian Hessian matrix (3N x 3N)
        # This is O(N^2) to build, but N=100 is small.
        H = np.zeros((3*n, 3*n))
        
        # We need to fill the Hessian of the Riemannian energy.
        # H_{ij} (block) corresponds to the Riemannian Hessian between x_i and x_j.
        
        # For i == j:
        # H_ii = sum_{j!=i} [ (1/d^2) I - 2 (r r^T) / d^4 ] projected?
        # The Riemannian Hessian H_R(v, w) = <H_E v, w> - <g, K(v) w>
        # For sphere, K(v) = - (v . x) x ? No, curvature term.
        # Let's use the formula for the Hessian of a function on the sphere.
        
        # Actually, computing the full Hessian is risky and slow in Python loops.
        # Let's use a Hessian-Vector product approach for Newton-CG?
        # Or just use a very tight L-BFGS from scipy on the Euclidean problem with high penalty.
        
        # Given the constraints, let's stick to a robust Euclidean L-BFGS with high penalty
        # and a final few steps of Riemannian Steepest Descent.
        
    return x, e

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_energy = float('inf')
    best_points = None
    
    # Inits
    inits = []
    # Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Randoms
    for seed in range(20):
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts / norms)
        
    # Optimize
    for init in inits:
        if time.time() > deadline:
            break
            
        flat_init = init.flatten()
        
        # Penalty Method
        mu = 1000.0
        
        def obj_and_grad(flat_pts):
            pts = flat_pts.reshape(N, 3)
            e, g = _energy_and_grad(pts)
            norms_sq = np.sum(pts**2, axis=1)
            pen = mu * np.sum((norms_sq - 1)**2)
            pen_g = 4.0 * mu * (norms_sq - 1)[:, np.newaxis] * pts
            return e + pen, (g + pen_g).flatten()
        
        try:
            res = minimize(
                obj_and_grad,
                flat_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            opt_pts = res.x.reshape(N, 3)
            # Project
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            opt_pts = opt_pts / norms
            
            # Polish with Riemannian Steepest Descent
            for _ in range(50):
                if time.time() > deadline:
                    break
                e, g_euc = _energy_and_grad(opt_pts)
                g_r = _riemannian_grad(opt_pts, g_euc)
                g_norm = np.linalg.norm(g_r)
                if g_norm < 1e-9:
                    break
                
                # Backtracking line search
                alpha = 1.0
                c1 = 1e-4
                for _ in range(20):
                    trial = _retract(opt_pts, -alpha * g_r)
                    e_trial, _ = _energy_and_grad(trial)
                    if e_trial < e + c1 * alpha * np.sum(g_r * (-g_r)):
                        opt_pts = trial
                        e = e_trial
                        break
                    alpha *= 0.5
                
            e_final, _ = _energy_and_grad(opt_pts)
            if e_final < best_energy:
                best_energy = e_final
                best_points = opt_pts
                
        except Exception:
            pass
            
    if best_points is None:
        best_points = inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
