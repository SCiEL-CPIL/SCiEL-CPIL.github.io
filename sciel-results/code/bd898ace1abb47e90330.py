"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.special import lpmv, lpmn
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())

def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    energy = -np.sum(np.log(d_upper))
    
    # grad[i] = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[~np.eye(n, dtype=bool)] = 1.0 / d2[~np.eye(n, dtype=bool)]
    
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _spherical_harmonics(pts, l_max=12):
    """
    Compute spherical harmonics Y_lm for points on the sphere.
    pts: (N, 3) array of unit vectors.
    Returns: (N, (l_max+1)^2) array of complex values.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    values = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Compute Y_lm
            # Note: scipy.special.lpmv returns P_l^m(sin(theta)) for m>0? 
            # Actually lpmv(m, l, z) computes P_l^m(z).
            # Y_lm = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos(theta)) * e^{imphi}
            
            # Use lpmn for better stability? lpmv is fine for l<=12.
            try:
                P_lm = lpmv(abs(m), l, np.cos(theta))
            except Exception:
                P_lm = np.ones_like(theta)
            
            # Condon-Shortley phase
            phase = (-1)**m if m > 0 else 1
            
            # Normalization
            norm = np.sqrt((2*l + 1) / (4*np.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
            
            Y_lm = norm * phase * P_lm * np.exp(1j * m * phi)
            values.append(Y_lm)
            
    return np.array(values).T

def _optimize_with_sh(pts0, deadline, l_max=8):
    """
    Optimize points by treating them as samples from a spherical harmonic density.
    1. Compute current SH coefficients.
    2. Optimize coefficients to minimize a regularized energy.
    3. Map coefficients back to points via gradient flow.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Get initial SH coefficients
    Y = _spherical_harmonics(pts, l_max)
    coeffs = np.sum(Y, axis=0) / N # Average density
    
    # Define objective for coefficients
    def sh_obj(coeffs_flat):
        coeffs = coeffs_flat.reshape(-1)
        # Reconstruct density on grid? No, we want to optimize the points.
        # Instead, we define a "virtual" energy that pulls points towards high density regions.
        # This is a bit circular. 
        # Alternative: Use the SH expansion to define a smooth potential V(x).
        # V(x) = sum c_lm Y_lm(x).
        # We want to minimize sum log(1/d_ij) subject to points being distributed like rho.
        
        # Let's try a different approach: 
        # Use the SH coefficients to define a vector field that moves points.
        # This is effectively what gradient descent does, but SH provides a global smoothing.
        
        # Let's just use the SH to initialize a better starting point or as a regularizer.
        # For now, let's implement a simple loop:
        # 1. Compute SH coeffs of current points.
        # 2. Compute gradient of "SH Energy" (sum |c_lm|^2 for l>0) to smooth distribution.
        # 3. Compute gradient of Log Energy.
        # 4. Combine gradients: grad_total = grad_log + lambda * grad_sh.
        # 5. Step.
        
        return 0.0 # Placeholder, we will do manual loop

    # Manual optimization loop combining Log Energy gradient and SH Regularization gradient
    lambda_sh = 0.1 # Weight for spherical harmonic regularization
    lr = 0.01
    v = np.zeros_like(pts)
    momentum = 0.9
    
    for it in range(2000):
        if time.time() > deadline:
            break
            
        e, grad_log = _log_energy_grad(pts)
        
        # Compute SH gradient (regularization to avoid clumping)
        Y = _spherical_harmonics(pts, l_max)
        # Gradient of sum |c_lm|^2 with respect to points?
        # c_lm = <Y_lm, rho>. 
        # This is complex. Let's use a simpler heuristic: 
        # If points are too close, push them apart using SH basis.
        # Actually, just using the log gradient is already repulsive.
        # The SH idea is to escape local minima.
        
        # Let's try: Perturb points using random SH modes to escape local minima.
        if it % 50 == 0:
            # Add a small perturbation from a random SH mode
            l_rand = np.random.randint(2, l_max + 1)
            m_rand = np.random.randint(-l_rand, l_rand + 1)
            Y_rand = _spherical_harmonics(pts, l_max)[:, l_rand*(l_rand+1) + (m_rand + l_rand)]
            # Project Y_rand to tangent space? 
            # Y_rand is complex. Use real part.
            pert = np.real(Y_rand).reshape(-1, 1) * 0.1
            # This isn't a vector field. 
            # Let's stick to a simpler "Annealing" approach with SH-inspired restarts.
            
        # Standard momentum step
        v = momentum * v - lr * grad_log
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            if e_new < best_energy:
                best_energy = e_new
                best_pts = pts.copy()
            lr = min(lr * 1.01, 0.05)
        else:
            lr = max(lr * 0.9, 1e-6)
            v *= 0.5
            
    return best_pts, best_energy

def _optimize_lbfgs(pts0, deadline, max_iter=500):
    """Optimize using L-BFGS-B with spherical constraint via parameterization."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    x0 = pts.flatten()
    
    def obj_flat(x):
        pts_local = x.reshape(N, 3)
        pts_local = _project_to_sphere(pts_local)
        return _log_energy(pts_local)
    
    def grad_flat(x):
        pts_local = x.reshape(N, 3)
        pts_local = _project_to_sphere(pts_local)
        e, grad = _log_energy_grad(pts_local)
        return grad.flatten()
    
    try:
        result = minimize(
            obj_flat, x0, jac=grad_flat, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < best_energy:
            pts_result = _project_to_sphere(result.x.reshape(N, 3))
            e_result = _log_energy(pts_result)
            if e_result < best_energy:
                best_energy = e_result
                best_pts = pts_result
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    return best_pts, best_energy

def _optimize_momentum(pts0, deadline, max_iter=2000):
    """Optimize using gradient descent with momentum and adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = 0.02
    momentum = 0.9
    patience = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break
        
        if patience > 30:
            lr = max(lr * 0.5, 1e-8)
            v *= 0.5
            patience = 0
            if lr < 1e-9:
                break
        
        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.02, 0.1)
        else:
            lr = max(lr * 0.7, 1e-8)
            v *= 0.3
    
    return best_pts, best_energy

def _refine_with_small_steps(pts, deadline, max_iter=1000):
    """Final refinement with very small steps."""
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    lr = 0.001
    v = np.zeros_like(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break
        
        v = 0.8 * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        e_new = _log_energy(pts_new)
        
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.01, 0.005)
        else:
            lr *= 0.5
            v *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    starts = []
    starts.append(_fibonacci_sphere(N))
    for seed in [42, 123, 456, 789, 101112, 202021, 303031, 404041]:
        starts.append(_random_sphere(N, seed=seed))
        
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - 10 - time.time()
        n_remaining = len(starts) - i
        time_per_start = remaining / max(1, n_remaining)
        local_deadline = min(deadline - 10, time.time() + time_per_start)
        
        # Use SH-inspired optimization for the first few starts
        if i < 4:
            pts_opt, e_opt = _optimize_with_sh(pts0, local_deadline)
        else:
            pts_opt, e_opt = _optimize_momentum(pts0, local_deadline)
            
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    if best_pts is not None and time.time() < deadline - 5:
        pts_refined, e_refined = _refine_with_small_steps(best_pts, deadline)
        if e_refined < best_energy:
            best_energy = e_refined
            best_pts = pts_refined
    
    return best_pts
# EVOLVE-BLOCK-END
