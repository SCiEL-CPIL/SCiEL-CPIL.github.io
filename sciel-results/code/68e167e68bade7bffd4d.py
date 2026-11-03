"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _remaining(deadline):
    return deadline - time.time()

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _energy_fast(pts):
    """Compute logarithmic energy efficiently."""
    pts = np.asarray(pts)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_grad(pts):
    """Compute energy and gradient."""
    pts = np.asarray(pts)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = diff / dist_sq[:, :, np.newaxis]
    idx = np.diag_indices(n)
    terms[idx[0], idx[1], :] = 0.0
    grad = np.sum(terms, axis=1)
    return energy, grad

def _precise_energy(pts):
    """Compute energy using math.fsum for precision."""
    pts = np.asarray(pts)
    n = len(pts)
    total = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            total.append(math.log(1.0 / d))
    return math.fsum(total)

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _exp_map(pts, v):
    """Exponential map on unit sphere: move point p by tangent vector v."""
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    norm_v = np.maximum(norm_v, 1e-15)
    angle = norm_v
    pts_new = pts * np.cos(angle) + v * (np.sin(angle) / norm_v)
    return _normalize(pts_new)

def _riemannian_polish(pts, maxiter=2000, tol=1e-12, deadline=None):
    """Riemannian gradient descent with backtracking line search."""
    pts = _normalize(pts.copy())
    energy, grad = _energy_grad(pts)
    
    for it in range(maxiter):
        if deadline is not None and _remaining(deadline) < 2.0:
            break
        
        # Project gradient to tangent space
        g_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Check convergence
        gnorm = np.max(np.linalg.norm(g_tan, axis=1))
        if gnorm < tol:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_pts = pts
        best_energy = energy
        for ls in range(30):
            pts_trial = _exp_map(pts, -alpha * g_tan)
            e_trial = _energy_fast(pts_trial)
            if e_trial < energy:
                best_pts = pts_trial
                best_energy = e_trial
                break
            alpha *= 0.5
        
        if best_pts is pts:
            # Try smaller step
            alpha = 1e-4
            pts_trial = _exp_map(pts, -alpha * g_tan)
            e_trial = _energy_fast(pts_trial)
            if e_trial < energy:
                best_pts = pts_trial
                best_energy = e_trial
        
        pts = best_pts
        energy = best_energy
        _, grad = _energy_grad(pts)
    
    return pts, energy

def _newton_polish(pts, maxiter=100, deadline=None):
    """Newton-style polish using Hessian approximation."""
    pts = _normalize(pts.copy())
    energy, grad = _energy_grad(pts)
    
    for it in range(maxiter):
        if deadline is not None and _remaining(deadline) < 2.0:
            break
        
        g_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        gnorm = np.max(np.linalg.norm(g_tan, axis=1))
        if gnorm < 1e-13:
            break
        
        # Use diagonal approximation of Hessian for Newton step
        # Approximate Hessian diagonal from finite differences
        h = 1e-6
        hessian_diag = np.zeros_like(pts)
        for i in range(len(pts)):
            for d in range(3):
                pts_plus = pts.copy()
                pts_plus[i, d] += h
                pts_plus = _normalize(pts_plus)
                _, g_plus = _energy_grad(pts_plus)
                g_plus_tan = g_plus[i] - np.sum(g_plus[i] * pts_plus[i]) * pts_plus[i]
                
                pts_minus = pts.copy()
                pts_minus[i, d] -= h
                pts_minus = _normalize(pts_minus)
                _, g_minus = _energy_grad(pts_minus)
                g_minus_tan = g_minus[i] - np.sum(g_minus[i] * pts_minus[i]) * pts_minus[i]
                
                hessian_diag[i, d] = (g_plus_tan[d] - g_minus_tan[d]) / (2 * h)
        
        # Newton step with damping
        hessian_diag = np.maximum(np.abs(hessian_diag), 1e-8)
        step = g_tan / hessian_diag
        step_norm = np.linalg.norm(step, axis=1, keepdims=True)
        step_norm = np.maximum(step_norm, 1e-12)
        step = step / step_norm * np.minimum(step_norm, 0.1)
        
        # Line search
        alpha = 1.0
        for ls in range(20):
            pts_trial = _exp_map(pts, -alpha * step)
            e_trial = _energy_fast(pts_trial)
            if e_trial < energy:
                pts = pts_trial
                energy = e_trial
                break
            alpha *= 0.5
        
        _, grad = _energy_grad(pts)
    
    return pts, energy

def _langevin_verlet(pts, deadline, n_steps=5000, dt=0.005, temp_init=0.5, temp_min=0.01, seed=42):
    """
    Langevin dynamics with velocity Verlet integrator.
    Points evolve under logarithmic repulsion with stochastic kicks.
    """
    rng = np.random.RandomState(seed)
    pts = _normalize(pts.copy())
    n = len(pts)
    
    # Initialize velocities to zero
    vel = np.zeros_like(pts)
    
    energy, grad = _energy_grad(pts)
    
    # Temperature schedule: geometric cooling
    temp_ratio = (temp_min / temp_init) ** (1.0 / max(n_steps, 1))
    
    best_pts = pts.copy()
    best_energy = energy
    
    for step in range(n_steps):
        if _remaining(deadline) < 3.0:
            break
        
        temp = temp_init * (temp_ratio ** step)
        
        # Compute force (negative gradient)
        force = -grad
        
        # Velocity Verlet: half-step velocity
        vel_half = vel + 0.5 * dt * force
        
        # Update positions using exponential map
        pts_new = _exp_map(pts, dt * vel_half)
        
        # Compute new forces
        energy_new, grad_new = _energy_grad(pts_new)
        force_new = -grad_new
        
        # Full-step velocity with stochastic kick
        vel_new = vel_half + 0.5 * dt * force_new
        # Add Langevin noise
        noise_scale = np.sqrt(2.0 * temp * dt)
        vel_new = vel_new + noise_scale * rng.randn(n, 3)
        
        # Damping (friction)
        vel_new *= 0.99
        
        # Check if this configuration is better
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
        
        pts = pts_new
        vel = vel_new
        energy = energy_new
        grad = grad_new
    
    return best_pts, best_energy

def _optimize_from_start(pts0, deadline, seed=0):
    """
    Full optimization pipeline: Langevin global search + Riemannian polish + Newton polish.
    """
    pts = _normalize(pts0.copy())
    
    # Phase 1: Langevin dynamics for global exploration
    pts_l, e_l = _langevin_verlet(pts, deadline, n_steps=3000, dt=0.004, temp_init=0.3, temp_min=0.005, seed=seed)
    
    # Phase 2: Riemannian gradient descent for local convergence
    pts_r, e_r = _riemannian_polish(pts_l, maxiter=1500, tol=1e-12, deadline=deadline)
    
    # Phase 3: Newton polish for high-precision local minimum
    pts_n, e_n = _newton_polish(pts_r, maxiter=50, deadline=deadline)
    
    return pts_n, e_n

def _spectral_init(n):
    """
    Spectral initialization: use eigenvectors of a random matrix to get
    roughly uniform points on the sphere.
    """
    rng = np.random.RandomState(12345)
    # Generate random points and use them to construct a Gram-like matrix
    pts_random = rng.randn(n, 3)
    pts_random = _normalize(pts_random)
    # Use SVD to get orthogonal directions, then map to sphere
    U, S, Vt = np.linalg.svd(pts_random)
    # Take first n columns of Vt (or rows of Vt.T)
    # Actually, let's just use the random points but ensure they're well-spread
    return pts_random

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin before 150s limit
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_random_sphere(N, seed=1))
    inits.append(_random_sphere(N, seed=2))
    inits.append(_random_sphere(N, seed=3))
    
    best_pts = None
    best_energy = float('inf')
    
    # Try each initialization with Langevin + polish
    for i, init in enumerate(inits):
        if _remaining(deadline) < 15.0:
            break
        
        pts_opt, energy = _optimize_from_start(init, deadline, seed=i)
        
        # Validate
        if np.all(np.isfinite(pts_opt)):
            diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            min_dist = np.sqrt(np.min(dist_sq))
            
            if min_dist > 1e-9 and energy < best_energy:
                best_energy = energy
                best_pts = pts_opt
    
    # If no valid result, fall back to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final projection and validation
    best_pts = _normalize(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
