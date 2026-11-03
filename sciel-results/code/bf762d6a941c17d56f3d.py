"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d2 = dist2[iu]
    d2 = np.maximum(d2, 1e-30)
    d = np.sqrt(d2)
    log_terms = np.log(1.0 / d)
    return math.fsum(log_terms.tolist())

def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points (in R^3)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    inv_dist2 = 1.0 / dist2
    term = diff * inv_dist2[:, :, np.newaxis]
    grad = np.sum(term, axis=1)
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _tangent_grad(pts):
    """Compute tangent-space gradient (projected to sphere)."""
    grad = _log_energy_grad(pts)
    g_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_x * pts

def _polish_lbfgs(pts, deadline):
    """Polish points using L-BFGS-B with tight tolerances."""
    try:
        from scipy.optimize import minimize
        
        def _energy_and_grad(x_flat):
            x = x_flat.reshape(N, 3)
            x = _project_to_sphere(x)
            e = _log_energy(x)
            g = _log_energy_grad(x)
            g_dot_x = np.sum(g * x, axis=1, keepdims=True)
            g_tan = g - g_dot_x * x
            return e, g_tan.flatten()
        
        x0 = pts.flatten()
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 3000, 'gtol': 1e-14, 'maxls': 200})
        if np.all(np.isfinite(res.x)):
            pts_polished = _project_to_sphere(res.x.reshape(N, 3))
            return pts_polished
    except Exception:
        pass
    return pts

def _gradient_descent_tangent(pts, deadline, lr_start=1e-6, lr_end=1e-10, n_steps=2000):
    """Gradient descent in tangent space with adaptive learning rate."""
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        
        t = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** t
        
        g_tan = _tangent_grad(pts)
        pts = pts - lr * g_tan
        pts = _project_to_sphere(pts)
    
    return pts

def _momentum_descent(pts, deadline, lr=1e-7, momentum=0.9, n_steps=5000):
    """Momentum-based gradient descent in tangent space."""
    velocity = np.zeros_like(pts)
    
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        
        g_tan = _tangent_grad(pts)
        velocity = momentum * velocity - lr * g_tan
        pts = pts + velocity
        pts = _project_to_sphere(pts)
        
        # Project velocity to tangent space
        v_dot_x = np.sum(velocity * pts, axis=1, keepdims=True)
        velocity = velocity - v_dot_x * pts
    
    return pts

def _newton_step(pts, deadline):
    """Attempt a few Newton-like steps using approximate Hessian."""
    # For log energy, the Hessian is complex; use L-BFGS as approximation
    return pts

def _refine_with_restart(pts, deadline, n_restarts=3):
    """Refine with small perturbations and re-optimize."""
    best = pts.copy()
    best_e = _log_energy(pts)
    
    rng = np.random.default_rng(123)
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Small perturbation
        noise = rng.normal(0, 1e-4, size=pts.shape)
        perturbed = _project_to_sphere(pts + noise)
        
        # Quick polish
        polished = _polish_lbfgs(perturbed, deadline - 5.0)
        e = _log_energy(polished)
        
        if e < best_e:
            best_e = e
            best = polished.copy()
    
    return best

def construct():
    """Construct 150 points on unit sphere minimizing log energy."""
    deadline = time.time() + 145.0  # Safety margin
    
    # Start from Fibonacci sphere
    pts = _fibonacci_sphere(N)
    pts = _project_to_sphere(pts)
    
    # Phase 1: Initial L-BFGS polish
    pts = _polish_lbfgs(pts, deadline - 60.0)
    e1 = _log_energy(pts)
    
    # Phase 2: Momentum descent for fine-tuning
    pts = _momentum_descent(pts, deadline - 30.0, lr=5e-8, momentum=0.95, n_steps=8000)
    e2 = _log_energy(pts)
    
    # Phase 3: Final L-BFGS polish
    pts = _polish_lbfgs(pts, deadline - 15.0)
    e3 = _log_energy(pts)
    
    # Phase 4: Very fine gradient descent
    pts = _gradient_descent_tangent(pts, deadline - 5.0, lr_start=1e-8, lr_end=1e-12, n_steps=3000)
    e4 = _log_energy(pts)
    
    # Phase 5: One more L-BFGS pass
    pts = _polish_lbfgs(pts, deadline - 2.0)
    e5 = _log_energy(pts)
    
    # Ensure validity
    pts = _project_to_sphere(pts)
    
    # Verify minimum distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        pts = _fibonacci_sphere(N)
    
    return pts

# EVOLVE-BLOCK-END
