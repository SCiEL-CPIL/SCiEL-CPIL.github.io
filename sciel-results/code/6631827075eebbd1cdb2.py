"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return total


def _log_energy_grad(pts):
    """Compute gradient of log energy with respect to points on sphere.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            grad[i] -= diff / (d * d)
    # Project gradient onto tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _conjugate_gradient_sphere(pts, max_iter=500, tol=1e-10):
    """Optimize points on sphere using conjugate gradient method.
    
    We minimize E(x) = sum_{i<j} log(1/|x_i - x_j|) subject to |x_i|=1.
    Using the gradient projected to tangent space.
    """
    n = len(pts)
    x = pts.copy().astype(np.float64)
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e = _log_energy(x)
    
    # Initial gradient
    g = _log_energy_grad(x)
    g = _project_to_sphere(g - np.sum(g * x, axis=1, keepdims=True) * x)
    
    # Conjugate gradient
    p = -g.copy()
    r = g.copy()
    rs_old = np.sum(r * r)
    
    for it in range(max_iter):
        # Line search: find alpha that minimizes E along direction p
        # Use a simple backtracking line search
        alpha = 0.01
        for ls_iter in range(50):
            x_trial = x + alpha * p
            x_trial = _project_to_sphere(x_trial)
            e_trial = _log_energy(x_trial)
            if e_trial < best_e or e_trial < best_e - 1e-12 * abs(best_e):
                # Check if this is a sufficient decrease
                break
            alpha *= 0.5
        else:
            break
        
        if e_trial < best_e:
            best_e = e_trial
            best_x = x_trial.copy()
        
        x = x_trial
        g_new = _log_energy_grad(x)
        g_new = _project_to_sphere(g_new - np.sum(g_new * x, axis=1, keepdims=True) * x)
        
        # Check convergence
        if np.linalg.norm(g_new) < tol:
            break
        
        r_new = g_new
        rs_new = np.sum(r_new * r_new)
        
        if rs_new < 1e-30:
            break
        
        beta = rs_new / rs_old
        p = -r_new + beta * p
        r = r_new
        rs_old = rs_new
        
        # Ensure p is in tangent space
        p = _project_to_sphere(p - np.sum(p * x, axis=1, keepdims=True) * x)
    
    return best_x, best_e


def _gradient_descent_sphere(pts, max_iter=1000, lr_init=0.01):
    """Gradient descent on sphere with adaptive learning rate."""
    n = len(pts)
    x = pts.copy().astype(np.float64)
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e = _log_energy(x)
    
    lr = lr_init
    for it in range(max_iter):
        g = _log_energy_grad(x)
        g = _project_to_sphere(g - np.sum(g * x, axis=1, keepdims=True) * x)
        
        # Adaptive step size
        alpha = lr
        x_trial = _project_to_sphere(x - alpha * g)
        e_trial = _log_energy(x_trial)
        
        if e_trial < best_e:
            best_e = e_trial
            best_x = x_trial.copy()
            x = x_trial
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return best_x, best_e


def _newton_like_sphere(pts, max_iter=100):
    """Simple Newton-like method using Hessian approximation via finite differences."""
    n = len(pts)
    x = pts.copy().astype(np.float64)
    x = _project_to_sphere(x)
    
    best_x = x.copy()
    best_e = _log_energy(x)
    
    for it in range(max_iter):
        g = _log_energy_grad(x)
        g = _project_to_sphere(g - np.sum(g * x, axis=1, keepdims=True) * x)
        
        if np.linalg.norm(g) < 1e-12:
            break
        
        # Use gradient descent with small step as fallback
        alpha = 0.001
        x_trial = _project_to_sphere(x - alpha * g)
        e_trial = _log_energy(x_trial)
        
        if e_trial < best_e:
            best_e = e_trial
            best_x = x_trial.copy()
            x = x_trial
    
    return best_x, best_e


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = 120.0 - 15.0  # 15 second safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    pts = _project_to_sphere(pts)
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Phase 1: Conjugate gradient optimization
    if time.time() < deadline:
        pts, e = _conjugate_gradient_sphere(pts, max_iter=200, tol=1e-10)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 2: Gradient descent polish
    if time.time() < deadline:
        pts, e = _gradient_descent_sphere(best_pts, max_iter=500, lr_init=0.005)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 3: Another CG pass with tighter tolerance
    if time.time() < deadline:
        pts, e = _conjugate_gradient_sphere(best_pts, max_iter=300, tol=1e-12)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 4: Final gradient descent with very small steps
    if time.time() < deadline:
        pts, e = _gradient_descent_sphere(best_pts, max_iter=300, lr_init=0.001)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Ensure all points are on the sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts = _project_to_sphere(best_pts)
    
    # Final check for finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
