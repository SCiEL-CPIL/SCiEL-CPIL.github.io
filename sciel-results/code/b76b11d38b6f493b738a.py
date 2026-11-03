"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _fibonacci_sphere(n, rot=None):
    """Generate n points on unit sphere via Fibonacci spiral, optionally rotated."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if rot is not None:
        pts = pts @ rot.T
    # Ensure unit length
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _log_energy(pts):
    """Compute log energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)"""
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
    """Compute gradient of log energy w.r.t. each point.
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
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
            grad[i] += diff / (d * d)
    return grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _gradient_descent(pts, max_iter=20000, lr_init=0.01, lr_min=1e-7, tol=1e-10, time_budget=None, start_time=None):
    """Gradient descent with adaptive learning rate on the sphere."""
    if start_time is None:
        start_time = time.time()
    pts = pts.copy()
    E_prev = _log_energy(pts)
    lr = lr_init
    for it in range(max_iter):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        grad = _log_energy_grad(pts)
        # Project gradient to tangent space
        for i in range(len(pts)):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        # Update
        pts_new = pts - lr * grad
        pts_new = _project_to_sphere(pts_new)
        E_new = _log_energy(pts_new)
        if E_new < E_prev:
            pts = pts_new
            E_prev = E_new
            # Decay learning rate slowly
            lr = max(lr * 0.999, lr_min)
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < lr_min:
                # Try smaller step
                pts_new = pts - lr * grad
                pts_new = _project_to_sphere(pts_new)
                E_new = _log_energy(pts_new)
                if E_new < E_prev:
                    pts = pts_new
                    E_prev = E_new
                else:
                    break
        if it > 100 and abs(E_prev - _log_energy(pts)) < tol:
            break
    return pts, E_prev

def _simulated_annealing(pts, max_iter=50000, T_init=0.1, T_min=1e-6, time_budget=None, start_time=None):
    """Simulated annealing on the sphere to escape local minima."""
    if start_time is None:
        start_time = time.time()
    pts = pts.copy()
    E_current = _log_energy(pts)
    best_pts = pts.copy()
    E_best = E_current
    T = T_init
    for it in range(max_iter):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        # Random perturbation
        i = np.random.randint(len(pts))
        angle = np.random.uniform(0.1, 0.5)
        # Random tangent direction
        tangent = np.random.randn(3)
        tangent -= np.dot(tangent, pts[i]) * pts[i]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            continue
        tangent /= tangent_norm
        # Rotate point i around tangent axis by angle
        # Use Rodrigues' rotation formula
        v = tangent
        theta = angle
        pts_new = pts.copy()
        x = pts[i]
        pts_new[i] = x * np.cos(theta) + np.cross(v, x) * np.sin(theta) + v * np.dot(v, x) * (1 - np.cos(theta))
        pts_new[i] = _project_to_sphere(pts_new[i:i+1])[0]
        
        E_new = _log_energy(pts_new)
        delta_E = E_new - E_current
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pts = pts_new
            E_current = E_new
            if E_current < E_best:
                E_best = E_current
                best_pts = pts.copy()
        # Cool down
        T = max(T * 0.9995, T_min)
    return best_pts, E_best

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    # Generate diverse initializations
    inits = []
    
    # 1. Standard Fibonacci
    inits.append(_fibonacci_sphere(N))
    
    # 2. Rotated Fibonacci (several rotations)
    rng = np.random.RandomState(42)
    for k in range(5):
        # Random rotation matrix
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        inits.append(_fibonacci_sphere(N, rot=Q))
    
    # 3. Random points (a few)
    for k in range(3):
        pts = rng.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_pts = None
    best_E = float('inf')
    
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time for this restart
        time_per_restart = min(remaining / (len(inits) - idx), 20.0)
        
        # Phase 1: Gradient descent
        gd_time = min(time_per_restart * 0.6, 10.0)
        pts_gd, E_gd = _gradient_descent(init_pts, max_iter=5000, lr_init=0.005, 
                                         time_budget=gd_time, start_time=start_time)
        
        # Phase 2: Simulated annealing
        sa_time = min(time_per_restart * 0.4, 10.0)
        if time.time() - start_time + sa_time < deadline - 2.0:
            pts_sa, E_sa = _simulated_annealing(pts_gd, max_iter=10000, T_init=0.05,
                                                time_budget=sa_time, start_time=start_time)
            if E_sa < E_gd:
                pts_gd = pts_sa
                E_gd = E_sa
        
        # Phase 3: Final polish with gradient descent
        polish_time = min(5.0, deadline - (time.time() - start_time) - 2.0)
        if polish_time > 1.0:
            pts_polish, E_polish = _gradient_descent(pts_gd, max_iter=3000, lr_init=0.001,
                                                     time_budget=polish_time, start_time=start_time)
            if E_polish < E_gd:
                pts_gd = pts_polish
                E_gd = E_polish
        
        if E_gd < best_E:
            best_E = E_gd
            best_pts = pts_gd.copy()
    
    # Final high-precision polish if time remains
    if best_pts is not None:
        remaining = deadline - (time.time() - start_time)
        if remaining > 3.0:
            pts_final, E_final = _gradient_descent(best_pts, max_iter=5000, lr_init=0.0005,
                                                   time_budget=min(remaining - 2.0, 10.0), start_time=start_time)
            if E_final < best_E:
                best_pts = pts_final
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
