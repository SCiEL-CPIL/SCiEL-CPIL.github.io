import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _halton_sphere(n, base):
    """Generate points using Halton sequence on sphere."""
    def halton(index, b):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= b
            r += f * (index % b)
            index //= b
        return r
    
    pts = []
    for i in range(1, n + 1):
        u = halton(i, base)
        v = halton(i, base + 1)
        theta = 2.0 * np.pi * u
        phi = np.arccos(1.0 - 2.0 * v)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    return np.array(pts)

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _energy_and_grad(pts):
    """
    Compute log energy and gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Gradient is the projected gradient on the sphere.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/d) = -sum_{i<j} log(d)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = np.sum(np.log(1.0 / dist[mask]))
    
    # Gradient of energy w.r.t. pts[i]:
    # d/dx_i [sum_{j!=i} log(1/|x_i - x_j|)] = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # But we need to be careful: log(1/d) = -log(d), so d/dx_i (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
    # Wait: d/dx log(1/|xi-xj|) = d/dx (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
    # So gradient contribution to point i from pair (i,j) is -(xi-xj)/|xi-xj|^2 = (xj-xi)/|xi-xj|^2
    
    # forces[i] = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # Note: diff[i,j] = xi - xj, so (xj - xi) = -diff[i,j]
    forces = np.sum(-diff / dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: grad_tangent = forces - (forces . xi) * xi
    dot_products = np.sum(forces * pts, axis=1)
    proj_forces = forces - dot_products[:, np.newaxis] * pts
    
    return energy, proj_forces

def _riemannian_gd(pts, max_iters=5000, lr_init=0.05, lr_min=1e-6, time_limit=60.0):
    """
    Riemannian gradient descent with adaptive step size and momentum.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    lr = lr_init
    momentum = 0.9
    velocity = np.zeros_like(pts)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Momentum update
        velocity = momentum * velocity + grad
        update = velocity
        
        # Adaptive step size: try to take a step that reduces energy
        step = lr
        for _ in range(10):
            new_pts = pts - step * update
            new_pts = _normalize(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                break
            step *= 0.5
        
        new_pts = _normalize(pts - step * update)
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
            lr = min(lr * 1.02, 0.1)
        else:
            lr *= 0.7
            if lr < lr_min:
                break
    
    return best_pts, best_energy

def _lbfgs_optimize(pts, time_limit=30.0):
    """
    Polish with L-BFGS-B in 3D coordinates with normalization.
    """
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return float('inf')
        p = _normalize(x.reshape(N, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient_func(x):
        p = _normalize(x.reshape(N, 3))
        _, g = _energy_and_grad(p)
        return g.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient_func,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < float('inf'):
            res_pts = _normalize(result.x.reshape(N, 3))
            return res_pts, result.fun
    except Exception:
        pass
    
    return pts, float('inf')

def _simulated_annealing(pts, time_limit=40.0):
    """
    Simulated annealing with random rotations to escape local minima.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    T = 1.0  # initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    current_pts = pts.copy()
    current_energy, _ = _energy_and_grad(current_pts)
    
    while time.time() - start_time < time_limit and T > T_min:
        # Random perturbation: small random rotation
        angle = np.random.uniform(0.1, 0.5)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        
        # Rodrigues rotation
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        new_pts = (R @ current_pts.T).T
        new_pts = _normalize(new_pts)
        new_energy, _ = _energy_and_grad(new_pts)
        
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.random() < math.exp(-delta_E / T):
            current_pts = new_pts
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
        
        T *= cooling_rate
    
    return best_pts, best_energy

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # Try multiple initializations
    initializations = [
        _fibonacci_sphere(N),
        _halton_sphere(N, 2),
        _halton_sphere(N, 3),
    ]
    
    best_pts = None
    best_energy = float('inf')
    
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        
        # Phase 1: Riemannian gradient descent
        gd_pts, gd_energy = _riemannian_gd(
            init_pts,
            max_iters=3000,
            lr_init=0.05,
            lr_min=1e-6,
            time_limit=20.0
        )
        
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_pts = gd_pts.copy()
        
        # Phase 2: Simulated annealing to escape local minima
        if time.time() > deadline - 25.0:
            break
        
        sa_pts, sa_energy = _simulated_annealing(
            gd_pts,
            time_limit=15.0
        )
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_pts.copy()
        
        # Phase 3: Polish with L-BFGS
        if time.time() > deadline - 10.0:
            break
        
        lbfgs_pts, lbfgs_energy = _lbfgs_optimize(
            sa_pts,
            time_limit=8.0
        )
        
        if lbfgs_energy < best_energy:
            best_energy = lbfgs_energy
            best_pts = lbfgs_pts.copy()
    
    # Final polish with L-BFGS on the best found
    if best_pts is not None and time.time() < deadline - 5.0:
        final_pts, final_energy = _lbfgs_optimize(best_pts, time_limit=5.0)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final verification
    final_pts = _normalize(best_pts)
    return final_pts
