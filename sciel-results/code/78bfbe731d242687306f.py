"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

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
    """Compute logarithmic energy and gradient on sphere."""
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
    
    weights = np.zeros_like(d2)
    np.fill_diagonal(weights, 0)
    weights = 1.0 / d2
    grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
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

def _md_optimize(pts0, deadline):
    """Molecular dynamics with velocity Verlet and simulated annealing."""
    n = len(pts0)
    pts = pts0.copy()
    v = np.zeros_like(pts)
    
    # Initial temperature
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.999
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Precompute some constants
    dt = 0.01
    
    # Initial force
    _, F = _log_energy_grad(pts)
    
    steps = 0
    max_steps = 20000
    
    while steps < max_steps:
        if time.time() > deadline:
            break
            
        # Velocity Verlet integration
        # 1. Update positions
        pts_new = pts + v * dt + 0.5 * F * (dt * dt)
        
        # 2. Project to sphere
        pts_new = _project_to_sphere(pts_new)
        
        # 3. Compute new forces
        e_new, F_new = _log_energy_grad(pts_new)
        
        # 4. Update velocities
        v_new = 0.9 * v + 0.5 * (F + F_new) * dt  # 0.9 is friction/damping
        
        # Simulated annealing: add random kicks based on temperature
        if T > T_min:
            rng = np.random.RandomState(steps)
            kicks = rng.randn(n, 3) * math.sqrt(T) * 0.1
            v_new += kicks
            
            # Cool down
            T *= cooling_rate
        
        # Check energy change for acceptance (Metropolis-like, but simplified)
        e_old = _log_energy(pts)
        
        # Accept if better, or with probability exp(-(E_new - E_old)/T) if worse
        if e_new < e_old or (T > T_min and np.random.rand() < math.exp(-(e_new - e_old) / T)):
            pts = pts_new
            v = v_new
            F = F_new
        else:
            # Revert
            pass
        
        # Track best
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts_new.copy()
            
        steps += 1
        
        # Convergence check
        if np.linalg.norm(F) < 1e-10 and T < 0.01:
            break
            
    return best_pts, best_energy

def _gradient_refine(pts0, deadline):
    """Short gradient descent refinement."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = 0.01
    momentum = 0.9
    
    for it in range(2000):
        if time.time() > deadline:
            break
            
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
            
        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.7
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
    
    # Start 1: Fibonacci spiral with MD
    pts0 = _fibonacci_sphere(N)
    pts_opt, e_opt = _md_optimize(pts0, deadline)
    if e_opt < best_energy:
        best_energy = e_opt
        best_pts = pts_opt
    
    # Refine best so far with gradient
    if best_pts is not None and time.time() < deadline - 10:
        pts_ref, e_ref = _gradient_refine(best_pts, deadline)
        if e_ref < best_energy:
            best_energy = e_ref
            best_pts = pts_ref
    
    # Start 2: Random points with MD
    if time.time() < deadline - 15:
        pts0 = _random_sphere(N, seed=42)
        pts_opt, e_opt = _md_optimize(pts0, deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
            
        # Refine
        if best_pts is not None and time.time() < deadline - 5:
            pts_ref, e_ref = _gradient_refine(best_pts, deadline)
            if e_ref < best_energy:
                best_energy = e_ref
                best_pts = pts_ref

    # Start 3: Another random with MD
    if time.time() < deadline - 15:
        pts0 = _random_sphere(N, seed=123)
        pts_opt, e_opt = _md_optimize(pts0, deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
            
        # Refine
        if best_pts is not None and time.time() < deadline - 5:
            pts_ref, e_ref = _gradient_refine(best_pts, deadline)
            if e_ref < best_energy:
                best_energy = e_ref
                best_pts = pts_ref

    return best_pts
# EVOLVE-BLOCK-END
