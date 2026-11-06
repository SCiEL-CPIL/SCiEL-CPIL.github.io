"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
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
    
    energy = math.fsum(np.log(1.0 / d_upper).tolist())
    
    # Gradient of sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    # d/dx_i E = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # weights_{ij} = 1/d_{ij}^2 for i != j, 0 for i=j
    weights = 1.0 / d2
    np.fill_diagonal(weights, 0.0)
    
    sum_weights = np.sum(weights, axis=1) # shape (n,)
    weighted_sum = weights @ pts # shape (n, 3)
    
    grad = -pts * sum_weights[:, np.newaxis] + weighted_sum
    
    # Project gradient to tangent space
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - radial * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _repulsive_force(pts):
    """
    Compute repulsive forces based on inverse square law (Coulomb-like).
    This is a surrogate for the gradient of log energy but with stronger 
    long-range behavior to help escape local minima and spread points.
    F_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^3
    Note: This is the force that would minimize sum 1/d.
    However, for log energy, the force is (x_i - x_j) / d^2.
    Using 1/d^3 (stronger repulsion at short range) or 1/d^2 (log gradient) 
    are both valid. We will use a mix or just the log gradient but with 
    larger steps in SA to escape.
    Actually, the gradient of log energy IS the repulsive force 1/d^2.
    So we can just use the gradient, but SA allows uphill moves.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    # Force direction: (x_i - x_j) / d^3 * d = (x_i - x_j) / d^2
    # This is exactly the gradient term.
    # We return the negative gradient (force pushing points apart)
    weights = 1.0 / d2
    np.fill_diagonal(weights, 0.0)
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ pts
    grad = -pts * sum_weights[:, np.newaxis] + weighted_sum
    
    # Project to tangent
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - radial * pts
    
    return grad_tangent

def _simulated_annealing(pts0, deadline, initial_temp=0.1, cooling=0.995):
    """
    Simulated Annealing on the sphere.
    Uses the negative gradient as the direction of improvement.
    Allows uphill moves with probability exp(-deltaE / T).
    """
    pts = pts0.copy()
    energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    temp = initial_temp
    rng = np.random.RandomState(42)
    
    # Pre-compute number of iterations we can afford
    # Each iteration involves energy calc (O(N^2)) and gradient (O(N^2)).
    # N=100, so 10k ops per eval. Very fast.
    # We can do ~100k iterations in 1s.
    
    while time.time() < deadline:
        # Generate a random perturbation in tangent space
        # Scale perturbation with temperature
        noise_scale = 0.01 * temp
        noise = rng.randn(N, 3) * noise_scale
        
        # Project noise to tangent space
        radial_noise = np.sum(noise * pts, axis=1, keepdims=True)
        noise_tangent = noise - radial_noise * pts
        
        new_pts = _project_to_sphere(pts + noise_tangent)
        new_energy = _log_energy(new_pts)
        
        delta_e = new_energy - energy
        
        if delta_e < 0 or (temp > 1e-6 and rng.rand() < math.exp(-delta_e / temp)):
            pts = new_pts
            energy = new_energy
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        temp *= cooling
        
        # If temperature is too low, we are effectively doing gradient descent
        # with very small steps. We can break and let GD take over.
        if temp < 1e-4:
            break
            
    return best_pts, best_energy

def _gradient_descent_polish(pts0, deadline, max_iter=2000):
    """
    High-precision Gradient Descent with adaptive step size.
    """
    pts = pts0.copy()
    energy, grad = _log_energy_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    step_size = 1e-3
    min_step = 1e-8
    max_step = 1e-1
    patience = 50
    no_improve_count = 0
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute gradient
        energy, grad = _log_energy_grad(pts)
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-10:
            break
            
        # Line search (backtracking)
        alpha = step_size
        found = False
        for _ in range(20):
            new_pts = _project_to_sphere(pts - alpha * grad)
            new_energy = _log_energy(new_pts)
            
            if new_energy < energy - 1e-12: # Strict improvement
                found = True
                break
            alpha *= 0.5
            
        if found:
            pts = new_pts
            energy = new_energy
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Increase step size if we made good progress
            if no_improve_count == 0:
                step_size = min(max_step, step_size * 1.1)
        else:
            no_improve_count += 1
            step_size = max(min_step, step_size * 0.5)
            
        if no_improve_count > patience:
            break
            
    return best_pts, best_energy

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

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    # Strategy:
    # 1. Run Simulated Annealing from a few diverse starts to find good basins.
    #    SA is good at escaping local minima due to uphill moves.
    # 2. Polish the best SA results with high-precision Gradient Descent.
    
    # Start 1: Fibonacci (good initial distribution)
    # Start 2: Random
    # Start 3: Random
    
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2)
    ]
    
    # Allocate time to SA
    # We have ~140s.
    # Let's give each SA run ~30s.
    sa_deadline = start_time + 90.0
    
    sa_results = []
    
    for i, pts0 in enumerate(starts):
        if time.time() > sa_deadline:
            break
            
        # Run SA
        # Initial temp should be high enough to move points significantly
        # but not so high that energy explodes.
        # Log energy for 100 points is around -1000.
        # Typical delta E per step is small.
        # Let's try T=0.1
        pts_sa, e_sa = _simulated_annealing(pts0, sa_deadline - 10, initial_temp=0.5, cooling=0.999)
        
        if e_sa < best_energy:
            best_energy = e_sa
            best_pts = pts_sa
            
        sa_results.append((e_sa, pts_sa))
        
    # Sort SA results by energy
    sa_results.sort(key=lambda x: x[0])
    
    # Phase 2: Polish the top candidates with Gradient Descent
    # We have ~50s left.
    gd_deadline = deadline - 5.0
    
    # Polish the best SA result
    if sa_results:
        e_best, pts_best = sa_results[0]
        pts_gd, e_gd = _gradient_descent_polish(pts_best, gd_deadline, max_iter=5000)
        
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = pts_gd
            
        # If time permits, polish the second best SA result too
        if len(sa_results) > 1 and time.time() < gd_deadline - 10:
            e_second, pts_second = sa_results[1]
            # Only if it's close to the best
            if e_second < best_energy + 1.0:
                pts_gd2, e_gd2 = _gradient_descent_polish(pts_second, gd_deadline - 5, max_iter=2000)
                if e_gd2 < best_energy:
                    best_energy = e_gd2
                    best_pts = pts_gd2

    # Final check
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
        
    return best_pts
# EVOLVE-BLOCK-END
