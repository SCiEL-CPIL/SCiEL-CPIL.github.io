"""Optimize 100 points on unit sphere to minimize logarithmic energy using Riemannian Basin Hopping."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its Riemannian gradient efficiently.
    Energy: sum_{i<j} log(1/||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    Gradient (Euclidean): grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    Riemannian Gradient: project onto tangent space: grad_tan = grad - <grad, x_i> x_i
    """
    n = len(pts)
    # Compute pairwise differences: (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Compute squared distances: (n, n)
    dist_sq = np.sum(diff * diff, axis=-1)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Distances: (n, n)
    dist = np.sqrt(dist_sq)
    
    # Energy terms: -log(dist) for i < j
    # We sum over all i != j and divide by 2, or just sum upper triangle
    # For speed, sum all i!=j and divide by 2.
    # log(1/d) = -log(d)
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Energy calculation
    # To be precise with math.fsum as per contract, we extract terms
    # But for optimization speed, we use np.sum first.
    # The contract says "Score: ... math.fsum". The evaluator uses math.fsum.
    # Our internal optimizer can use np.sum for speed, but the final return value
    # is scored by the evaluator's fsum.
    # However, for comparison during optimization, np.sum is sufficient.
    
    # Vectorized energy
    log_dists = -np.log(dist)
    np.fill_diagonal(log_dists, 0.0)
    energy = 0.5 * np.sum(log_dists)
    
    # Gradient calculation
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # diff[i, j] = x_i - x_j
    # dist_sq[i, j] = ||x_i - x_j||^2
    # We want sum over j of diff[i, j] / dist_sq[i, j]
    
    # Avoid division by zero
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i, j] * inv_dist_sq[i, j]
    grad = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    
    # Project to tangent space
    # <grad_i, x_i>
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - dot * pts
    
    return energy, grad_tan

def _retract_sphere(pts, grad):
    """
    Geodesic retraction on the sphere.
    p_new = sin(||g||)/||g|| * g + cos(||g||) * p
    """
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    # Handle zero norms
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    sin_norms = np.sin(safe_norms)
    cos_norms = np.cos(safe_norms)
    
    # If norm is 0, sin(norm)/norm -> 1, cos(norm) -> 1. 
    # Actually sin(0)/0 is undefined, limit is 1.
    # So p_new = 1 * 0 + 1 * p = p. Correct.
    
    pts_new = (sin_norms / safe_norms) * grad + cos_norms * pts
    return pts_new

def _local_minimize(pts, max_iter=100, time_limit=5.0):
    """
    Riemannian Gradient Descent with backtracking line search.
    Returns (pts, energy).
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy, current_grad = _energy_and_grad(current_pts)
    
    lr = 0.1
    min_lr = 1e-8
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        grad_norm = np.linalg.norm(current_grad)
        if grad_norm < 1e-8:
            break
            
        # Direction: negative gradient
        direction = -current_grad
        
        # Backtracking line search
        alpha = lr
        best_alpha = alpha
        best_energy = current_energy
        best_pts = current_pts
        
        for _ in range(20):
            new_pts = _retract_sphere(current_pts, alpha * direction)
            new_energy, _ = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
            else:
                alpha *= 0.5
                if alpha < min_lr:
                    break
        
        # Update
        if best_energy < current_energy:
            current_pts = best_pts
            current_energy = best_energy
            _, current_grad = _energy_and_grad(current_pts)
        else:
            # If no improvement, reduce learning rate
            lr *= 0.5
            if lr < min_lr:
                break
                
    return current_pts, current_energy

def _basin_hopping(pts, n_iterations=50, time_budget=100.0):
    """
    Riemannian Basin Hopping.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy, _ = _energy_and_grad(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    T = 1.0
    T_final = 0.01
    cool_rate = (T_final / T) ** (1.0 / n_iterations)
    
    rng = np.random.RandomState(42)
    
    for i in range(n_iterations):
        if time.time() - start_time > time_budget:
            break
            
        # 1. Random perturbation
        # Perturb in tangent space
        perturb = rng.randn(N, 3)
        # Project to tangent space
        dot = np.sum(perturb * current_pts, axis=1, keepdims=True)
        perturb_tan = perturb - dot * current_pts
        
        # Scale perturbation
        # Typical distance between points is ~1/sqrt(N) ~ 0.1
        # Perturbation magnitude should be comparable
        scale = 0.5 
        new_pts = _retract_sphere(current_pts, scale * perturb_tan)
        
        # 2. Local minimization of perturbed point
        new_pts_opt, new_energy = _local_minimize(new_pts, max_iter=50, time_limit=2.0)
        
        # 3. Acceptance criterion
        delta_E = new_energy - current_energy
        accept = False
        if delta_E < 0:
            accept = True
        else:
            # Metropolis criterion
            if rng.rand() < math.exp(-delta_E / T):
                accept = True
        
        if accept:
            current_pts = new_pts_opt
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        
        # Cool down
        T *= cool_rate
        
    return best_pts, best_energy

def _sobol_sphere(n, seed=0):
    """Generate points using Sobol sequence mapped to sphere."""
    # Use qmc.Sobol for quasi-random points in [0,1]^3
    # We need 2 dimensions for sphere (theta, phi) or 3 for rejection
    # Let's use 2 dimensions: u1 for z (cos theta), u2 for phi
    sampler = qmc.Sobol(d=2, seed=seed)
    # Number of points: power of 2 for best results
    m = 10
    while (1 << m) < n:
        m += 1
    samples = sampler.random(1 << m)
    samples = samples[:n]
    
    u1 = samples[:, 0]
    u2 = samples[:, 1]
    
    # Map to sphere
    # z in [-1, 1]
    z = 2.0 * u1 - 1.0
    # phi in [0, 2pi]
    phi = 2.0 * np.pi * u2
    
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    return np.column_stack([x, y, z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Strategy: 
    # 1. Start with a good initial guess (Sobol)
    # 2. Run Basin Hopping from this start
    # 3. If time remains, try a few other starts (Fibonacci, Random)
    
    initializers = [
        _sobol_sphere,
        lambda n: _fibonacci_sphere(n),
        lambda n: _random_sphere(n, seed=1),
        lambda n: _random_sphere(n, seed=2),
    ]
    
    # Allocate time budget
    # Basin hopping is the main workhorse.
    # We will run BH on the first initializer with most of the time.
    # Then quick local minimization on others.
    
    # First initializer: Sobol
    if time.time() < deadline:
        try:
            start_pts = initializers[0](N)
            # Quick local minimization first to get a good basin
            pts1, e1 = _local_minimize(start_pts, max_iter=100, time_limit=10.0)
            
            # Basin Hopping
            # Estimate how many iterations we can afford
            # Each iteration takes ~2-5s for local minimization
            # Let's give 80s to BH
            bh_time = min(80.0, deadline - time.time())
            if bh_time > 10.0:
                pts2, e2 = _basin_hopping(pts1, n_iterations=100, time_budget=bh_time)
            else:
                pts2, e2 = pts1, e1
            
            if e2 < best_energy:
                best_energy = e2
                best_pts = pts2
                
        except Exception as e:
            pass
    
    # Other initializers: Quick local minimization
    for init_func in initializers[1:]:
        if time.time() > deadline - 15.0:
            break
            
        try:
            start_pts = init_func(N)
            pts, e = _local_minimize(start_pts, max_iter=100, time_limit=10.0)
            
            if e < best_energy:
                best_energy = e
                best_pts = pts
                
        except Exception:
            continue
            
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
        
    return best_pts

def _fibonacci_sphere(n):
    """Generate Fibonacci lattice points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

# EVOLVE-BLOCK-END
