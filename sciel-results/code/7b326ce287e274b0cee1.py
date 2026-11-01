import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _energy_and_grad_sphere(pts):
    """
    Compute energy and Riemannian gradient (tangent space) for points on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy (float), grad (N, 3) array of tangent vectors.
    """
    n = len(pts)
    # Compute pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero on diagonal
    
    # inv_d2[i,j] = 1 / |x_i - x_j|^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient of E = sum log(1/d) = - sum log(d)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = - x_i * sum_{j!=i} inv_d2[i,j] + sum_{j!=i} inv_d2[i,j] * x_j
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _lbfgs_sphere(pts, time_limit=10.0, penalty_lambda=100.0):
    """
    Optimize points on the sphere using L-BFGS-B in ambient space.
    Uses a penalty term to enforce the sphere constraint and the tangent gradient
    for the energy term.
    """
    start = time.time()
    x0 = pts.flatten().copy()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        
        pts_x = x.reshape(N, 3)
        # Normalize to get points on sphere for energy calculation
        pts_norm = _normalize(pts_x)
        
        # Compute energy and tangent gradient on the sphere
        energy, grad_tan = _energy_and_grad_sphere(pts_norm)
        
        # Compute penalty term and gradient
        # Penalty: lambda * sum(||x_i||^2 - 1)^2
        norms_sq = np.sum(pts_x * pts_x, axis=1)
        penalty = penalty_lambda * np.sum((norms_sq - 1.0) ** 2)
        
        # Gradient of penalty: d/dx lambda * (||x||^2 - 1)^2 
        # = lambda * 2 * (||x||^2 - 1) * 2x = 4 * lambda * (||x||^2 - 1) * x
        grad_penalty = 4.0 * penalty_lambda * (norms_sq - 1.0)[:, np.newaxis] * pts_x
        
        # Total gradient: Tangent gradient (projected) + Penalty gradient
        # Note: grad_tan is orthogonal to pts_norm. 
        # We add it to the ambient gradient. This is an approximation that works well
        # because the penalty term handles the radial correction.
        total_grad = grad_tan + grad_penalty
        
        return energy + penalty, total_grad.flatten()

    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 1000,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxfun': 5000,
            'maxls': 50
        }
    )

    best_pts = _normalize(result.x.reshape(N, 3))
    best_energy = _log_energy_fast(best_pts)
    
    return best_pts, best_energy

def _fps_fast(n, seed=42, batch_size=2000):
    """Fast Farthest Point Sampling."""
    rng = np.random.default_rng(seed)
    # Start with a random point
    first_pt = rng.standard_normal(3)
    first_pt /= np.linalg.norm(first_pt)
    pts = np.array([first_pt])

    for _ in range(n - 1):
        # Generate batch of candidates
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)

        # Compute min distance from each candidate to existing points
        min_dists = np.full(batch_size, np.inf)
        # Chunked computation to manage memory
        chunk_size = 100
        for i in range(0, len(pts), chunk_size):
            chunk_pts = pts[i:i+chunk_size]
            # (batch, chunk, 3)
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))

        best_idx = np.argmax(min_dists)
        pts = np.vstack([pts, candidates[best_idx:best_idx+1]])

    return _normalize(pts)

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _basin_hopping(pts, time_limit=20.0, max_restarts=15):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left_init = time_limit - (time.time() - start) - 2
    if time_left_init > 5:
         current, energy = _lbfgs_sphere(current, time_limit=min(10, time_left_init))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.5
    decay = 0.9
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2:
            break
            
        # Adaptive perturbation
        # Use a mix of small and large perturbations
        if np.random.rand() < 0.2:
            scale = 0.5 * temp
        else:
            scale = 0.1 * temp
            
        rng = np.random.default_rng(rh * 12345 + 678)
        perturbations = rng.standard_normal((N, 3)) * scale
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        # Apply perturbation (approximate exponential map for small steps)
        new_pts = current + perturbations
        new_pts = _normalize(new_pts)
        
        # Optimize
        time_left = time_limit - (time.time() - start) - 1
        if time_left < 3:
            break
            
        new_opt_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=min(8, time_left))
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(temp, 1e-6)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        temp *= decay
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    
    # FPS with different seeds
    for s in [42, 123, 999]:
        inits.append(_fps_fast(N, seed=s))
        
    # Random inits
    for s in [1, 2, 3]:
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    # We have ~6 inits. Total time 145s.
    # Allocate ~20s per init for basin hopping (which includes L-BFGS).
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=20.0)
            opt_energy = _log_energy_fast(opt_pts)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish with L-BFGS
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 2
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
