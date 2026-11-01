"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i (in R^3) = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    We project gradient to tangent space: grad_tan = grad - (grad . x_i) * x_i
    """
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)  # avoid div by zero
    dist = np.sqrt(dist2)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = sum_j diff[i,j] / dist2[i,j]
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project to tangent space
    grad_dot_x = np.sum(grad * pts, axis=1)  # (n,)
    grad_tan = grad - grad_dot_x[:, np.newaxis] * pts  # (n, 3)
    
    return energy, grad_tan


def _retract(pts, step):
    """Retract from tangent space back to sphere.
    pts: (n, 3) on sphere
    step: (n, 3) tangent vectors
    Returns normalized pts + step.
    """
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1.0, norms)
    return new_pts / norms


def _line_search(pts, grad_tan, energy, max_iter=50):
    """Backtracking line search in tangent space.
    Returns step size and new points.
    """
    n = len(pts)
    # Normalize gradient direction for stable step sizing
    grad_norms = np.linalg.norm(grad_tan, axis=1, keepdims=True)
    grad_norms = np.where(grad_norms < 1e-15, 1.0, grad_norms)
    direction = -grad_tan / grad_norms  # unit direction
    
    # Initial step size based on typical gradient magnitude
    avg_grad_norm = np.mean(np.linalg.norm(grad_tan, axis=1))
    step_size = min(1.0, max(0.01, 1.0 / (avg_grad_norm + 1.0)))
    
    for _ in range(max_iter):
        new_pts = _retract(pts, step_size * direction)
        new_energy, _ = _energy_and_grad(new_pts)
        
        if np.isfinite(new_energy) and new_energy < energy - 1e-12 * step_size * avg_grad_norm:
            return step_size, new_pts, new_energy
        
        step_size *= 0.5
        if step_size < 1e-10:
            break
    
    return step_size, new_pts, new_energy


def _fibonacci_sphere():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _delaunay_smooth(pts, lam=0.1, iters=50):
    """Apply spectral Laplacian flow using Delaunay triangulation.
    
    This acts as a low-pass filter on the point distribution,
    enforcing global geometric consistency.
    """
    for _ in range(iters):
        try:
            tri = Delaunay(pts)
            # Build adjacency from Delaunay
            adj = np.zeros((N, N), dtype=bool)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            adj[simplex[i], simplex[j]] = True
        except Exception:
            break
        
        # Laplacian flow: move each point toward average of neighbors
        neighbor_sum = np.zeros_like(pts)
        for i in range(N):
            neighbors = np.where(adj[i])[0]
            if len(neighbors) > 0:
                neighbor_sum[i] = np.mean(pts[neighbors], axis=0)
        
        # Blend: new = (1-lam)*old + lam*neighbor_avg
        pts = (1.0 - lam) * pts + lam * neighbor_sum
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        pts = pts / norms
    
    return pts


def _optimize_from_start(pts, time_budget):
    """Optimize points from a starting configuration within time budget."""
    start_time = time.time()
    
    # First, apply some Delaunay smoothing
    pts = _delaunay_smooth(pts, lam=0.05, iters=20)
    
    energy, grad_tan = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Riemannian gradient descent with line search
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_budget:
        if no_improve_count > max_no_improve:
            break
        
        step_size, new_pts, new_energy = _line_search(pts, grad_tan, energy)
        
        if not np.isfinite(new_energy):
            break
        
        if new_energy < energy - 1e-12:
            pts = new_pts
            energy = new_energy
            no_improve_count = 0
            
            if energy < best_energy:
                best_pts = pts.copy()
                best_energy = energy
        else:
            no_improve_count += 1
        
        # Recompute gradient
        energy, grad_tan = _energy_and_grad(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-8:
            break
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    total_budget = 140.0  # Leave 10s safety margin for 150s limit
    
    # Multiple restarts with different initializations
    candidates = []
    
    # Fibonacci spiral start
    fib_pts = _fibonacci_sphere()
    candidates.append(fib_pts)
    
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.standard_normal((N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        candidates.append(pts)
    
    # Optimize each candidate with allocated time
    time_per_candidate = (total_budget - 5.0) / len(candidates)  # 5s reserved for final polish
    
    best_pts = None
    best_energy = np.inf
    
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 5.0:
            break
        
        alloc_time = min(time_per_candidate, remaining - 5.0)
        if alloc_time < 1.0:
            break
        
        pts, energy = _optimize_from_start(init_pts, alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final high-precision polish pass on the best candidate
    if best_pts is not None:
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining > 2.0:
            # Do a final round of optimization with tighter tolerance
            pts = best_pts
            energy, grad_tan = _energy_and_grad(pts)
            
            # More iterations with smaller steps
            for _ in range(100):
                if time.time() - start_time > total_budget - 1.0:
                    break
                
                step_size, new_pts, new_energy = _line_search(pts, grad_tan, energy, max_iter=30)
                
                if not np.isfinite(new_energy) or new_energy >= energy - 1e-14:
                    break
                
                pts = new_pts
                energy = new_energy
                _, grad_tan = _energy_and_grad(pts)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
    
    # Final validation and normalization
    if best_pts is None:
        best_pts = fib_pts
    
    # Ensure all points are on unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms
    
    # Check for points too close together
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    if np.min(dist2) < 1e-18:  # distance < 1e-9
        # Perturb the closest pair
        min_idx = np.unravel_index(np.argmin(dist2), dist2.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        # Move point i slightly
        direction = rng.standard_normal(3)
        direction = direction - np.dot(direction, best_pts[i]) * best_pts[i]
        direction /= np.linalg.norm(direction)
        best_pts[i] = best_pts[i] + 1e-8 * direction
        norms_i = np.linalg.norm(best_pts[i])
        best_pts[i] = best_pts[i] / norms_i
    
    return best_pts

# EVOLVE-BLOCK-END
