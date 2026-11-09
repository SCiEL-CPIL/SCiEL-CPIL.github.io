"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (in R^3, not projected)."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    np.fill_diagonal(dist, 1.0)  # avoid div by zero
    
    # Energy = sum_{i<j} log(1/dist_ij) = -0.5 * sum_{i≠j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, 1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = -sum_{j≠i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j≠i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    # grad[i] = sum_j (pts[j] - pts[i]) / dist[i,j]^2
    # = -sum_j diff[i,j] / dist[i,j]^2 where diff[i,j] = pts[i] - pts[j]
    # diff[i,j] = pts[i] - pts[j]
    # So grad[i] = -sum_j diff[i,j] * inv_dist2[i,j]
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _tangent_project(pts, vec):
    """Project vec to tangent space of sphere at pts."""
    dot = np.sum(vec * pts, axis=1)
    return vec - dot[:, np.newaxis] * pts

def _line_search(pts, direction, energy, grad, max_iter=50):
    """Backtracking line search in tangent space."""
    step = 0.01
    # Approximate curvature bound
    L = 1.0
    for _ in range(max_iter):
        # Move in tangent direction and renormalize
        new_pts = pts + step * direction
        norms = np.sqrt(np.sum(new_pts * new_pts, axis=1))
        new_pts = new_pts / norms[:, np.newaxis]
        
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy - 1e-4 * step * np.sum(grad * direction):
            return new_pts, new_energy, step
        step *= 0.5
        if step < 1e-15:
            break
    
    return pts, energy, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    energy, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Conjugate gradient in tangent space
    p = _tangent_project(pts, -grad)
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Check convergence
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
        
        # Line search
        new_pts, new_energy, step = _line_search(pts, p, energy, grad)
        
        if step == 0.0:
            # Line search failed, reset direction
            p = _tangent_project(pts, -grad)
            continue
        
        # Compute new gradient
        new_grad = _energy_and_grad(new_pts)[1]
        
        # Update conjugate direction (Polak-Ribiere)
        if it > 0:
            beta = max(0, np.sum(new_grad * (new_grad - grad)) / (np.sum(grad * grad) + 1e-30))
            p_new = _tangent_project(new_pts, -new_grad + beta * p)
        else:
            p_new = _tangent_project(new_pts, -new_grad)
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
        p = p_new
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check for stagnation
        if it > 100 and abs(energy - best_energy) < 1e-12:
            # Try a small random perturbation to escape local minimum
            if time.time() < deadline - 10.0:
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    noise = np.random.randn(N, 3) * 0.01
                    perturbed = pts + _tangent_project(pts, noise)
                    norms = np.sqrt(np.sum(perturbed * perturbed, axis=1))
                    perturbed = perturbed / norms[:, np.newaxis]
                    e_pert, g_pert = _energy_and_grad(perturbed)
                    if e_pert < best_energy:
                        pts = perturbed
                        energy = e_pert
                        grad = g_pert
                        p = _tangent_project(pts, -grad)
                        break
    
    # Final polish: more iterations with tighter tolerance if time remains
    if time.time() < deadline - 10.0:
        p = _tangent_project(best_pts, -_energy_and_grad(best_pts)[1])
        for it in range(2000):
            if time.time() > deadline - 5.0:
                break
            new_pts, new_energy, step = _line_search(best_pts, p, best_energy, _energy_and_grad(best_pts)[1])
            if step == 0.0:
                break
            new_grad = _energy_and_grad(new_pts)[1]
            if it > 0:
                beta = max(0, np.sum(new_grad * (new_grad - grad)) / (np.sum(grad * grad) + 1e-30))
                p = _tangent_project(new_pts, -new_grad + beta * p)
            else:
                p = _tangent_project(new_pts, -new_grad)
            best_pts = new_pts
            best_energy = new_energy
            grad = new_grad
    
    # Ensure points are on unit sphere
    norms = np.sqrt(np.sum(best_pts * best_pts, axis=1))
    best_pts = best_pts / norms[:, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
