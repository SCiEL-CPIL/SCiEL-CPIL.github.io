"""Improved logarithmic energy minimizer for 100 points on the sphere."""
import numpy as np
import time
import math

N = 100


def _energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/d_ij)."""
    # points: (N, 3)
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Zero out diagonal and take upper triangle
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Avoid log(0) - though points shouldn't be identical
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))  # sum of log(1/d) = -sum log(d)


def _energy_and_grad(points):
    """Compute energy and gradient on the sphere (projected gradient).
    
    The gradient of E = sum_{i<j} log(1/|x_i - x_j|) w.r.t. x_i is:
    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    We project to the tangent space: grad_tangent = grad - (grad . x_i) * x_i
    """
    # Pairwise differences and distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dist, 1.0)  # avoid division by zero on diagonal
    
    # Gradient: for point i, grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # weight[i,j] = 1 / dist[i,j]^2
    weight = 1.0 / (dist * dist)  # (N, N)
    np.fill_diagonal(weight, 0.0)
    
    # grad[i] = sum_j weight[i,j] * diff[i,j]
    grad = np.einsum('ij,ijc->ic', weight, diff)
    
    # Project to tangent space: grad_t = grad - (grad . x_i) x_i
    dot = np.sum(grad * points, axis=1, keepdims=True)  # (N, 1)
    grad_tangent = grad - dot * points  # (N, 3)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    return energy, grad_tangent


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral with offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent(points, max_iters=5000, tol=1e-12, deadline=None):
    """Gradient descent with adaptive step size and momentum on the sphere."""
    points = points.copy().astype(np.float64)
    best_points = points.copy()
    best_energy = _energy(points)
    
    # Momentum
    v = np.zeros_like(points)
    momentum = 0.9
    lr = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Adaptive step size: backtracking line search
        # Try to decrease energy
        step = lr
        for _ in range(20):
            new_points = points + step * (v + grad)
            # Project back to sphere
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            new_points = new_points / norms
            
            new_energy = _energy(new_points)
            if new_energy < energy:
                break
            step *= 0.5
        
        # Update momentum
        v = momentum * v + grad
        points = new_points
        
        # Decay learning rate slowly
        if it % 500 == 0 and it > 0:
            lr *= 0.95
        
        # Convergence check
        if it > 100 and abs(energy - best_energy) < tol and np.max(np.abs(grad)) < 1e-10:
            break
    
    return best_points, best_energy


def _local_polish(points, max_iters=2000, deadline=None):
    """High-precision local polish using Newton-like method with small steps."""
    points = points.copy().astype(np.float64)
    best_points = points.copy()
    best_energy = _energy(points)
    
    lr = 0.001
    v = np.zeros_like(points)
    momentum = 0.8
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Backtracking line search
        step = lr
        new_points = points
        for _ in range(30):
            candidate = points + step * (v + grad)
            norms = np.linalg.norm(candidate, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            candidate = candidate / norms
            
            cand_energy = _energy(candidate)
            if cand_energy < energy:
                new_points = candidate
                break
            step *= 0.5
        
        v = momentum * v + grad
        points = new_points
        
        if it % 200 == 0 and it > 0:
            lr *= 0.9
    
    return best_points, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Multiple starting configurations
    starts = []
    
    # Fibonacci spiral (base)
    starts.append(_fibonacci_spiral(N, 0.0))
    
    # Perturbed Fibonacci spirals
    np.random.seed(42)
    for k in range(5):
        pts = _fibonacci_spiral(N, float(k) * 0.1)
        # Add small random perturbation
        noise = np.random.randn(N, 3) * 0.01
        pts = pts + noise
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        starts.append(pts)
    
    # Sobol-like sequence (using Halton sequence as approximation)
    def halton_sequence(n, base):
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0
            r = 0.0
            j = i + 1
            while j > 0:
                f /= base
                r += f * (j % base)
                j //= base
            seq[i] = r
        return seq
    
    # Generate points using Halton in 2D, map to sphere
    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    # Map to spherical coordinates
    theta = 2 * np.pi * h1
    phi = np.arccos(1 - 2 * h2)
    pts_h = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    starts.append(pts_h)
    
    # Run gradient descent from each start
    best_points = None
    best_energy = np.inf
    
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Phase 1: Coarse gradient descent
        pts, e = _gradient_descent(start_pts, max_iters=3000, tol=1e-10, deadline=deadline)
        
        if e < best_energy:
            best_energy = e
            best_points = pts.copy()
        
        # Phase 2: Fine polish
        pts2, e2 = _local_polish(pts, max_iters=1000, deadline=deadline)
        
        if e2 < best_energy:
            best_energy = e2
            best_points = pts2.copy()
    
    # Final verification and normalization
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_points = best_points / norms
    
    # Final check: no points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dist[iu]) < 1e-9:
        # Fallback to Fibonacci spiral if points are too close
        best_points = _fibonacci_spiral(N)
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / np.maximum(norms, 1e-15)
    
    return best_points
