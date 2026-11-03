import numpy as np
import time
import math

N = 150

def _energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient on the sphere.
    points: (N, 3) array of points on the unit sphere.
    Returns: energy (float), grad (N, 3) array of gradients projected to tangent space.
    """
    # Pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Squared distances
    sq_dist = np.sum(diff ** 2, axis=2)
    # Avoid division by zero for i=j
    np.fill_diagonal(sq_dist, 1.0)
    
    # Distances
    dist = np.sqrt(sq_dist)
    
    # Energy: sum_{i<j} -log(dist_ij)
    # We can compute the sum over all i != j and divide by 2
    # log(1/d) = -log(d)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_inv_dist = -np.log(dist)
    np.fill_diagonal(log_inv_dist, 0.0)
    energy = 0.5 * np.sum(log_inv_dist)
    
    # Gradient calculation
    # grad_k = sum_{j != k} (x_j - x_k) / ||x_j - x_k||^2
    # Note: (x_j - x_k) is -diff[k,j] if diff[i,j] = x_i - x_j.
    # So diff[k,j] = x_k - x_j. We need x_j - x_k = -diff[k,j].
    # Term: -diff[k,j] / sq_dist[k,j]
    
    # We want G[k] = sum_j (-diff[k,j]) / sq_dist[k,j] for j != k
    # Let's create a matrix W where W[k,j] = 1/sq_dist[k,j] if k!=j else 0
    inv_sq_dist = 1.0 / sq_dist
    np.fill_diagonal(inv_sq_dist, 0.0)
    
    # G = sum_j inv_sq_dist[k,j] * (-diff[k,j])
    # G[k] = - sum_j inv_sq_dist[k,j] * diff[k,j]
    # This is a matrix multiplication: - (inv_sq_dist @ diff)
    # Wait, diff is (N, N, 3).
    # For each k, sum over j of inv_sq_dist[k,j] * diff[k,j]
    # This is equivalent to np.einsum('ij,ijk->ik', inv_sq_dist, diff)
    grad_euclidean = -np.einsum('ij,ijk->ik', inv_sq_dist, diff)
    
    # Project gradient to tangent space
    # g_proj = g - (g . x) x
    # g . x is sum over components
    dot = np.sum(grad_euclidean * points, axis=1, keepdims=True)
    grad_tangent = grad_euclidean - dot * points
    
    return energy, grad_tangent

def _optimize(points, max_iter=1000, tol=1e-8):
    """
    Gradient descent on the sphere with adaptive step size.
    """
    pts = points.copy()
    energy, grad = _energy_and_grad(pts)
    
    step = 0.01
    for _ in range(max_iter):
        # Line search / adaptive step
        # Try moving in negative gradient direction
        # Since it's on a sphere, we can just take a step and normalize
        # To ensure decrease, we can do a simple backtracking line search
        # Or just use a fixed small step and hope for the best, but adaptive is safer.
        
        # Let's try a simple approach: take a step, normalize, check energy.
        # If energy increases, reduce step.
        
        # Actually, for logarithmic potential, the function is convex near the minimum?
        # No, it's not globally convex, but local minima are likely good.
        
        # Let's use a simpler heuristic: 
        # new_pts = pts - step * grad
        # new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        # new_energy, _ = _energy_and_grad(new_pts)
        # if new_energy < energy:
        #     pts = new_pts
        #     energy = new_energy
        #     step *= 1.05 # increase step slightly
        # else:
        #     step *= 0.5
        #     if step < 1e-10: break
        
        # This might be slow if we calculate energy every time.
        # Let's stick to this for robustness.
        
        new_pts = pts - step * grad
        # Normalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        # Avoid zero norm
        norms[norms < 1e-10] = 1.0
        new_pts = new_pts / norms
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy < energy - 1e-12:
            pts = new_pts
            energy = new_energy
            grad = new_grad
            step *= 1.02
            if step > 0.1: step = 0.1
            if np.linalg.norm(grad) < tol:
                break
        else:
            step *= 0.5
            if step < 1e-10:
                break

    return pts, energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize the initial guess
    best_pts, best_energy = _optimize(init_pts)
    
    # Multi-start strategy
    # Generate random perturbations or random points
    # Random points on sphere
    def random_sphere_points(n):
        v = np.random.randn(n, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v
    
    while time.time() < deadline:
        # Try a few random restarts
        for _ in range(5):
            if time.time() >= deadline:
                break
            
            # Perturb the best points found so far, or start fresh?
            # Starting fresh from random might find better basins.
            # Perturbing best points might refine local minima.
            # Let's mix: 50% random, 50% perturbed best.
            
            if np.random.rand() < 0.5:
                start_pts = random_sphere_points(N)
            else:
                # Add small noise to best points
                noise = np.random.randn(N, 3) * 0.1
                start_pts = best_pts + noise
                start_pts /= np.linalg.norm(start_pts, axis=1, keepdims=True)
            
            curr_pts, curr_energy = _optimize(start_pts, max_iter=500)
            
            if curr_energy < best_energy:
                best_energy = curr_energy
                best_pts = curr_pts
    
    return best_pts
