import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds buffer
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Gradient descent with momentum
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    best_energy = float('inf')
    best_points = points.copy()
    
    while time.time() < deadline:
        # Compute pairwise distances
        # d[i,j] = ||points[i] - points[j]||
        # To avoid division by zero, we add a small epsilon
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist_sq[dist_sq < 1e-12] = 1e-12  # Avoid division by zero
        
        dist = np.sqrt(dist_sq)
        
        # Compute energy
        # E = sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # We use the full matrix and divide by 2
        with np.errstate(divide='ignore'):
            log_dist = np.log(dist)
        # Set diagonal to 0
        np.fill_diagonal(log_dist, 0.0)
        energy = -0.5 * np.sum(log_dist)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Compute gradient
        # grad_i = -sum_{j != i} (x_i - x_j) / dist_ij^2
        # Note: grad of log(1/d) = -1/d * grad d = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
        # Wait, the energy is sum log(1/d) = -sum log(d).
        # d/dx_i log(d_ij) = (x_i - x_j)/d_ij^2
        # So d/dx_i (-log(d_ij)) = -(x_i - x_j)/d_ij^2
        # The total gradient for point i is sum_{j != i} -(x_i - x_j)/d_ij^2
        
        # Let's compute the gradient matrix
        # G[i,j] = (x_i - x_j) / dist_sq[i,j]
        # We need to sum over j for each i, excluding j=i
        # G is (N, N, 3)
        G = diff / dist_sq[:, :, np.newaxis]
        # Set diagonal to 0
        G[np.arange(N), np.arange(N), :] = 0.0
        
        # Gradient for each point i is sum_j G[i,j]
        # But wait, the term is -(x_i - x_j)/dist^2.
        # My G is (x_i - x_j)/dist^2.
        # So grad_i = -sum_j G[i,j]
        grad = -np.sum(G, axis=1)
        
        # Project gradient onto tangent space
        # Tangent space at x_i is perpendicular to x_i
        # proj(v) = v - (v . x_i) x_i
        dot_prod = np.sum(grad * points, axis=1, keepdims=True)
        grad_tangent = grad - dot_prod * points
        
        # Update velocity
        velocity = momentum * velocity - lr * grad_tangent
        
        # Update points
        points = points + velocity
        
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Adaptive learning rate (simple heuristic)
        # If energy increased, reduce lr
        # We don't have previous energy easily, so we skip this for simplicity
        # Or we can just keep lr fixed if it's working
        
    return best_points
