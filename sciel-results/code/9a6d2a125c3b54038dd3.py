import numpy as np
import time
import math

N = 100

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its Euclidean gradient.
    points: (N, 3) array of unit vectors.
    Returns: energy (float), grad (N, 3) array.
    """
    # Pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero for i == j
    # We only need i < j for energy, but gradient needs all j != i
    # Set diagonal to 1 to avoid inf, then mask out later if needed or just rely on 0/1=0
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -0.5 * sum_{i!=j} log(dist)
    # Since log(dist) is symmetric, sum_{i<j} = 0.5 * sum_{i!=j}
    # log(1/d) = -log(d)
    # So E = -0.5 * sum_{i!=j} log(dist_{ij})
    
    # For gradient:
    # grad_k = sum_{j != k} (x_k - x_j) / dist_{kj}^2
    # Note: The term for j=k is 0.
    
    # Vectorized gradient calculation
    # diff is (N, N, 3). dist_sq is (N, N).
    # We want to sum over j for each i.
    # term_{ij} = diff_{ij} / dist_sq_{ij}
    # grad_i = sum_j term_{ij}
    
    # dist_sq has 1.0 on diagonal, so diff_ii / 1.0 = 0. This is correct.
    grad_terms = diff / dist_sq[:, :, np.newaxis]
    grad = np.sum(grad_terms, axis=1)
    
    # Energy calculation
    # We need sum_{i<j} -log(dist_{ij})
    # Using the upper triangle
    iu = np.triu_indices(N, k=1)
    e = -np.sum(np.log(dist[iu]))
    
    return e, grad

def _project_tangent(grad, points):
    """
    Project Euclidean gradient to the tangent space of the sphere at each point.
    grad: (N, 3)
    points: (N, 3)
    """
    # g_tan = g - (g . x) x
    dot = np.sum(grad * points, axis=1, keepdims=True)
    return grad - dot * points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_energy = float('inf')
    best_points = None
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple restarts with different rotations
    # We can use random rotations generated from QR decomposition of random matrices
    # or simple axis-angle rotations.
    
    max_restarts = 20
    max_iters = 500
    eta = 0.1
    eta_min = 1e-8
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Generate a random rotation
        # Use a random orthogonal matrix
        A = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(A)
        
        # Apply rotation to base points
        points = base_points @ Q.T
        
        # Small perturbation to break symmetry
        points += np.random.randn(N, 3) * 0.01
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        current_energy, _ = _energy_and_grad(points)
        
        # Optimization loop
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradient
            e, g = _energy_and_grad(points)
            
            if e < best_energy:
                best_energy = e
                best_points = points.copy()
                
            # Project gradient to tangent space
            g_tan = _project_tangent(g, points)
            
            # Check convergence
            if np.linalg.norm(g_tan) < 1e-10:
                break
                
            # Backtracking line search
            step_dir = -g_tan
            alpha = eta
            current_e = e
            improved = False
            
            for _ in range(20):
                # Update points
                new_points = points + alpha * step_dir
                # Normalize to sphere
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                # Check for collisions (distance < 1e-9)
                # This is expensive, so we might skip explicit check and rely on energy spike
                # or just check min distance occasionally.
                # For now, assume energy calculation handles it (log(0) = inf)
                
                new_e, _ = _energy_and_grad(new_points)
                
                if np.isfinite(new_e) and new_e < current_e:
                    points = new_points
                    current_e = new_e
                    improved = True
                    break
                else:
                    alpha *= 0.5
                    if alpha < eta_min:
                        break
            
            if not improved:
                break
                
            # Decay learning rate slightly
            eta = max(eta * 0.99, 1e-4)

    if best_points is None:
        # Fallback to base points if optimization failed completely
        return base_points
        
    return best_points
