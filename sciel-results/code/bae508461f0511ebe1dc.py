import numpy as np
import time

N = 150

def _energy_and_grad(points):
    """
    Compute logarithmic energy and gradient for points on the sphere.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    Gradient: -dE/dx_i
    """
    N = points.shape[0]
    # Compute pairwise differences
    # diff[i, j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Compute squared distances
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    dist_sq[dist_sq < 1e-12] = 1e-12
    dist = np.sqrt(dist_sq)
    
    # Energy calculation
    # We need sum over i < j
    # Mask for i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    energy = np.sum(np.log(1.0 / dist[i_idx, j_idx]))
    
    # Gradient calculation
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: This is -dE/dx_i
    # We want to move in direction -grad_E = sum (x_i - x_j) / r^2
    # Let's call this force F
    # F_i = sum_{j != i} (x_i - x_j) / r_ij^2
    
    # diff / dist_sq[:,:,np.newaxis]
    # But we need to exclude i=j
    # Create a mask for i != j
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Compute force
    # force[i, j] = (x_i - x_j) / r_ij^2 if i != j else 0
    # We can compute (diff / dist_sq[:,:,None]) and then zero out diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        force_mat = diff / dist_sq[:, :, np.newaxis]
    force_mat[~mask] = 0.0
    
    # Sum over j for each i
    grad_neg = np.sum(force_mat, axis=1) # This is -dE/dx
    
    return energy, grad_neg

def _halton_init(N, bases=None):
    if bases is None:
        bases = [2, 3, 5]
    points = np.zeros((N, 3))
    for d in range(3):
        b = bases[d]
        for i in range(N):
            f = 1.0
            r = 0.0
            n = i + 1
            while n > 0:
                f /= b
                r += f * (n % b)
                n //= b
            points[i, d] = r
    # Map to sphere? Halton is in [0,1]^3.
    # This is not on the sphere. We need to map it or use it as a seed for random-ish points.
    # Better to use Halton for spherical coordinates or just use it as a starting point and normalize?
    # Normalizing [0,1]^3 points biases towards the corners.
    # Let's stick to Fibonacci and Random for simplicity and reliability.
    pass

def _fibonacci_init(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(N):
    points = np.random.randn(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _optimize(points, max_iters=5000, step_size=0.01, momentum=0.9, time_limit=120):
    """
    Gradient descent on the sphere.
    """
    current_points = points.copy()
    current_energy = _energy_and_grad(current_points)[0]
    v = np.zeros_like(current_points)
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        energy, grad_neg = _energy_and_grad(current_points)
        
        # Update velocity
        v = momentum * v + grad_neg
        
        # Update points
        new_points = current_points + step_size * v
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check if energy decreased
        new_energy = _energy_and_grad(new_points)[0]
        
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
        else:
            # If not, reduce step size or break
            step_size *= 0.9
            if step_size < 1e-6:
                break
                
    return current_points, current_energy

def construct():
    start_time = time.time()
    best_points = None
    best_energy = float('inf')
    
    inits = [
        _fibonacci_init(N),
        _random_init(N),
        _random_init(N),
        _random_init(N)
    ]
    
    for init in inits:
        if time.time() - start_time > 110: # Leave 10s margin
            break
        points, energy = _optimize(init, max_iters=2000, step_size=0.05, momentum=0.9, time_limit=100)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
    if best_points is None:
        best_points = _fibonacci_init(N)
        
    return best_points
