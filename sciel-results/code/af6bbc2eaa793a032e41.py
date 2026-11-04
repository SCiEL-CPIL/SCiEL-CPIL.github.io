import numpy as np
import time
import math

N = 150

def compute_energy(x):
    # x is (N, 3)
    # distances
    d = np.linalg.norm(x[:, np.newaxis] - x[np.newaxis, :], axis=2)
    # avoid log(0)
    d = np.maximum(d, 1e-10)
    # sum over i < j
    # mask upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return np.sum(np.log(1.0 / d[mask]))

def compute_grad(x):
    # x is (N, 3)
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis] - x[np.newaxis, :]
    # dist[i, j] = |x[i] - x[j]|
    dist = np.linalg.norm(diff, axis=2)
    # avoid division by zero
    dist = np.maximum(dist, 1e-10)
    # grad[i] = - sum_{j != i} (x[i] - x[j]) / dist[i, j]^2
    # term[i, j] = (x[i] - x[j]) / dist[i, j]^2
    # We want - sum term
    # term shape (N, N, 3)
    # dist shape (N, N)
    # term = diff / dist[:, :, np.newaxis]**2
    # But we need to exclude i=j
    # Set diagonal to 0
    np.fill_diagonal(dist, 1.0) # avoid div by 0 for diagonal, though diff is 0
    # Actually, diff diagonal is 0, so term diagonal is 0/1 = 0.
    # So we can just compute:
    inv_dist_sq = 1.0 / (dist * dist)
    # term[i, j] = diff[i, j] * inv_dist_sq[i, j]
    term = diff * inv_dist_sq[:, :, np.newaxis]
    # grad[i] = - sum_j term[i, j]
    grad = -np.sum(term, axis=1)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_energy = np.inf
    best_x = None
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try a few random restarts as well if time permits
    restarts = 1
    if time.time() < deadline - 50:
        restarts = 3
        
    for restart in range(restarts):
        if restart == 0:
            x = x0.copy()
        else:
            # Random start with some structure
            x = np.random.randn(N, 3)
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            
        energy = compute_energy(x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            
        # Gradient descent
        eta = 0.01
        for step in range(10000):
            if time.time() > deadline:
                break
                
            grad = compute_grad(x)
            # Update
            x_new = x - eta * grad
            # Normalize
            norms = np.linalg.norm(x_new, axis=1, keepdims=True)
            x_new = x_new / norms
            x = x_new
            
            # Decay learning rate
            eta = max(eta * 0.999, 1e-5)
            
            # Check energy
            e = compute_energy(x)
            if e < best_energy:
                best_energy = e
                best_x = x.copy()
                
    return best_x
