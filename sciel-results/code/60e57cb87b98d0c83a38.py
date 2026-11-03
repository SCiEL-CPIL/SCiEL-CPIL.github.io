import numpy as np
import math
from scipy.optimize import minimize

N = 150

def compute_energy_and_grad(points):
    # points shape (N, 3)
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for gradient calculation
    # Set diagonal to 1 to avoid inf, but we will mask it out in sum
    # Actually, (x_i - x_i) is 0, so 0/1 = 0. So we just need to ensure dist_sq[i,i] != 0
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy
    triu = np.triu_indices(N, k=1)
    d_pair = np.sqrt(dist_sq[triu])
    d_pair = np.maximum(d_pair, 1e-15)
    energy = -np.sum(np.log(d_pair))
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    # forces[i, j] = (x_i - x_j) / |x_i - x_j|^2
    forces = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(forces, axis=1)
    
    return energy, grad

def construct():
    best_energy = np.inf
    best_points = None
    
    # Adam Optimizer Parameters
    lr = 0.005
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Number of steps
    steps = 3000
    
    # Initializations
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(init_fib)
    
    # 2. Random Initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)

    for init in inits:
        points = init.copy()
        
        # Adam state
        m = np.zeros_like(points)
        v = np.zeros_like(points)
        
        for t in range(1, steps + 1):
            energy, grad = compute_energy_and_grad(points)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            points = points - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Project to sphere
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
        # Final energy calculation
        final_energy, _ = compute_energy_and_grad(points)
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = points.copy()
            
    return best_points
