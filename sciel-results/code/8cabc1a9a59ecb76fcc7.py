import numpy as np
import time

N = 100

def _energy_and_grad(x):
    """
    Calculate log energy and Riemannian gradient.
    x: (N, 3) array of points on unit sphere.
    """
    # diff[i, j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist_sq[i, j] = ||x_i - x_j||^2
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    # We set diagonal to 1, then mask it out later
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # d = sqrt(dist_sq)
    # log(d) = 0.5 * log(dist_sq)
    # We sum over all i != j, then divide by 2
    # log(1/d) = -log(d)
    # E = -0.5 * sum_{i!=j} log(sqrt(dist_sq)) = -0.25 * sum_{i!=j} log(dist_sq)
    # Wait, sum_{i<j} is half of sum_{i!=j}.
    # E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # Let S = sum_{i!=j} log(d_ij). Then sum_{i<j} log(d_ij) = 0.5 * S.
    # E = -0.5 * S.
    # S = sum_{i!=j} 0.5 * log(dist_sq_ij) = 0.5 * sum_{i!=j} log(dist_sq_ij).
    # So E = -0.5 * (0.5 * sum_{i!=j} log(dist_sq_ij)) = -0.25 * sum_{i!=j} log(dist_sq_ij).
    
    # Gradient of E w.r.t x_i:
    # d/dx_i ( - log ||x_i - x_j|| ) = - (x_i - x_j) / ||x_i - x_j||^2
    # Sum over j != i.
    # g_i = - sum_{j != i} (x_i - x_j) / dist_sq_ij
    
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient to tangent space
    # grad_proj = grad - (grad . x) x
    proj_coeff = np.sum(grad * x, axis=1)
    grad_tangent = grad - proj_coeff[:, np.newaxis] * x
    
    # Calculate energy
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    log_dist_sq = np.log(dist_sq)
    log_dist_sq[~mask] = 0.0
    energy = -0.25 * np.sum(log_dist_sq)
    
    return energy, grad_tangent

def _retract(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1.0, norms)
    return x / norms

def _optimize_single(start_x, max_iter=2000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    x = start_x.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    best_x = x.copy()
    best_e, _ = _energy_and_grad(x)
    
    for t in range(1, max_iter + 1):
        e, g = _energy_and_grad(x)
        
        if e < best_e:
            best_e = e
            best_x = x.copy()
            
        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Step in Euclidean space
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Retract to sphere
        x = _retract(x_new)
        
        # Check convergence
        if t % 100 == 0 and np.linalg.norm(g) < 1e-6:
            break
            
    return best_x, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_x = None
    best_e = float('inf')
    
    # Initialization strategies
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib)
    
    # 2. Random
    np.random.seed(42)
    rand = np.random.randn(N, 3)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    inits.append(rand)
    
    # 3. Random with different seed
    np.random.seed(123)
    rand2 = np.random.randn(N, 3)
    rand2 /= np.linalg.norm(rand2, axis=1, keepdims=True)
    inits.append(rand2)
    
    # 4. Grid-like (if time permits)
    # Generate points on a grid and project
    # This is less critical, random is usually good enough for N=100
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Run optimization
        # Use a moderate number of iterations
        # Adam is usually fast
        x_opt, e_opt = _optimize_single(init, max_iter=3000, lr=0.05)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt
            
        # Check time
        if time.time() > deadline - 10:
             # Try a quick local polish on best_x if time is tight?
             # Or just break
             break
             
    # If we have time, do a final polish on the best solution with smaller LR
    if best_x is not None and time.time() < deadline - 5:
        x_polished, e_polished = _optimize_single(best_x, max_iter=5000, lr=0.01)
        if e_polished < best_e:
            best_x = x_polished
            best_e = e_polished
            
    return best_x
