import numpy as np
import time

N = 100

def _energy_and_gradient(X):
    """
    Compute the logarithmic energy and its gradient for points X (N x 3).
    Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    Gradient: - sum_{j != i} (x_i - x_j) / d_ij^2
    """
    # Compute pairwise squared distances
    # X: (N, 3)
    # X2: (N, 1, 3), X3: (1, N, 3)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    dist = np.sqrt(dist_sq)
    
    # Energy: - sum_{i<j} log(dist_ij)
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    energy = -np.sum(np.log(dist[mask])) / 2.0 # Divide by 2 because sum is over all pairs i!=j, but we want i<j
    
    # Gradient
    # G_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    # We need to sum over j for each i
    # Weight matrix W_ij = 1 / dist_sq_ij for i != j, 0 otherwise
    W = np.zeros_like(dist_sq)
    W[mask] = 1.0 / dist_sq[mask]
    
    # G = sum_j W_ij * (x_i - x_j)
    # G_i = sum_j W_ij * diff[i,j]
    G = np.sum(W[:, :, np.newaxis] * diff, axis=1)
    G = -G # Because of the negative sign in derivative of -log(d)
    
    return energy, G

def _riemannian_step(X, lr):
    """
    Perform one Riemannian gradient descent step.
    """
    energy, G = _energy_and_gradient(X)
    
    # Project gradient to tangent space
    # G_proj = G - (G . X) X
    dot = np.sum(G * X, axis=1, keepdims=True)
    G_proj = G - dot * X
    
    # Update
    X_new = X - lr * G_proj
    
    # Normalize
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    X_new = X_new / norms
    
    return energy, X_new

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin for 150s limit
    
    best_energy = np.inf
    best_X = None
    
    # Initializations
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [X0]
    
    # 2. Random points on sphere
    # Use random normal vectors and normalize
    for _ in range(10):
        X_rand = np.random.randn(N, 3)
        X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
        inits.append(X_rand)
        
    # 3. Perturbed Fibonacci
    for _ in range(10):
        X_pert = X0.copy()
        # Add small random noise
        noise = np.random.randn(N, 3) * 0.1
        X_pert += noise
        X_pert /= np.linalg.norm(X_pert, axis=1, keepdims=True)
        inits.append(X_pert)
        
    # Optimization loop
    for X_init in inits:
        if time.time() > deadline:
            break
            
        X = X_init.copy()
        lr = 0.01
        min_lr = 1e-8
        max_iter = 2000
        tol = 1e-10
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            energy, G = _energy_and_gradient(X)
            
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
                
            # Check convergence
            # Project gradient
            dot = np.sum(G * X, axis=1, keepdims=True)
            G_proj = G - dot * X
            grad_norm = np.linalg.norm(G_proj)
            
            if grad_norm < tol:
                break
            
            # Backtracking line search
            alpha = lr
            current_energy = energy
            success = False
            for _ in range(20):
                X_trial = X - alpha * G_proj
                X_trial /= np.linalg.norm(X_trial, axis=1, keepdims=True)
                trial_energy, _ = _energy_and_gradient(X_trial)
                
                if trial_energy < current_energy:
                    X = X_trial
                    success = True
                    break
                alpha *= 0.5
                
            if not success:
                # If line search fails, reduce learning rate
                lr *= 0.5
                if lr < min_lr:
                    break

    # Final check
    if best_X is None:
        # Fallback to initial
        best_X = X0
    else:
        # Ensure normalization
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        
    return best_X
