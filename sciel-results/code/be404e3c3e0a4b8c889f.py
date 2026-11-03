import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initial guess: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    # Compute initial energy
    def compute_energy(X):
        # Vectorized distance calculation
        # X shape (N, 3)
        # D2 shape (N, N)
        D2 = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * (X @ X.T)
        np.fill_diagonal(D2, 1.0) # Avoid log(0)
        # Energy is sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(sqrt(D2)) = -0.25 * sum_{i!=j} log(D2)
        # Wait, sum_{i<j} is half of sum_{i!=j}.
        # log(1/d) = -log(d) = -0.5 log(d^2).
        # So E = -0.5 * sum_{i<j} log(D2_ij) = -0.25 * sum_{i!=j} log(D2_ij).
        # Let's just compute the sum directly to be safe and match the definition.
        # But for speed, we can use the matrix form.
        # Sum over i<j of log(1/sqrt(D2_ij))
        # This is equivalent to -0.5 * sum_{i<j} log(D2_ij)
        # Sum_{i<j} log(D2_ij) = 0.5 * (Sum_{i!=j} log(D2_ij))
        # So E = -0.25 * Sum_{i!=j} log(D2_ij)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        D2_masked = np.where(mask, D2, 1.0)
        return -0.25 * np.sum(np.log(D2_masked))
    
    E_current = compute_energy(X)
    
    # 2. Riemannian Gradient Descent
    # Parameters
    T = 50000 # Max iterations
    lr_start = 1e-3
    lr_end = 1e-6
    momentum = 0.9
    V = np.zeros_like(X)
    
    best_X = X.copy()
    best_E = E_current
    
    for t in range(T):
        if time.time() > deadline:
            break
            
        # Compute Gradient
        # D2 = ||xi - xj||^2
        D2 = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * (X @ X.T)
        # Clip small values to avoid inf
        D2 = np.maximum(D2, 1e-12)
        np.fill_diagonal(D2, 1.0)
        
        # Force F_i = sum_{j != i} (x_i - x_j) / D2_ij
        # Diff_ij = x_i - x_j
        Diff = X[:, None, :] - X[None, :, :]
        InvD2 = 1.0 / D2
        np.fill_diagonal(InvD2, 0.0)
        
        # F_i = sum_j Diff_ij * InvD2_ij
        F = np.sum(Diff * InvD2[:, :, None], axis=1)
        
        # Project to Tangent Space
        # grad_perp = F - (F . x_i) x_i
        dot_F_X = np.sum(F * X, axis=1, keepdims=True)
        grad_perp = F - dot_F_X * X
        
        # Learning Rate Schedule
        lr = lr_start * (lr_end / lr_start) ** (t / T)
        
        # Update Momentum
        V = momentum * V + lr * grad_perp
        
        # Update Position
        X_new = X + V
        
        # Normalize
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        X = X_new / norms
        
        # Compute Energy
        E_new = compute_energy(X)
        
        # Update Best
        if E_new < best_E:
            best_E = E_new
            best_X = X.copy()
            
    return best_X
