import numpy as np
import time

N = 150

def log_energy(X):
    """
    Calculate logarithmic energy and gradient.
    X: (N, 3) array of points on unit sphere.
    Returns: energy (float), gradient (N, 3) array.
    """
    # Compute pairwise squared distances
    # D2[i, j] = ||X[i] - X[j]||^2
    # = 2 - 2 * X[i] dot X[j]
    dots = X @ X.T
    D2 = 2.0 - 2.0 * dots
    np.fill_diagonal(D2, 1.0) # Avoid division by zero
    
    # W[i, j] = 1 / D2[i, j]
    W = 1.0 / D2
    np.fill_diagonal(W, 0.0)
    
    # Energy = - sum_{i<j} log(sqrt(D2[i,j])) = -0.5 * sum_{i!=j} log(sqrt(D2[i,j]))?
    # Energy = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # d_ij = sqrt(D2_ij)
    # log(d_ij) = 0.5 * log(D2_ij)
    # So Energy = -0.5 * sum_{i<j} log(D2_ij)
    # Sum over i<j is 0.5 * Sum over i!=j
    # So Energy = -0.5 * 0.5 * sum_{i!=j} log(D2_ij) = -0.25 * sum_{i!=j} log(D2_ij)
    # Wait.
    # Sum_{i<j} f(i,j) = 0.5 * Sum_{i!=j} f(i,j) if f is symmetric.
    # E = - Sum_{i<j} log(d_ij) = -0.5 * Sum_{i!=j} log(d_ij) = -0.5 * Sum_{i!=j} 0.5 log(D2_ij) = -0.25 * Sum_{i!=j} log(D2_ij).
    
    # Gradient derivation again:
    # E = -0.25 * sum_{i!=j} log(D2_ij)
    # dE/dX_k = -0.25 * sum_{j!=k} (1/D2_kj) * d(D2_kj)/dX_k
    # D2_kj = 2 - 2 X_k . X_j
    # d(D2_kj)/dX_k = -2 X_j
    # dE/dX_k = -0.25 * sum_{j!=k} (1/D2_kj) * (-2 X_j) = 0.5 * sum_{j!=k} X_j / D2_kj
    # Wait, earlier I derived:
    # E = - sum_{i<j} log(||x_i - x_j||)
    # dE/dx_k = - sum_{j!=k} (1/||x_k-x_j||) * (x_k-x_j)/||x_k-x_j|| = - sum_{j!=k} (x_k-x_j)/||x_k-x_j||^2
    # = sum_{j!=k} (x_j-x_k)/D2_kj
    # Let's check the factor.
    # My previous derivation: G_k = sum (X_k - X_j)/D2.
    # Let's re-verify.
    # E = - sum_{i<j} ln(r_ij)
    # dE/dx_k = - sum_{j!=k} (1/r_kj) * (x_k - x_j)/r_kj = - sum_{j!=k} (x_k - x_j)/r_kj^2
    # = sum_{j!=k} (x_j - x_k)/D2_kj
    # So G_k = sum_{j!=k} (X_j - X_k) / D2_kj
    # G_k = (sum_{j!=k} X_j / D2_kj) - X_k (sum_{j!=k} 1/D2_kj)
    # G_k = T_k - S_k X_k
    # Where T_k = sum_{j!=k} X_j / D2_kj and S_k = sum_{j!=k} 1/D2_kj.
    
    # Let's check the energy value calculation with this gradient.
    # If G = T - S X, then moving x in direction -G should decrease E.
    
    # Energy calculation:
    # E = -0.25 * sum_{i!=j} log(D2_ij)
    # Note: sum_{i!=j} includes both (i,j) and (j,i).
    # log(D2_ij) is symmetric.
    # So E = -0.25 * sum_{i!=j} log(D2_ij) is correct.
    
    energy = -0.25 * np.sum(np.log(D2))
    
    # Gradient
    S = W.sum(axis=1)
    T = W @ X
    G = T - S[:, np.newaxis] * X
    
    return energy, G

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X0.copy()
    best_E, _ = log_energy(best_X)
    
    # Multi-start
    # Try a few random perturbations of the Fibonacci spiral
    # And maybe some purely random starts? Fibonacci is usually better.
    
    rng = np.random.default_rng(42)
    
    num_starts = 20
    # We can't run 20 full optimizations if each takes 5s. 100s total.
    # Let's limit time per start.
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Create initial guess
        if start_idx == 0:
            X = X0.copy()
        else:
            # Perturb Fibonacci
            X = X0 + rng.normal(0, 0.1, size=(N, 3))
            # Normalize
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms
            
        # Optimization
        alpha = 0.01
        prev_E = np.inf
        X_opt = X.copy()
        E_opt, _ = log_energy(X_opt)
        
        max_iters = 500
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            E, G = log_energy(X_opt)
            
            # Check convergence
            if np.abs(prev_E - E) < 1e-10:
                break
            prev_E = E
            
            # Project gradient to tangent space
            # G_tan = G - (G . X) X
            G_dot_X = np.sum(G * X_opt, axis=1, keepdims=True)
            G_tan = G - G_dot_X * X_opt
            
            # Update
            X_new = X_opt - alpha * G_tan
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            # Check if energy decreased
            E_new, _ = log_energy(X_new)
            
            if E_new < E:
                X_opt = X_new
                # Maybe increase alpha slightly? Keep it simple.
            else:
                # Decrease alpha
                alpha *= 0.5
                if alpha < 1e-8:
                    break
        
        # Update best
        E_final, _ = log_energy(X_opt)
        if E_final < best_E:
            best_E = E_final
            best_X = X_opt.copy()
            
    return best_X
