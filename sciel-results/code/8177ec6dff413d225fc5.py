import numpy as np
import time
import math

N = 100

def _energy(X):
    """Calculate logarithmic energy."""
    # X is (N, 3)
    # Pairwise distances
    # D[i, j] = ||X[i] - X[j]||
    # We need sum_{i<j} log(1/D[i,j])
    
    # Vectorized distance calculation
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    # Since X is on unit sphere, ||x||^2 = 1
    # D^2 = 1 + 1 - 2 X @ X.T = 2 - 2 X @ X.T
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0) # Avoid division by zero, diagonal is not used in sum
    D = np.sqrt(D2)
    
    # Sum log(1/D) for i < j
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    vals = -np.log(D[iu])
    return math.fsum(vals)

def _gradient(X):
    """Calculate gradient of energy w.r.t X."""
    # G_i = sum_{j != i} (X_i - X_j) / ||X_i - X_j||^2
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0)
    
    # M_ij = 1/D2_ij for i != j, 0 otherwise
    M = 1.0 / D2
    np.fill_diagonal(M, 0.0)
    
    # S_i = sum_j M_ij
    S = np.sum(M, axis=1)
    
    # W_i = sum_j M_ij X_j
    W = M @ X
    
    # G_i = S_i X_i - W_i
    G = S[:, np.newaxis] * X - W
    return G

def _project_tangent(G, X):
    """Project gradient onto tangent space."""
    # G_tan = G - (G . X) X
    G_dot_X = np.sum(G * X, axis=1, keepdims=True)
    return G - G_dot_X * X

def _optimize(X, max_iters=2000, time_limit=100):
    """Optimize X using gradient descent in tangent space."""
    start_time = time.time()
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    E_old = _energy(X)
    eta = 0.01
    best_X = X.copy()
    best_E = E_old
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        G = _gradient(X)
        G_tan = _project_tangent(G, X)
        
        # Step size adaptation
        # Try to take a step
        for _ in range(10):
            X_new = X - eta * G_tan
            X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
            E_new = _energy(X_new)
            if E_new < E_old:
                break
            else:
                eta *= 0.5
        else:
            # If we couldn't find a good step, stop or break
            break
            
        X = X_new
        E_old = E_new
        
        if E_old < best_E:
            best_E = E_old
            best_X = X.copy()
            
        # Optional: stop if gradient is small
        # if np.linalg.norm(G_tan) < 1e-6:
        #     break
        
    return best_X, best_E

def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_start():
    X = np.random.randn(N, 3)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def construct():
    start_time = time.time()
    deadline = 120.0
    time_margin = 5.0
    
    best_X = None
    best_E = float('inf')
    
    # Try a few starts
    starts = [_fibonacci_start()]
    # Add a couple of random starts if time permits
    for _ in range(2):
        starts.append(_random_start())
        
    for X0 in starts:
        if time.time() > deadline - time_margin:
            break
        X_opt, E_opt = _optimize(X0, max_iters=5000, time_limit=deadline - time.time() - time_margin)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    if best_X is None:
        # Fallback to Fibonacci if optimization failed completely
        best_X = _fibonacci_start()
        
    return best_X
