import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad_fast(X):
    """
    Compute logarithmic energy and gradient using Gram matrix for speed.
    E = sum_{i<j} log(1/d_ij) where d_ij = ||x_i - x_j||
    Gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    """
    # Gram matrix: G[i,j] = x_i . x_j
    G = X @ X.T
    # d_ij^2 = 2 - 2*G[i,j]
    dist_sq = 2.0 - 2.0 * G
    # Ensure positive (numerical safety)
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    # Zero diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # log(d_ij) = 0.5 * log(d_ij^2)
    with np.errstate(divide='ignore'):
        log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    E = -0.25 * np.sum(log_dist_sq)
    
    # Gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # = sum_{j!=i} x_j / d_ij^2 - x_i * sum_{j!=i} 1/d_ij^2
    # Let W[i,j] = 1/d_ij^2 for i!=j, W[i,i]=0
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    Grad = W @ X - row_sum[:, np.newaxis] * X
    
    return E, Grad

def _fibonacci_sphere(n, seed=0):
    """Generate points on sphere using Fibonacci spiral with optional perturbation."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.standard_normal(n) * 0.01
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

def _random_sphere(n, seed=0):
    """Generate random points on sphere."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

def _optimize_lbfgsb(X, max_time=30.0, tol=1e-12):
    """Optimize using L-BFGS-B on the sphere with projection."""
    start = time.time()
    best_X = X.copy()
    best_E, _ = _energy_and_grad_fast(X)
    
    def objective_and_grad(flat_X):
        X_local = flat_X.reshape(N, 3)
        # Project to sphere
        norms = np.linalg.norm(X_local, axis=1, keepdims=True)
        X_local = X_local / norms
        E, Grad = _energy_and_grad_fast(X_local)
        # Project gradient to tangent space
        proj = np.sum(Grad * X_local, axis=1, keepdims=True)
        Grad_tan = Grad - proj * X_local
        return E, Grad_tan.flatten()
    
    x0 = X.flatten()
    
    # Use L-BFGS-B
    res = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
    )
    
    X_new = res.x.reshape(N, 3)
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    X_new = X_new / norms
    
    E_new, _ = _energy_and_grad_fast(X_new)
    if E_new < best_E:
        best_X = X_new
        best_E = E_new
    
    return best_X, best_E

def _polish(X, max_time, tol=1e-14):
    """
    High-precision Riemannian gradient descent with backtracking line search.
    Uses the fast Gram-matrix based energy/gradient.
    """
    E, Grad = _energy_and_grad_fast(X)
    best_E = E
    best_X = X.copy()
    
    lr = 0.01
    min_lr = 1e-16
    max_rejects = 30
    reject_count = 0
    start = time.time()
    
    while time.time() - start < max_time:
        # Project gradient to tangent space
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Grad_tan = Grad - proj * X
        
        # Skip if gradient is tiny
        grad_norm = np.linalg.norm(Grad_tan)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        accepted = False
        for _ in range(max_rejects):
            X_new = X + lr * Grad_tan
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            E_new, Grad_new = _energy_and_grad_fast(X_new)
            
            if E_new < E:
                # Accept
                X = X_new
                E = E_new
                Grad = Grad_new
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                accepted = True
                # Slightly increase lr on success
                lr = min(lr * 1.02, 1.0)
                reject_count = 0
                break
            else:
                lr *= 0.5
        
        if not accepted:
            # Reset learning rate and try a small random perturbation
            lr = 0.01
            # Small random kick in tangent space
            rng = np.random.default_rng(42 + int(time.time() * 1000) % 10000)
            kick = rng.standard_normal((N, 3)) * 1e-4
            kick_proj = np.sum(kick * X, axis=1, keepdims=True)
            kick = kick - kick_proj * X
            X = X + kick
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms
            E, Grad = _energy_and_grad_fast(X)
            reject_count += 1
            if reject_count > 10:
                break
    
    return best_X, best_E

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s buffer before 150s kill
    
    # Try multiple initializations and keep the best
    best_X = None
    best_E = np.inf
    
    # List of initialization strategies
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N, seed=0))
    
    # Multiple random initializations
    for seed in range(1, 6):
        initializations.append(_random_sphere(N, seed=seed))
    
    # Optimized Fibonacci with different seeds
    for seed in range(1, 4):
        initializations.append(_fibonacci_sphere(N, seed=seed))
    
    # Process each initialization
    for idx, X_init in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 10.0:
            break
        
        # Allocate time for each optimization
        time_per_init = min(20.0, remaining / (len(initializations) - idx))
        
        # Phase 1: L-BFGS-B optimization
        X_opt, E_opt = _optimize_lbfgsb(X_init, max_time=time_per_init * 0.7)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
        
        # Phase 2: Polish if time remains
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5.0 and E_opt < best_E + 1e-3:
            X_polished, E_polished = _polish(X_opt, max_time=min(10.0, remaining - 5.0), tol=1e-14)
            if E_polished < best_E:
                best_E = E_polished
                best_X = X_polished.copy()
    
    # Final polish on the best result
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0 and best_X is not None:
        best_X, best_E = _polish(best_X, max_time=remaining - 5.0, tol=1e-15)
    
    # Final check: ensure points are on unit sphere and valid
    if best_X is None:
        # Fallback to Fibonacci
        best_X = _fibonacci_sphere(N, seed=0)
    
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    best_X = best_X / norms
    
    # Verify no points are too close
    D = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dist = np.linalg.norm(D, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fallback to Fibonacci if collision (unlikely)
        best_X = _fibonacci_sphere(N, seed=0)
    
    return best_X
