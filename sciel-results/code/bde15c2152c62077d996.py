"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _golden_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy(X):
    # sum over i<j of log(1/|xi - xj|)
    n = X.shape[0]
    # pairwise distances
    # Use broadcasting for memory efficiency with n=100
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # mask diagonal
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    # avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)

def _log_energy_grad(X):
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # gradient of log(1/|xi-xj|) wrt xi is -(xi-xj)/|xi-xj|^2
    # sum over j != i
    # For each i, grad_i = - sum_{j!=i} (xi - xj) / |xi - xj|^2
    # Let's compute the matrix G where G[i,j] = (xi - xj) / |xi - xj|^2
    # dist2 = dist**2
    dist2 = dist * dist
    # avoid division by zero on diagonal
    dist2[dist2 < 1e-15] = 1e-15
    # G = diff / dist2[..., np.newaxis]
    G = diff / dist2[..., np.newaxis]
    # grad_i = - sum_{j != i} G[i, j]
    # sum over j
    grad = -np.sum(G, axis=1)
    # diagonal terms are zero anyway since diff[i,i]=0
    return grad

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return X / norms

def _optimize_from_start(X0, time_limit, tol=1e-12, max_iter=10000):
    X = X0.copy()
    X = _project_to_sphere(X)
    best_X = X.copy()
    best_E = _log_energy(X)
    
    # L-BFGS implementation or use scipy?
    # SciPy is available. Let's use scipy.optimize.minimize with L-BFGS-B?
    # But we are on the sphere. We can use scipy.optimize.minimize with a constraint?
    # Or use a custom L-BFGS on the manifold?
    # Simpler: use scipy.optimize.minimize with 'L-BFGS-B' on the sphere? No, it doesn't handle sphere constraints well.
    # We can use 'SLSQP' with constraints, but it's slow.
    # Better: use a line search on the sphere.
    # Or use the fact that for small N, we can just do gradient descent with adaptive step size on the sphere.
    
    # Let's try a simple gradient descent with backtracking line search on the sphere.
    lr = 1e-3
    prev_E = best_E
    stagnation = 0
    
    start_time = time.time()
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        grad = _log_energy_grad(X)
        # Project gradient to tangent space
        # Tangent space: v such that v . x = 0
        # proj_grad = grad - (grad . x) x
        dot = np.sum(grad * X, axis=1, keepdims=True)
        proj_grad = grad - dot * X
        
        # Step
        X_new = X - lr * proj_grad
        X_new = _project_to_sphere(X_new)
        
        E_new = _log_energy(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            X = X_new
            lr = min(lr * 1.05, 1e-1)
            stagnation = 0
        else:
            lr *= 0.5
            stagnation += 1
            if stagnation > 50:
                break
            if lr < 1e-10:
                break
        
        if abs(prev_E - E_new) < tol:
            # Try to reduce step size further or break
            if lr < 1e-8:
                break
            lr *= 0.5
        else:
            prev_E = E_new
            
    return best_X, best_E

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Start from golden spiral
    X0 = _golden_spiral(N)
    X_best, E_best = _optimize_from_start(X0, 10.0)
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        X_rand = np.random.randn(N, 3)
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        X_try, E_try = _optimize_from_start(X_rand, 10.0)
        if E_try < E_best:
            E_best = E_try
            X_best = X_try
    
    # Final polish with tighter tolerance and smaller steps
    if time.time() < deadline - 2.0:
        X_best, E_best = _optimize_from_start(X_best, 5.0, tol=1e-14, max_iter=5000)
        
    return X_best

# EVOLVE-BLOCK-END
