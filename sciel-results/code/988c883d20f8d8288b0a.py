import numpy as np
import time
import math
import scipy.optimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_E = np.inf
    
    def energy(X):
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        np.fill_diagonal(D2, 1.0)
        D = np.sqrt(D2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        E = -np.sum(np.log(D[mask]))
        return E
    
    def gradient(X):
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        np.fill_diagonal(D2, 1.0)
        inv_D2 = 1.0 / D2
        np.fill_diagonal(inv_D2, 0.0)
        sum_inv = np.sum(inv_D2, axis=1)
        term1 = X * sum_inv[:, np.newaxis]
        term2 = inv_D2 @ X
        G = term1 - term2
        return G
    
    def project_tangent(G, X):
        dot = np.sum(G * X, axis=1)
        return G - dot[:, np.newaxis] * X
    
    def lbfgs_optimize(X_init, max_iter=500, ftol=1e-12, gtol=1e-8):
        """L-BFGS optimization on the sphere using Cartesian coordinates with projection."""
        
        def objective_and_grad(x_flat):
            pts = x_flat.reshape(-1, 3)
            # Normalize to ensure we're on the sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms
            
            e = energy(pts)
            g = gradient(pts)
            # Project gradient to tangent space
            g_tan = project_tangent(g, pts)
            return e, g_tan.reshape(-1)
        
        try:
            res = scipy.optimize.minimize(
                objective_and_grad, 
                X_init.reshape(-1), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            if res.fun < best_E:
                X_opt = res.x.reshape(-1, 3)
                norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
                norms[norms < 1e-15] = 1.0
                X_opt = X_opt / norms
                return X_opt, res.fun
            return X_init, energy(X_init)
        except Exception as e:
            return X_init, energy(X_init)
    
    def gradient_descent(X_init, max_iter=5000):
        """Gradient descent with adaptive step size."""
        X = X_init.copy()
        step_size = 0.01
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            G = gradient(X)
            G_tan = project_tangent(G, X)
            
            E_curr = energy(X)
            
            X_new = X + step_size * G_tan
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            X_new = X_new / norms
            
            E_new = energy(X_new)
            
            if E_new < E_curr:
                X = X_new
                step_size = min(step_size * 1.05, 0.1)
                if E_new < best_E:
                    best_E = E_new
                    best_X = X.copy()
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        
        return X, energy(X)
    
    # Strategy 1: L-BFGS from Fibonacci spiral
    try:
        X_opt, E_opt = lbfgs_optimize(X, max_iter=500)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    except Exception:
        pass
    
    # Strategy 2: Gradient descent from Fibonacci spiral
    if time.time() < deadline - 20.0:
        try:
            X_opt, E_opt = gradient_descent(X, max_iter=5000)
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
        except Exception:
            pass
    
    # Strategy 3: Random restarts with L-BFGS
    rng = np.random.default_rng(42)
    while time.time() < deadline - 15.0:
        # Generate random points
        X_rand = rng.randn(N, 3)
        X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        try:
            X_opt, E_opt = lbfgs_optimize(X_rand, max_iter=200)
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
        except Exception:
            pass
        
        # Check if we should continue
        if time.time() > deadline - 10.0:
            break
    
    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_X = best_X / norms
    
    return best_X
