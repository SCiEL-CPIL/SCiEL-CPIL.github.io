import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_points():
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms

    def energy(X):
        """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
        # Pairwise squared distances
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        # Clamp to avoid log(0) or negative
        D2 = np.maximum(D2, 1e-30)
        np.fill_diagonal(D2, 1.0)
        D = np.sqrt(D2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        E = -np.sum(np.log(D[mask]))
        return E

    def gradient(X):
        """Gradient of energy w.r.t. X (Cartesian)."""
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        D2 = np.maximum(D2, 1e-30)
        np.fill_diagonal(D2, 1.0)
        inv_D2 = 1.0 / D2
        np.fill_diagonal(inv_D2, 0.0)
        sum_inv = np.sum(inv_D2, axis=1)
        term1 = X * sum_inv[:, np.newaxis]
        term2 = inv_D2 @ X
        return term1 - term2

    def project_tangent(G, X):
        """Project gradient onto tangent space of sphere."""
        dot = np.sum(G * X, axis=1)
        return G - dot[:, np.newaxis] * X

    def energy_and_gradient(X):
        E = energy(X)
        G = gradient(X)
        return E, G

    def optimize_from(X0, max_iter=2000, tol=1e-10):
        """Riemannian gradient descent with adaptive step size and L-BFGS-like momentum."""
        X = X0.copy()
        E_curr = energy(X)
        G_curr = gradient(X)
        G_tan = project_tangent(G_curr, X)
        
        # Simple momentum
        V = np.zeros_like(X)
        beta = 0.9
        step_size = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Update velocity (momentum)
            V = beta * V + (1 - beta) * G_tan
            
            # Try step
            alpha = step_size
            for _ in range(10):
                X_new = X + alpha * V
                norms = np.linalg.norm(X_new, axis=1, keepdims=True)
                X_new = X_new / norms
                E_new = energy(X_new)
                if E_new < E_curr:
                    break
                alpha *= 0.5
            else:
                # Failed to reduce energy, shrink step
                step_size *= 0.5
                if step_size < 1e-12:
                    break
                continue
            
            X = X_new
            E_curr = E_new
            G_curr = gradient(X)
            G_tan = project_tangent(G_curr, X)
            
            # Check convergence
            if np.linalg.norm(G_tan) < tol:
                break
            
            # Increase step size if we took a full step
            if alpha > step_size * 0.9:
                step_size = min(step_size * 1.1, 0.1)
        
        return X, E_curr

    def polish(X, max_iter=10000, tol=1e-12):
        """High-precision polish using smaller steps and tighter tolerance."""
        X = X.copy()
        E_curr = energy(X)
        step_size = 0.001
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            G = gradient(X)
            G_tan = project_tangent(G, X)
            
            if np.linalg.norm(G_tan) < tol:
                break
            
            alpha = step_size
            for _ in range(20):
                X_new = X + alpha * G_tan
                norms = np.linalg.norm(X_new, axis=1, keepdims=True)
                X_new = X_new / norms
                E_new = energy(X_new)
                if E_new < E_curr:
                    break
                alpha *= 0.5
            else:
                step_size *= 0.5
                if step_size < 1e-15:
                    break
                continue
            
            X = X_new
            E_curr = E_new
        
        return X, E_curr

    # Phase 1: Multiple diverse starts with moderate optimization
    best_X = None
    best_E = np.inf
    
    # Start 1: Fibonacci spiral
    X0 = fibonacci_spiral()
    X_opt, E_opt = optimize_from(X0, max_iter=3000)
    if E_opt < best_E:
        best_E = E_opt
        best_X = X_opt.copy()
    
    # Start 2: Random
    if time.time() < deadline - 30.0:
        X0 = random_points()
        X_opt, E_opt = optimize_from(X0, max_iter=3000)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # Start 3: Another random
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        X0 = random_points()
        X_opt, E_opt = optimize_from(X0, max_iter=3000)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # Start 4: Perturbed Fibonacci
    if time.time() < deadline - 30.0:
        X0 = fibonacci_spiral()
        X0 = X0 + 0.05 * np.random.randn(N, 3)
        norms = np.linalg.norm(X0, axis=1, keepdims=True)
        X0 = X0 / norms
        X_opt, E_opt = optimize_from(X0, max_iter=3000)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # Start 5: Structured grid
    if time.time() < deadline - 30.0:
        # Generate points on a grid on the sphere
        n_lat = 12
        n_lon = 13
        pts = []
        for lat_i in range(n_lat):
            theta = (lat_i + 0.5) * np.pi / n_lat
            for lon_j in range(n_lon):
                phi = (lon_j + 0.5) * 2 * np.pi / n_lon
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                pts.append([x, y, z])
        # Take first N points
        pts = np.array(pts[:N])
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        X_opt, E_opt = optimize_from(pts, max_iter=3000)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # Phase 2: More random restarts if time permits
    restart_count = 0
    while time.time() < deadline - 20.0 and restart_count < 10:
        X0 = random_points()
        X_opt, E_opt = optimize_from(X0, max_iter=2000)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
        restart_count += 1
    
    # Phase 3: High-precision polish on the best solution
    if time.time() < deadline - 5.0:
        best_X, best_E = polish(best_X, max_iter=5000, tol=1e-13)
    
    # Final energy computation
    final_E = energy(best_X)
    
    return best_X
