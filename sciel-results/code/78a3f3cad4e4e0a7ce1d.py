"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    def _energy_and_grad(X):
        """
        X: (N, 3) unit vectors.
        Returns: Energy (scalar), Gradient in Euclidean space (N, 3).
        Energy = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
        """
        D = X @ X.T
        # Ensure diagonal is exactly 1 for stability, though X is normalized
        np.fill_diagonal(D, 1.0)
        
        # d_sq = 2 - 2 * D
        d_sq = 2.0 - 2.0 * D
        # Avoid division by zero or log(0)
        # Diagonal is 0, so we mask it out later or handle carefully.
        # For gradient, we need d_sq for i != j.
        
        # Mask for upper triangle (excluding diagonal)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Safe distances for energy calculation
        # d_sq on diagonal is 0, log(0) is -inf, but we only sum off-diagonal
        # However, np.log on an array with 0s will warn or inf.
        # Let's compute log only on the mask.
        
        # To be safe with vectorization:
        d_sq_safe = np.where(mask, d_sq, 1.0) # Set diagonal to 1 to avoid log(0) warnings
        dist = np.sqrt(d_sq_safe)
        
        # Energy: sum of -log(dist) for i<j
        # dist[mask] gives the upper triangle distances
        energy = -np.sum(np.log(dist[mask]))
        
        # Gradient
        # grad_i = sum_{j != i} (x_j - x_i) / d_sq_ij
        # Vectorized:
        # Num = X[None, :, :] - X[:, None, :]  # (N, N, 3)
        # Denom = d_sq[:, :, None]  # (N, N, 1)
        
        Num = X[None, :, :] - X[:, None, :]
        
        # Denom needs to be safe for division.
        # Diagonal d_sq is 0. We divide by it, but Num diagonal is 0. 0/0 is NaN.
        # We can set Denom diagonal to 1, and then zero out the diagonal of the result.
        d_sq_safe_div = np.where(mask, d_sq, 1.0)
        Denom = d_sq_safe_div[:, :, None]
        
        G = Num / Denom
        # Zero out diagonal (which was 0/1 = 0 anyway, but just in case)
        G[~np.eye(N, dtype=bool)] = G[~np.eye(N, dtype=bool)] # No-op, but clarity
        # Actually, Num diagonal is 0, Denom diagonal is 1, so G diagonal is 0.
        
        grad = np.sum(G, axis=1)
        
        return energy, grad

    def _exp_map(X, V):
        """
        Riemannian Exponential Map on the Sphere.
        """
        norm_V = np.linalg.norm(V, axis=1, keepdims=True)
        safe_norm_V = np.where(norm_V < 1e-12, 1.0, norm_V)
        V_hat = V / safe_norm_V
        
        cos_term = np.cos(norm_V)
        sin_term = np.sin(norm_V)
        
        X_new = X * cos_term + V_hat * sin_term
        
        # Re-normalize
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_new = X_new / norms
        
        return X_new

    def _local_minimize_riemannian(X0, max_iter=500, tol=1e-8):
        """
        Riemannian Gradient Descent with Adaptive Step Size.
        """
        X = X0.copy()
        energy, grad = _energy_and_grad(X)
        
        # Project gradient to tangent space
        grad_proj = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        step_size = 0.05
        min_step = 1e-6
        max_step = 0.5
        
        for i in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            V = -step_size * grad_proj
            X_new = _exp_map(X, V)
            
            energy_new, grad_new = _energy_and_grad(X_new)
            
            if energy_new < energy:
                X = X_new
                energy = energy_new
                grad = grad_new
                if i % 5 == 0:
                    step_size = min(step_size * 1.1, max_step)
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
            
            grad_proj_new = grad_new - np.sum(grad_new * X, axis=1, keepdims=True) * X
            if np.linalg.norm(grad_proj_new) < tol:
                break
                
        return X, energy

    def _spectral_init(n_restarts=50):
        """
        Generate initial configurations by maximizing the spectral gap of the Gram matrix.
        We optimize a function that penalizes small eigenvalues of G = X X^T.
        This encourages points to be spread out.
        """
        inits = []
        
        # Also include a good deterministic start
        def _fibonacci_init():
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            X = np.column_stack([x, y, z])
            return X
        
        inits.append(_fibonacci_init())
        
        for r in range(n_restarts):
            if time.time() > deadline - 30.0:
                break
                
            # Start from random or perturbed fibonacci
            if r % 2 == 0:
                rng = np.random.default_rng(100 + r)
                X = rng.normal(size=(N, 3))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                X = _fibonacci_init()
                rng = np.random.default_rng(200 + r)
                # Add small noise
                noise = rng.normal(scale=0.1, size=(N, 3))
                X = X + noise
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            
            # Objective: Minimize -min_eigenvalue(G)
            # G = X @ X.T
            # Eigenvalues of G are related to the spread of points.
            # For a set of N points on S^2, the sum of eigenvalues is N (trace(G) = N).
            # We want to maximize the smallest non-zero eigenvalue? 
            # Actually, for a configuration of points, the eigenvalues of the Gram matrix 
            # indicate the "dimensions" spanned. 
            # A better proxy for "well-separated" is maximizing the minimum distance.
            # But min distance is non-smooth.
            
            # Let's use a different spectral objective:
            # Minimize the sum of squares of eigenvalues? No, that's fixed.
            
            # Let's try to maximize the "determinant" of the Gram matrix of a subset? No.
            
            # Alternative: Use the fact that for well-separated points, the Gram matrix 
            # has many small off-diagonal entries. 
            # We can minimize the sum of squared off-diagonal entries of G.
            # This is equivalent to minimizing sum_{i != j} (x_i . x_j)^2.
            # This is a smooth function!
            
            def _obj_and_grad(X):
                G = X @ X.T
                # Zero out diagonal
                G_off = G.copy()
                np.fill_diagonal(G_off, 0.0)
                
                # Objective: sum(G_off^2)
                val = np.sum(G_off ** 2)
                
                # Gradient of sum(G_ij^2) w.r.t X
                # d/dX sum (X X^T)_{ij}^2
                # Let F = X X^T. dF = dX X^T + X dX^T.
                # d (F_{ij}^2) = 2 F_{ij} dF_{ij}
                # This is complex. Let's use numerical gradient or a simpler proxy.
                
                # Actually, minimizing sum_{i!=j} (x_i.x_j)^2 is equivalent to making the points orthogonal.
                # Since they are on S^2, they can't be fully orthogonal if N > 3.
                # But it encourages spreading.
                
                # Gradient:
                # grad_X = 2 * (X @ (G_off @ X) + X @ (G_off.T @ X))? 
                # Let's derive:
                # E = sum_{i,j} G_{ij}^2 - N (diagonal is 1, so G_ii^2 = 1, sum is N)
                # E = ||G||_F^2 - N
                # dE = 2 Tr(G^T dG) = 2 Tr(G (dX X^T + X dX^T))
                # = 2 Tr(G dX X^T) + 2 Tr(G X dX^T)
                # = 2 Tr(X^T G dX) + 2 Tr(dX^T G^T X)  (using cyclic property)
                # = 2 Tr(dX^T (G X + G^T X))  -- Wait, G is symmetric.
                # = 2 Tr(dX^T (2 G X))
                # So grad_X = 4 G X.
                
                # Wait, we excluded diagonal.
                # Let M = G - I.
                # E = sum_{i!=j} M_{ij}^2 = ||M||_F^2 - 0 (diag is 0)
                # dE = 2 Tr(M^T dM) = 2 Tr(M (dX X^T + X dX^T))
                # = 2 Tr(M dX X^T) + 2 Tr(M X dX^T)
                # = 2 Tr(X^T M dX) + 2 Tr(dX^T M^T X)
                # = 2 Tr(dX^T (M X + M^T X))
                # Since M is symmetric, M^T = M.
                # grad_X = 4 M X.
                
                M = G - np.eye(N)
                grad_X = 4.0 * M @ X
                
                return val, grad_X
            
            # Optimize this spectral objective
            # Use L-BFGS-B on Cartesian coordinates, projecting to sphere after each step?
            # Or just do a few steps of gradient descent on the sphere.
            
            X_curr = X.copy()
            for step in range(50):
                if time.time() > deadline - 25.0:
                    break
                val, grad_X = _obj_and_grad(X_curr)
                # Project to tangent
                grad_tan = grad_X - np.sum(grad_X * X_curr, axis=1, keepdims=True) * X_curr
                # Step
                alpha = 0.01
                X_new = X_curr - alpha * grad_tan
                X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
                
                val_new, _ = _obj_and_grad(X_new)
                if val_new < val:
                    X_curr = X_new
                else:
                    alpha *= 0.5
                    if alpha < 1e-6:
                        break
            
            inits.append(X_curr)
        
        return inits

    # Generate diverse initializations using Spectral Gap Maximization
    initializations = _spectral_init(n_restarts=30)
    
    best_X = initializations[0].copy()
    best_energy = float('inf')
    
    # Run local minimization for log-energy from each initialization
    for i, init_X in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Limit iterations for each start to ensure we finish in time
            max_iter = 300 if i < 5 else 100
            res_X, res_energy = _local_minimize_riemannian(init_X, max_iter=max_iter)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_X = res_X.copy()
        except Exception:
            continue

    # Final polish with the best configuration
    if time.time() < deadline - 10.0:
        try:
            res_X, res_energy = _local_minimize_riemannian(best_X, max_iter=1000)
            if res_energy < best_energy:
                best_energy = res_energy
                best_X = res_X
        except Exception:
            pass

    # Normalize final result
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    return best_X
# EVOLVE-BLOCK-END
