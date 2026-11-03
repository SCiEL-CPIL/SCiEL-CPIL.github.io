"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Precompute pair indices for efficient energy/gradient calculation
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)
    
    def _energy_and_tangent_grad(X):
        """Compute log energy and tangent-space gradient on the sphere."""
        # X: (N, 3) unit vectors
        # Compute all pairwise differences
        diff = X[idx_i] - X[idx_j]  # (n_pairs, 3)
        dist = np.linalg.norm(diff, axis=1)  # (n_pairs,)
        
        # Avoid division by zero (shouldn't happen if points are distinct)
        dist = np.maximum(dist, 1e-12)
        
        # Energy: sum log(1/d) = -sum log(d)
        energy = -np.sum(np.log(dist))
        
        # Gradient: d/dX_i [sum_{j != i} log(1/|X_i - X_j|)]
        # = sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
        # But we need the tangent-space gradient: project onto tangent space
        # grad_tangent_i = grad_euclidean_i - (grad_euclidean_i . X_i) * X_i
        
        # Compute Euclidean gradient contributions
        # For each pair (i, j), contribution to grad_i is (X_i - X_j) / d^2
        # Contribution to grad_j is (X_j - X_i) / d^2 = -(X_i - X_j) / d^2
        
        grad_euclidean = np.zeros((N, 3))
        inv_d2 = 1.0 / (dist * dist)  # (n_pairs,)
        # grad_i += (X_i - X_j) / d^2
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        # grad_j += (X_j - X_i) / d^2 = -diff / d^2
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])
        
        # Project to tangent space: grad_tangent = grad_euclidean - (grad_euclidean . X) * X
        dot = np.sum(grad_euclidean * X, axis=1, keepdims=True)  # (N, 1)
        grad_tangent = grad_euclidean - dot * X
        
        return energy, grad_tangent
    
    def _exp_map(X, V, step):
        """Exponential map on the sphere: move from X in direction V by step."""
        # V is in tangent space at X
        norm_V = np.linalg.norm(V, axis=1, keepdims=True)
        norm_V = np.maximum(norm_V, 1e-12)
        # V_unit = V / norm_V
        # theta = step * norm_V
        # X_new = cos(theta) * X + sin(theta) * V_unit
        theta = step * norm_V
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        V_unit = V / norm_V
        X_new = cos_theta * X + sin_theta * V_unit
        # Normalize to ensure unit norm
        X_new_norm = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new_norm = np.maximum(X_new_norm, 1e-12)
        X_new = X_new / X_new_norm
        return X_new
    
    def _local_minimize(X0, max_iter=500, initial_step=0.01):
        """Riemannian gradient descent with backtracking line search."""
        X = X0.copy()
        step = initial_step
        energy, grad_tangent = _energy_and_tangent_grad(X)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Gradient norm
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-8:
                break
            
            # Direction: negative gradient
            direction = -grad_tangent
            
            # Backtracking line search
            alpha = step
            best_energy = energy
            best_X = X
            best_grad = grad_tangent
            
            for ls in range(20):
                X_trial = _exp_map(X, direction, alpha)
                e_trial, g_trial = _energy_and_tangent_grad(X_trial)
                
                if e_trial < best_energy - 1e-4 * alpha * grad_norm:
                    best_energy = e_trial
                    best_X = X_trial
                    best_grad = g_trial
                    break
                else:
                    alpha *= 0.5
            
            # Update
            if best_energy < energy:
                X = best_X
                energy = best_energy
                grad_tangent = best_grad
            else:
                # Reduce step size
                step *= 0.5
                if step < 1e-10:
                    break
        
        return X, energy
    
    # Generate diverse initializations
    def _fibonacci_init():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def _random_init(seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms
    
    def _scaled_fibonacci_init(scale):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i * scale
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple initializations
    initializations = []
    initializations.append(_fibonacci_init())
    initializations.append(_random_init(42))
    initializations.append(_random_init(123))
    initializations.append(_scaled_fibonacci_init(1.1))
    initializations.append(_scaled_fibonacci_init(0.9))
    
    best_X = initializations[0].copy()
    best_energy = float('inf')
    
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
        
        X_opt, e_opt = _local_minimize(init)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt
    
    # Final check: ensure all points are distinct and on the unit sphere
    # Normalize again for safety
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    return best_X
# EVOLVE-BLOCK-END
