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
    
    def _energy(X):
        """Compute log energy."""
        diff = X[idx_i] - X[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-12)
        return -np.sum(np.log(dist))
    
    def _energy_and_tangent_grad(X):
        """Compute log energy and tangent-space gradient on the sphere."""
        diff = X[idx_i] - X[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-12)
        
        energy = -np.sum(np.log(dist))
        
        grad_euclidean = np.zeros((N, 3))
        inv_d2 = 1.0 / (dist * dist)
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])
        
        dot = np.sum(grad_euclidean * X, axis=1, keepdims=True)
        grad_tangent = grad_euclidean - dot * X
        
        return energy, grad_tangent
    
    def _exp_map(X, V, step):
        """Exponential map on the sphere."""
        norm_V = np.linalg.norm(V, axis=1, keepdims=True)
        norm_V = np.maximum(norm_V, 1e-12)
        theta = step * norm_V
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        V_unit = V / norm_V
        X_new = cos_theta * X + sin_theta * V_unit
        X_new_norm = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new_norm = np.maximum(X_new_norm, 1e-12)
        X_new = X_new / X_new_norm
        return X_new
    
    def _local_minimize(X0, max_iter=200, initial_step=0.01):
        """Riemannian gradient descent with backtracking line search."""
        X = X0.copy()
        step = initial_step
        energy, grad_tangent = _energy_and_tangent_grad(X)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-8:
                break
            
            direction = -grad_tangent
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
            
            if best_energy < energy:
                X = best_X
                energy = best_energy
                grad_tangent = best_grad
            else:
                step *= 0.5
                if step < 1e-10:
                    break
        
        return X, energy
    
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
    
    def _simulate_annealing(X0, rng, t_start=0.1, t_end=0.001, max_iter=2000):
        """Simulated annealing on the sphere."""
        X = X0.copy()
        current_energy = _energy(X)
        best_X = X.copy()
        best_energy = current_energy
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Temperature schedule (geometric cooling)
            t = t_start * (t_end / t_start) ** (it / max_iter)
            
            # Generate random perturbation: move each point slightly
            # Use tangent-space random directions
            V = rng.standard_normal((N, 3))
            # Project to tangent space
            dot = np.sum(V * X, axis=1, keepdims=True)
            V_tangent = V - dot * X
            # Normalize and scale by temperature
            norm_V = np.linalg.norm(V_tangent, axis=1, keepdims=True)
            norm_V = np.maximum(norm_V, 1e-12)
            V_tangent = V_tangent / norm_V
            step_size = 0.1 * t / t_start  # Scale perturbation with temperature
            
            X_trial = _exp_map(X, V_tangent, step_size)
            trial_energy = _energy(X_trial)
            
            delta_e = trial_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / t):
                X = X_trial
                current_energy = trial_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_X = X.copy()
        
        return best_X, best_energy
    
    # Generate diverse initializations
    initializations = []
    initializations.append(_fibonacci_init())
    initializations.append(_random_init(42))
    initializations.append(_random_init(123))
    initializations.append(_random_init(456))
    initializations.append(_random_init(789))
    
    best_X = initializations[0].copy()
    best_energy = float('inf')
    
    # Phase 1: Simulated Annealing from multiple initializations
    for i, init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        rng = np.random.default_rng(1000 + i)
        X_sa, e_sa = _simulate_annealing(init, rng, t_start=0.1, t_end=0.001, max_iter=1500)
        
        if e_sa < best_energy:
            best_energy = e_sa
            best_X = X_sa
    
    # Phase 2: Final Riemannian gradient polish
    if time.time() < deadline - 5.0:
        X_polished, e_polished = _local_minimize(best_X, max_iter=300, initial_step=0.01)
        if e_polished < best_energy:
            best_energy = e_polished
            best_X = X_polished
    
    # Final check: ensure all points are distinct and on the unit sphere
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    return best_X
# EVOLVE-BLOCK-END
