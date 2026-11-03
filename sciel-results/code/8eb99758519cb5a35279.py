"""
Strategy: Global stochastic search via Langevin Dynamics / Metropolis-Hastings.
The current program uses local gradient descent, which is prone to getting stuck
in local minima. This program uses a Monte Carlo method with a cooling schedule
(annealing) to explore the configuration space globally.
"""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 115s out of 120s limit
    
    # Precompute pair indices for vectorized energy calculation
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)
    
    def _compute_energy(X):
        """Compute total logarithmic energy."""
        # X: (N, 3) unit vectors
        diff = X[idx_i] - X[idx_j]
        # dist^2 = |xi - xj|^2
        dist_sq = np.sum(diff * diff, axis=1)
        # Avoid log(0)
        dist_sq = np.maximum(dist_sq, 1e-18)
        # Energy = sum log(1/d) = -sum log(d) = -0.5 * sum log(d^2)
        energy = -0.5 * np.sum(np.log(dist_sq))
        return energy

    def _random_tangent(X, rng):
        """Generate a random vector in the tangent space at each point of X."""
        # Random normal vector
        R = rng.standard_normal((N, 3))
        # Project to tangent space: R - (R . X) X
        dot = np.sum(R * X, axis=1, keepdims=True)
        T = R - dot * X
        # Normalize to unit length for consistent step scaling
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        T = T / norms
        return T

    def _step_langevin(X, step_size, rng):
        """
        Perform a Langevin step:
        1. Calculate gradient
        2. Add random noise
        3. Project to tangent space
        4. Normalize step to step_size
        5. Exponential map
        """
        # Compute gradient
        diff = X[idx_i] - X[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-18)
        inv_d2 = 1.0 / dist_sq
        
        grad_euclidean = np.zeros((N, 3))
        # grad_i += (xi - xj) / d^2
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        # grad_j += (xj - xi) / d^2 = -diff / d^2
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])
        
        # Project gradient to tangent space
        dot_grad = np.sum(grad_euclidean * X, axis=1, keepdims=True)
        grad_tangent = grad_euclidean - dot_grad * X
        
        # Add random noise
        noise = rng.standard_normal((N, 3))
        # Project noise to tangent space
        dot_noise = np.sum(noise * X, axis=1, keepdims=True)
        noise_tangent = noise - dot_noise * X
        
        # Combine: -alpha * grad + beta * noise
        # We want the total displacement vector V
        # Let's scale gradient and noise appropriately
        # Typically, Langevin is dX = -grad U dt + sqrt(2D dt) dW
        # Here we mix them.
        V = -0.5 * grad_tangent + 0.5 * noise_tangent
        
        # Normalize V to have unit length per point? Or fixed magnitude?
        # To ensure step_size controls the magnitude of movement:
        V_norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_norms = np.maximum(V_norms, 1e-12)
        V_unit = V / V_norms
        
        # Apply exponential map with step_size
        # X_new = cos(step) * X + sin(step) * V_unit
        cos_step = math.cos(step_size)
        sin_step = math.sin(step_size)
        X_new = cos_step * X + sin_step * V_unit
        
        # Renormalize
        X_new_norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new_norms = np.maximum(X_new_norms, 1e-12)
        X_new = X_new / X_new_norms
        
        return X_new

    # Initialize with Fibonacci sphere for good starting spread
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_energy = _compute_energy(X)
    best_X = X.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    # Start with larger steps to explore, decrease to refine
    # Temperature T controls acceptance of worse moves
    T_start = 10.0
    T_end = 0.01
    step_start = 0.2
    step_end = 0.01
    
    # Number of iterations to attempt
    # We don't know how many fit, so we loop until deadline
    iter_count = 0
    
    while time.time() < deadline:
        # Calculate current temperature and step size based on progress
        # Simple linear cooling if we knew total iters, but we don't.
        # Use time-based cooling? Or just fixed schedule per "epoch"?
        # Let's use a fixed number of iterations per "phase" and cool down.
        
        # Actually, let's just use a global counter for cooling
        # Assume we can do ~5000-10000 iterations in 115s?
        # Let's estimate: 100 points, 4950 pairs.
        # Energy calc: ~4950 ops. Gradient: ~4950 ops.
        # Vectorized NumPy is fast. Let's aim for high iteration count.
        
        # Dynamic cooling:
        # We'll use a fixed number of iterations for the first phase, then cool.
        # But since we have a time limit, let's just run as many as possible
        # and cool based on the fraction of time elapsed.
        
        elapsed = time.time() - start_time
        fraction = elapsed / (deadline - start_time)
        
        # Cool temperature and step size
        T = T_start * (T_end / T_start) ** fraction
        step_size = step_start * (step_end / step_start) ** fraction
        
        # Generate candidate
        X_new = _step_langevin(X, step_size, rng)
        E_new = _compute_energy(X_new)
        
        # Metropolis criterion
        delta_E = E_new - current_energy
        if delta_E < 0:
            # Accept if better
            X = X_new
            current_energy = E_new
            if current_energy < best_energy:
                best_energy = current_energy
                best_X = X.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            # If T is very small, this will be 0 for significant delta_E
            if T > 1e-6:
                prob = math.exp(-delta_E / T)
                if rng.random() < prob:
                    X = X_new
                    current_energy = E_new
            # If T is small and delta_E > 0, reject
            
        iter_count += 1
        
        # Check time frequently to avoid overshooting
        if iter_count % 100 == 0 and time.time() > deadline:
            break

    # Final local refinement using gradient descent to squeeze out more energy
    # Use the best_X found by Monte Carlo as starting point
    X_final = best_X.copy()
    
    # Quick local refinement
    for _ in range(50):
        if time.time() > deadline - 1.0:
            break
            
        # Compute gradient
        diff = X_final[idx_i] - X_final[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-18)
        inv_d2 = 1.0 / dist_sq
        
        grad_euclidean = np.zeros((N, 3))
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])
        
        dot_grad = np.sum(grad_euclidean * X_final, axis=1, keepdims=True)
        grad_tangent = grad_euclidean - dot_grad * X_final
        
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-8:
            break
            
        # Step size based on gradient magnitude
        # Use a simple line search or fixed small step
        step = 0.05 / (grad_norm + 1e-6)
        # Cap step to avoid large jumps
        step = min(step, 0.1)
        
        # Exponential map
        V = -grad_tangent
        V_norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_norms = np.maximum(V_norms, 1e-12)
        V_unit = V / V_norms
        
        cos_step = math.cos(step)
        sin_step = math.sin(step)
        X_trial = cos_step * X_final + sin_step * V_unit
        X_trial = X_trial / np.linalg.norm(X_trial, axis=1, keepdims=True)
        
        E_trial = _compute_energy(X_trial)
        
        if E_trial < current_energy:
            X_final = X_trial
            current_energy = E_trial
            if current_energy < best_energy:
                best_energy = current_energy
                best_X = X_final.copy()
        else:
            # Reduce step size if it doesn't improve
            # For simplicity, just break or try smaller step
            # Let's try a smaller step
            step *= 0.5
            cos_step = math.cos(step)
            sin_step = math.sin(step)
            X_trial = cos_step * X_final + sin_step * V_unit
            X_trial = X_trial / np.linalg.norm(X_trial, axis=1, keepdims=True)
            E_trial = _compute_energy(X_trial)
            if E_trial < current_energy:
                X_final = X_trial
                current_energy = E_trial
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_X = X_final.copy()
            else:
                break

    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    return best_X
# EVOLVE-BLOCK-END
