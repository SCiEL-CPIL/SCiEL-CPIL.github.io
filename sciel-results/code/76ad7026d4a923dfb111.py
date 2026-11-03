"""
Strategy: Langevin Monte Carlo on the sphere manifold.
Instead of pure gradient descent which gets stuck in local minima,
we use a stochastic differential equation (Langevin dynamics) on the sphere.
The gradient pulls the system toward lower energy, while random noise
allows it to jump over energy barriers, exploring multiple local minima.
We run multiple short chains and keep the best configuration found.
"""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (10s buffer)
    
    # Precompute pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)
    
    def _energy_and_tangent_grad(X):
        """
        Compute log energy and tangent-space gradient.
        Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
        Gradient w.r.t x_i in Euclidean space: sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
        Tangent gradient: (I - x_i x_i^T) * grad_euclidean_i
        """
        diff = X[idx_i] - X[idx_j]  # (n_pairs, 3)
        dist_sq = np.sum(diff * diff, axis=1)
        dist = np.sqrt(dist_sq)
        
        # Clip to avoid log(0) or div(0)
        dist_safe = np.maximum(dist, 1e-12)
        
        # Energy
        energy = -np.sum(np.log(dist_safe))
        
        # Euclidean Gradient
        # grad_i += (x_i - x_j) / d^2
        # grad_j += (x_j - x_i) / d^2
        grad_euclidean = np.zeros((N, 3))
        inv_d2 = 1.0 / dist_sq
        # Contribution to i
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        # Contribution to j
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])
        
        # Project to Tangent Space
        # dot_i = grad_euclidean_i . x_i
        dot = np.sum(grad_euclidean * X, axis=1, keepdims=True)
        grad_tangent = grad_euclidean - dot * X
        
        return energy, grad_tangent

    def _exp_map(X, V):
        """
        Exponential map on the sphere.
        X: (N, 3) current positions
        V: (N, 3) tangent vectors (displacements)
        Returns: (N, 3) new positions
        """
        norm_V = np.linalg.norm(V, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norm_V_safe = np.maximum(norm_V, 1e-12)
        V_unit = V / norm_V_safe
        theta = norm_V_safe
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Handle zero norm V: sin(0)=0, cos(0)=1, so result is X
        # But V_unit is 0/1e-12 = 0, so it works out.
        X_new = cos_theta * X + sin_theta * V_unit
        
        # Re-normalize to ensure unit sphere
        X_norm = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_norm = np.maximum(X_norm, 1e-12)
        X_new = X_new / X_norm
        
        return X_new

    def _initialize_points():
        """
        Initialize points using a Fibonacci spiral for good coverage,
        then add a small amount of random noise to break symmetry.
        """
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add small random perturbation
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        noise = rng.standard_normal((N, 3)) * 0.01
        X = X + noise
        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / X_norm
        return X

    best_X = None
    best_energy = float('inf')
    
    # Langevin Dynamics Parameters
    # We will run several short chains.
    # Temperature schedule: Start high to explore, cool down to refine.
    # Since we have a fixed time budget, we divide time into chunks.
    
    num_chains = 5  # Number of independent Langevin chains to run
    chain_duration_frac = 0.8 # Use 80% of time for Langevin, 20% for final polish
    
    # Estimate steps per chain
    # We don't know exact time per step, so we'll loop until time runs out
    # for each chain, but we need to ensure we finish all chains or use time wisely.
    
    # Strategy: Run chains sequentially. If time is left, start a new one.
    # Each chain starts from a slightly perturbed best known or random init.
    
    current_X = _initialize_points()
    
    # Initial energy
    e_curr, g_curr = _energy_and_tangent_grad(current_X)
    
    if e_curr < best_energy:
        best_energy = e_curr
        best_X = current_X.copy()
    
    # Langevin Step Parameters
    # dt is the step size for the stochastic process.
    # The noise magnitude is sqrt(2 * T * dt).
    # The gradient step is -grad * dt.
    # We need to tune T (temperature) and dt.
    
    # A common approach is to use a fixed dt and anneal T.
    # Let's try a simple annealing schedule.
    
    total_time_budget = deadline - time.time()
    time_per_chain = total_time_budget / num_chains
    
    for chain_idx in range(num_chains):
        # Reset chain start
        if chain_idx == 0:
            X_chain = current_X.copy()
            e_chain, g_chain = e_curr, g_curr
        else:
            # Start from best known so far, with perturbation
            X_chain = best_X.copy()
            rng = np.random.default_rng(42 + chain_idx)
            noise = rng.standard_normal((N, 3)) * 0.1
            X_chain = X_chain + noise
            X_norm = np.linalg.norm(X_chain, axis=1, keepdims=True)
            X_chain = X_chain / X_norm
            e_chain, g_chain = _energy_and_tangent_grad(X_chain)
            
            if e_chain < best_energy:
                best_energy = e_chain
                best_X = X_chain.copy()
        
        # Time limit for this chain
        chain_deadline = time.time() + time_per_chain
        
        # Annealing parameters
        # Start with high temperature, end with low
        # T(t) = T_start * exp(-k * t) or linear
        # Let's use linear cooling for simplicity and stability
        T_start = 0.5
        T_end = 0.01
        # dt needs to be small enough for stability. 
        # Gradient magnitude can be large. 
        # Let's adapt dt based on gradient norm.
        
        step_count = 0
        while time.time() < chain_deadline:
            # Temperature annealing
            # Estimate progress: 0.0 to 1.0
            # We don't know total steps in advance, so use time elapsed
            elapsed = time.time() - (chain_deadline - time_per_chain)
            progress = min(1.0, elapsed / time_per_chain)
            T = T_start * (1.0 - progress) + T_end * progress
            
            # Adaptive step size
            # grad_norm = np.linalg.norm(g_chain)
            # dt = 0.01 / (1.0 + grad_norm) # Scale down if gradient is large
            
            # Fixed small step for stability, scaled by temperature?
            # In Langevin: dx = -grad * dt + sqrt(2*T*dt) * noise
            # Let's fix dt = 0.005
            dt = 0.005
            
            # Generate noise in tangent space
            rng = np.random.default_rng()
            noise_vec = rng.standard_normal((N, 3))
            
            # Project noise to tangent space
            # noise_tangent = noise - (noise . x) x
            noise_dot = np.sum(noise_vec * X_chain, axis=1, keepdims=True)
            noise_tangent = noise_vec - noise_dot * X_chain
            
            # Update
            # V = -g_chain * dt + sqrt(2*T*dt) * noise_tangent
            noise_scale = np.sqrt(2.0 * T * dt)
            V = -g_chain * dt + noise_scale * noise_tangent
            
            X_chain = _exp_map(X_chain, V)
            
            # Recompute energy and gradient
            e_new, g_new = _energy_and_tangent_grad(X_chain)
            
            # Acceptance? 
            # In Langevin, we usually just take the step. 
            # However, to ensure we don't diverge to infinity (though sphere is compact),
            # we can check if energy increased too much. 
            # But standard Langevin doesn't have Metropolis acceptance.
            # For optimization, we might want to bias towards lower energy.
            # Let's stick to standard Langevin for exploration.
            
            e_chain = e_new
            g_chain = g_new
            step_count += 1
            
            # Update best
            if e_chain < best_energy:
                best_energy = e_chain
                best_X = X_chain.copy()
        
        # End of chain
        
    # Final Polish: Run a few steps of gradient descent from the best found
    # to squeeze out any remaining error.
    if time.time() < deadline - 5.0:
        X_polish = best_X.copy()
        e_polish, g_polish = _energy_and_tangent_grad(X_polish)
        
        # Simple Riemannian Gradient Descent with backtracking
        step_size = 0.01
        for i in range(100): # Max 100 steps
            if time.time() > deadline - 5.0:
                break
            
            # Direction
            direction = -g_polish
            
            # Line search
            alpha = step_size
            improved = False
            for _ in range(10):
                X_trial = _exp_map(X_polish, direction * alpha)
                e_trial, _ = _energy_and_tangent_grad(X_trial)
                
                if e_trial < e_polish:
                    X_polish = X_trial
                    e_polish = e_trial
                    # Recompute gradient for next step
                    _, g_polish = _energy_and_tangent_grad(X_polish)
                    improved = True
                    break
                else:
                    alpha *= 0.5
            
            if not improved:
                break
            
            if e_polish < best_energy:
                best_energy = e_polish
                best_X = X_polish.copy()
                
            # Check convergence
            if np.linalg.norm(g_polish) < 1e-6:
                break

    # Final Normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    return best_X
# EVOLVE-BLOCK-END
