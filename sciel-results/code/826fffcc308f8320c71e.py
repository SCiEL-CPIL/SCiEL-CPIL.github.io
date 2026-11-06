# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('volatility', contract.get('sigma', 0.2))
    T = contract.get('maturity', contract.get('T', 1.0))
    
    # 12 monthly exercise dates, no time-0 exercise
    n_exercise = 12
    dt = T / n_exercise  # monthly step
    
    # Set up Gauss-Hermite quadrature
    n_quad = 80  # Increased from 50 for better accuracy
    
    x_gh, w_gh = roots_hermite(n_quad)
    w_gh_norm = w_gh / np.sqrt(np.pi)
    
    mu = (r - 0.5 * sigma**2) * dt
    var = sigma**2 * dt
    sqrt_2var = np.sqrt(2.0 * var)
    
    gh_nodes = x_gh * sqrt_2var
    gh_weights = w_gh_norm
    
    # Grid of spot prices
    # Finer grid for better resolution of the exercise boundary
    n_grid = 1000
    S_min = 0.01 * K
    S_max = 2.5 * K
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V[i] stores the value function at exercise date i
    V = np.zeros((n_exercise, n_grid))
    cont_stored = np.zeros((n_exercise, n_grid))
    
    # Last exercise date: only exercise
    V[11] = np.maximum(K - S_grid, 0.0)
    cont_stored[11] = 0.0
    
    # Pre-compute terms for vectorization
    exp_mu = np.exp(mu)
    exp_gh_nodes = np.exp(gh_nodes)
    discount_factor = np.exp(-r * dt)
    
    # Backward induction
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: fill remaining with exercise value
            for ii in range(i, -1, -1):
                V[ii] = np.maximum(K - S_grid, 0.0)
                cont_stored[ii] = 0.0
            break
        
        # Vectorized calculation of next step spots
        # S_next[j, k] = S_grid[j] * exp(mu) * exp(gh_nodes[k])
        S_next = S_grid[:, np.newaxis] * (exp_mu * exp_gh_nodes[np.newaxis, :])
        
        # Clip to grid range
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Flatten for interpolation
        S_next_flat = S_next_clipped.flatten()
        
        # Interpolate V[i+1] at all next step spots
        V_next_flat = np.interp(S_next_flat, S_grid, V[i+1])
        
        # Reshape back to (n_grid, n_quad)
        V_next_2d = V_next_flat.reshape(n_grid, n_quad)
        
        # Calculate expected continuation value
        cont_vals = np.sum(V_next_2d * gh_weights[np.newaxis, :], axis=1)
        continuation = discount_factor * cont_vals
        
        # Value = max(exercise, continuation)
        exercise_vals = np.maximum(K - S_grid, 0.0)
        V[i] = np.maximum(exercise_vals, continuation)
        cont_stored[i] = continuation

    class ExercisePolicy:
        def __init__(self, V, cont_stored, S_grid, n_exercise, K):
            self.V = V
            self.cont_stored = cont_stored
            self.S_grid = S_grid
            self.n_exercise = n_exercise
            self.K = K
        
        def exercise(self, current_state, time_idx):
            t = int(np.clip(time_idx, 0, self.n_exercise - 1))
            
            S_clipped = np.clip(current_state, self.S_grid[0], self.S_grid[-1])
            cont_interp = np.interp(S_clipped, self.S_grid, self.cont_stored[t])
            
            # Exercise if immediate payoff >= continuation value
            exercise_payoff = np.maximum(self.K - current_state, 0.0)
            result = exercise_payoff >= cont_interp - 1e-10
            
            return result.astype(bool)
    
    return ExercisePolicy(V, cont_stored, S_grid, n_exercise, K)

# EVOLVE-BLOCK-END
