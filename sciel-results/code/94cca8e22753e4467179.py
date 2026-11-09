"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class ExercisePolicy:
    def __init__(self, K, r, sigma, T, n_steps, thresholds):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.thresholds = thresholds

    def exercise(self, current_state, time):
        # time: could be float (0, T] or int index
        # current_state: array of spots
        
        # Determine step index
        # The exercise dates are t_i = i * T / n_steps for i=1..n_steps
        if isinstance(time, float):
            # Map to index 1..n_steps
            idx = int(round(time * self.n_steps / self.T))
            # Clamp
            idx = max(1, min(idx, self.n_steps))
        else:
            # Assume integer index 1..n_steps or 0..n_steps-1
            # If 0-indexed, add 1. If 1-indexed, use as is.
            # The prompt says "no time-0 exercise", so valid times are 1/12 .. 1.
            # Let's assume time is the step number 1..12 or fraction.
            # Safest: treat as float if < 1.5, else index.
            if time < 1.5:
                idx = int(round(time * self.n_steps / self.T))
            else:
                idx = int(time)
            
        idx = max(1, min(idx, self.n_steps))
        
        # Get threshold for this step
        # thresholds is 0-indexed list, so step idx uses threshold at idx-1
        threshold = self.thresholds[idx - 1]
        
        S = np.asarray(current_state, dtype=float)
        
        # Exercise if S < threshold
        return S < threshold

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = 140.0 # 10s margin from 150s limit
    
    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12
    
    # Parse training paths
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    N = paths.shape[0]
    
    # Determine columns for exercise dates
    if paths.shape[1] == n_steps + 1:
        S_paths = paths[:, 1:] # t=1 to t=12
    else:
        S_paths = paths # t=1 to t=12
    
    # Ensure S_paths shape is (N, 12)
    
    # Backward Induction
    # V_next is the realized value at the next step (for fitting the previous step)
    # At t=12 (last step), V is the European payoff
    V_realized = np.maximum(K - S_paths[:, -1], 0)
    
    # We will store the fitted models for each step i=11 down to 0 (indices in S_paths)
    thresholds = [0.0] * n_steps # Default fallback
    # Store fitted coefficients for potential refinement
    fitted_coeffs = [None] * n_steps
    
    # We fit models for steps i=10 down to 0 (0-indexed in S_paths)
    # i=10 corresponds to t=11/12. Next step is t=12.
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Create a fine grid for threshold finding
    S_grid = np.linspace(10, 150, 2000)
    
    for i in range(n_steps - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_current = S_paths[:, i]
        
        # Fit regression: V_realized (from step i+1) ~ Basis(S_current)
        # Use a basis that better captures the put payoff shape
        x = np.log(S_current / K)
        # Use degree 6 polynomial with some extra features
        basis = np.column_stack([
            np.ones_like(x), 
            x, x**2, x**3, x**4, x**5, x**6,
            np.exp(-0.5 * x**2),  # Gaussian-like feature
            np.maximum(x, 0),     # Hinge feature
            np.maximum(-x, 0)     # Hinge feature
        ])
        
        # Simple Least Squares with regularization
        try:
            A = basis
            b = V_realized
            # Add small ridge regularization
            reg = 1e-3
            coeffs, residuals, rank, s = np.linalg.lstsq(A + reg * np.eye(A.shape[1]), b, rcond=None)
            
            # Store coefficients
            fitted_coeffs[i] = coeffs
            
            # Predict continuation value on the grid
            x_grid = np.log(S_grid / K)
            basis_grid = np.column_stack([
                np.ones_like(x_grid), 
                x_grid, x_grid**2, x_grid**3, x_grid**4, x_grid**5, x_grid**6,
                np.exp(-0.5 * x_grid**2),
                np.maximum(x_grid, 0),
                np.maximum(-x_grid, 0)
            ])
            v_cont_grid = discount * (basis_grid @ coeffs)
            
            # Immediate payoff on grid
            payoff_grid = np.maximum(K - S_grid, 0)
            
            # Find threshold: where payoff = v_cont
            # We want the largest S where payoff > v_cont
            diff = payoff_grid - v_cont_grid
            # Find where diff crosses from positive to negative
            exercise_on_grid = diff > 0
            
            if np.any(exercise_on_grid):
                # Find the rightmost point where we exercise
                exercise_indices = np.where(exercise_on_grid)[0]
                last_exercise_idx = exercise_indices[-1]
                
                # Interpolate to find the exact crossing point
                if last_exercise_idx < len(S_grid) - 1:
                    # Linear interpolation between last_exercise_idx and last_exercise_idx+1
                    S1 = S_grid[last_exercise_idx]
                    S2 = S_grid[last_exercise_idx + 1]
                    d1 = diff[last_exercise_idx]
                    d2 = diff[last_exercise_idx + 1]
                    if d1 != d2:
                        threshold = S1 + (S2 - S1) * d1 / (d1 - d2)
                    else:
                        threshold = S1
                else:
                    threshold = S_grid[last_exercise_idx]
                
                # Clamp to reasonable range
                threshold = max(0.0, min(threshold, K * 1.5))
                thresholds[i] = threshold
            else:
                thresholds[i] = 0.0
        except Exception:
            # Fallback: use a simple heuristic
            in_money = S_current < K
            if np.any(in_money):
                thresholds[i] = np.median(S_current[in_money]) * 0.95
            else:
                thresholds[i] = K * 0.8
        
        # Update V_realized using the threshold
        S_current = S_paths[:, i]
        exercise_mask = S_current < thresholds[i]
        
        # Immediate payoff
        payoff = np.maximum(K - S_current, 0)
        
        # If we exercised, V_realized = payoff; otherwise keep the continuation value
        # We need to recompute continuation value for non-exercised paths
        if fitted_coeffs[i] is not None:
            x = np.log(S_current / K)
            basis = np.column_stack([
                np.ones_like(x), 
                x, x**2, x**3, x**4, x**5, x**6,
                np.exp(-0.5 * x**2),
                np.maximum(x, 0),
                np.maximum(-x, 0)
            ])
            v_cont = discount * (basis @ fitted_coeffs[i])
            V_realized = np.where(exercise_mask, payoff, v_cont)
        else:
            V_realized = np.where(exercise_mask, payoff, V_realized)

    # Fill remaining thresholds with a default
    for i in range(n_steps):
        if thresholds[i] == 0.0 and i < n_steps - 1:
            # If we didn't calculate it, use a default
            thresholds[i] = K * 0.9

    # Final polish: refine thresholds using the full model
    if time.time() < deadline - 5:
        for i in range(n_steps - 2, -1, -1):
            if fitted_coeffs[i] is None:
                continue
            if time.time() > deadline - 3:
                break
            
            # Use a finer grid for more precise threshold
            S_grid_fine = np.linspace(20, 120, 5000)
            x_grid = np.log(S_grid_fine / K)
            basis_grid = np.column_stack([
                np.ones_like(x_grid), 
                x_grid, x_grid**2, x_grid**3, x_grid**4, x_grid**5, x_grid**6,
                np.exp(-0.5 * x_grid**2),
                np.maximum(x_grid, 0),
                np.maximum(-x_grid, 0)
            ])
            v_cont_grid = discount * (basis_grid @ fitted_coeffs[i])
            payoff_grid = np.maximum(K - S_grid_fine, 0)
            diff = payoff_grid - v_cont_grid
            exercise_on_grid = diff > 0
            
            if np.any(exercise_on_grid):
                exercise_indices = np.where(exercise_on_grid)[0]
                last_exercise_idx = exercise_indices[-1]
                
                if last_exercise_idx < len(S_grid_fine) - 1:
                    S1 = S_grid_fine[last_exercise_idx]
                    S2 = S_grid_fine[last_exercise_idx + 1]
                    d1 = diff[last_exercise_idx]
                    d2 = diff[last_exercise_idx + 1]
                    if d1 != d2:
                        new_threshold = S1 + (S2 - S1) * d1 / (d1 - d2)
                    else:
                        new_threshold = S1
                else:
                    new_threshold = S_grid_fine[last_exercise_idx]
                
                new_threshold = max(0.0, min(new_threshold, K * 1.5))
                thresholds[i] = new_threshold

    policy = ExercisePolicy(K, r, sigma, T, n_steps, thresholds)
    return policy

# EVOLVE-BLOCK-END
