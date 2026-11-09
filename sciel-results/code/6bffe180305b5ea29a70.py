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
    deadline = 110.0 # 10s margin
    
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
    
    # We fit models for steps i=10 down to 0 (0-indexed in S_paths)
    # i=10 corresponds to t=11/12. Next step is t=12.
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    for i in range(n_steps - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_current = S_paths[:, i]
        
        # Fit regression: V_realized (from step i+1) ~ Basis(S_current)
        # Use high-degree polynomial basis for better approximation
        x = np.log(S_current / K)
        # Use degree 5 polynomial
        basis = np.column_stack([np.ones_like(x), x, x**2, x**3, x**4, x**5])
        
        # Simple Least Squares with regularization
        try:
            A = basis
            b = V_realized
            # Add small ridge regularization
            reg = 1e-4
            coeffs, residuals, rank, s = np.linalg.lstsq(A + reg * np.eye(A.shape[1]), b, rcond=None)
            
            # Predict continuation value
            predicted_val = basis @ coeffs
            v_cont = discount * predicted_val
            
            # Immediate payoff
            payoff = np.maximum(K - S_current, 0)
            
            # Exercise decision
            exercise_mask = payoff > v_cont
            
            # Update V_realized
            V_realized = np.where(exercise_mask, payoff, V_realized)
            
            # Calculate threshold
            # The threshold is the spot where payoff = v_cont
            # We can estimate this by finding the maximum S where exercise is optimal
            if np.any(exercise_mask):
                # Find the maximum S where we exercise
                max_exercise_S = np.max(S_current[exercise_mask])
                # Add a small buffer to ensure we don't miss borderline cases
                # But be careful not to make it too high
                thresholds[i] = max_exercise_S * 1.01
            else:
                thresholds[i] = 0.0
        except Exception:
            # Fallback: use a simple heuristic
            # Estimate threshold from the distribution of S where payoff > 0
            in_money = S_current < K
            if np.any(in_money):
                thresholds[i] = np.median(S_current[in_money]) * 0.95
            else:
                thresholds[i] = K * 0.8

    # Fill remaining thresholds with a default
    for i in range(n_steps):
        if thresholds[i] == 0.0 and i < n_steps - 1:
            # If we didn't calculate it, use a default
            thresholds[i] = K * 0.9

    policy = ExercisePolicy(K, r, sigma, T, n_steps, thresholds)
    return policy

# EVOLVE-BLOCK-END
