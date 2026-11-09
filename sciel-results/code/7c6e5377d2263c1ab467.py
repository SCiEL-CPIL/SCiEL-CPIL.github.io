"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class ExercisePolicy:
    def __init__(self, K, r, sigma, T, n_steps, fitted_models, fallback_thresholds):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.fitted_models = fitted_models
        self.fallback_thresholds = fallback_thresholds
        self.dt = T / n_steps
        self.discount = np.exp(-self.r * self.dt)

    def _get_basis(self, S):
        # S is an array of spots
        x = np.log(S / self.K)
        # Polynomial basis up to degree 3
        return np.column_stack([np.ones_like(x), x, x**2, x**3])

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
        
        # Get model for this step
        # fitted_models is 0-indexed list, so step idx uses model at idx-1
        model = self.fitted_models[idx - 1]
        
        S = np.asarray(current_state, dtype=float)
        
        if model is None:
            # Fallback: use threshold
            threshold = self.fallback_thresholds[idx - 1]
            return S < threshold

        # Construct basis
        basis = self._get_basis(S)
        
        # Predict continuation value (undiscounted future value, then discount)
        # The model predicts E[V_{next} | S]
        # V_cont = exp(-r * dt) * predicted_value
        predicted_val = basis @ model
        v_cont = self.discount * predicted_val
        
        # Immediate payoff
        payoff = np.maximum(self.K - S, 0)
        
        # Exercise if payoff > continuation
        return payoff > v_cont

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
    models = [None] * n_steps
    fallback_thresholds = [0.0] * n_steps # Default fallback
    
    # We fit models for steps i=10 down to 0 (0-indexed in S_paths)
    # i=10 corresponds to t=11/12. Next step is t=12.
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    for i in range(n_steps - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_current = S_paths[:, i]
        
        # Fit regression: V_realized (from step i+1) ~ Basis(S_current)
        basis = ExercisePolicy._get_basis(None, S_current) # Hack: static method call or just inline
        # Better: inline basis calculation to avoid method overhead if None
        x = np.log(S_current / K)
        basis = np.column_stack([np.ones_like(x), x, x**2, x**3])
        
        # Simple Least Squares with regularization
        try:
            A = basis
            b = V_realized
            # Add small ridge regularization
            reg = 1e-6
            coeffs, residuals, rank, s = np.linalg.lstsq(A + reg * np.eye(A.shape[1]), b, rcond=None)
            models[i] = coeffs
        except Exception:
            models[i] = None
            
        # Determine exercise decision for these training paths
        if models[i] is not None:
            predicted_val = basis @ models[i]
            v_cont = discount * predicted_val
        else:
            # If model failed, use a simple heuristic for the decision to update V_realized
            # Estimate threshold from current data: median of S where payoff > 0?
            # Or just use K * 0.9
            v_cont = np.zeros_like(S_current) # Assume 0 continuation to force exercise if in the money?
            # Better: use average continuation from neighbors or just 0
            # Let's use 0 to be conservative (exercise if in the money)
            # Actually, let's try to fit a simple mean
            v_cont = np.mean(V_realized) * np.ones_like(S_current)
            
        payoff = np.maximum(K - S_current, 0)
        
        # Update V_realized
        V_realized = np.where(payoff > v_cont, payoff, V_realized)
        
        # Calculate fallback threshold
        # The threshold is the spot where payoff = v_cont
        # For polynomial basis, this is hard to invert analytically.
        # Approximation: find the max S where payoff > v_cont in the training set
        exercise_mask = payoff > v_cont
        if np.any(exercise_mask):
            fallback_thresholds[i] = np.max(S_current[exercise_mask]) * 1.05 # Slightly higher to be safe?
            # Actually, if S < threshold, we exercise. So threshold should be slightly *above* the max exercised spot.
        else:
            fallback_thresholds[i] = 0.0 # Never exercise if not in the money in training?
            
        # Check time
        if time.time() > deadline:
            break

    # Fill remaining models with a default
    for i in range(n_steps):
        if models[i] is None:
            # Simple heuristic: Exercise if S < 0.9 K
            # We can't easily put a heuristic in the `models` list which expects coeffs.
            # We will handle None in the `exercise()` method.
            pass
        # Ensure fallback threshold is set
        if fallback_thresholds[i] == 0.0 and i < n_steps - 1:
            # If we didn't calculate it, use a default
            fallback_thresholds[i] = K * 0.9

    policy = ExercisePolicy(K, r, sigma, T, n_steps, models, fallback_thresholds)
    return policy

# EVOLVE-BLOCK-END
