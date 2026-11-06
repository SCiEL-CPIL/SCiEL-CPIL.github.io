"""Bermudan exercise policy using Longstaff-Schwarz Monte Carlo."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin (limit is 150s)
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # training_paths shape: (num_paths, num_steps)
    # Assume num_steps is 13 (0 to 12 months)
    num_paths, num_steps = training_paths.shape
    
    # Exercise dates are at steps 1 to 12
    # dt is T / num_exercise_dates? No, T is 1 year, 12 monthly dates.
    # So dt = 1/12.
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # Initialize values at the last step (step 12)
    # At step 12, it is an exercise date.
    # The continuation value is 0 (option expires after step 12).
    # So value at step 12 is max(K - S_12, 0)
    S_12 = training_paths[:, 12]
    V_12 = np.maximum(K - S_12, 0.0)
    
    # We will store critical prices for each exercise date 1..11
    # At date 12, we always exercise if S < K. So critical price is K.
    critical_prices = np.zeros(n_exercise)
    critical_prices[11] = K  # index 11 corresponds to step 12
    
    # Backward induction from step 11 down to 1
    # V_next is the value at step t+1
    V_next = V_12.copy()
    
    # We need to regress the discounted V_next on S_t
    # Let's use a polynomial regression of degree 3 on S_t
    # Or maybe just linear on [1, S, S^2, S^3]
    
    for t in range(11, 0, -1):
        if time.time() > DEADLINE:
            break
            
        S_t = training_paths[:, t]
        
        # Target for regression: Discounted value from next step
        # This is the "continuation value" estimate
        Y_t = discount * V_next
        
        # Design matrix
        # Use features: 1, S_t, S_t^2, S_t^3
        # Clip S_t to avoid extreme values if necessary, but GBM is positive.
        X_t = np.column_stack([
            np.ones(num_paths),
            S_t,
            S_t**2,
            S_t**3
        ])
        
        # Solve least squares
        # beta = (X^T X)^-1 X^T Y
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_t, Y_t, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to simple linear regression if lstsq fails
            beta = np.array([0, 1, 0, 0])
            
        # Calculate continuation value estimate for all paths
        C_t = X_t @ beta
        
        # Determine exercise decision for training paths
        # Exercise if immediate payoff > continuation value
        payoff_t = np.maximum(K - S_t, 0.0)
        exercise_mask = payoff_t > C_t
        
        # Update V_next for the next iteration (going backwards)
        # V_t = payoff if exercise, else C_t
        V_t = np.where(exercise_mask, payoff_t, C_t)
        
        # Find the critical price S* for this step t
        # We want to find S such that payoff(S) = C(S)
        # payoff(S) = max(K-S, 0)
        # C(S) = beta[0] + beta[1]S + beta[2]S^2 + beta[3]S^3
        #
        # For S <= K: K - S = C(S) => beta[3]S^3 + beta[2]S^2 + (beta[1]+1)S + (beta[0]-K) = 0
        # For S > K: 0 = C(S). Since C(S) is usually positive for puts, this might not have a solution or S* > K implies never exercise (which is wrong for puts usually).
        # Actually, for a put, if S is very high, C(S) is small but positive, payoff is 0. So we hold.
        # If S is very low, payoff is high, C(S) is low. So we exercise.
        # There is typically one crossing point S* < K.
        
        # Let's scan a grid of S values to find the crossing point robustly.
        # This is safer than solving the cubic analytically.
        S_grid = np.linspace(0, K * 1.5, 1000)
        payoff_grid = np.maximum(K - S_grid, 0.0)
        C_grid = beta[0] + beta[1]*S_grid + beta[2]*S_grid**2 + beta[3]*S_grid**3
        
        # Find where payoff - C changes sign from positive to negative
        diff = payoff_grid - C_grid
        # We want the last index where diff > 0 before it becomes <= 0
        # Or the first index where diff < 0 after being > 0
        
        # Since payoff is decreasing and C is typically decreasing (but slower), 
        # diff starts positive (at S=0, payoff=K, C=beta0) and ends negative (at S=K, payoff=0, C>0).
        
        # Find crossing
        sign_change_idx = np.where(diff[:-1] * diff[1:] < 0)[0]
        
        if len(sign_change_idx) > 0:
            # Take the first sign change (lowest S) or the relevant one?
            # Usually there's one.
            idx = sign_change_idx[0]
            # Linear interpolation
            S1 = S_grid[idx]
            S2 = S_grid[idx+1]
            d1 = diff[idx]
            d2 = diff[idx+1]
            S_star = S1 - d1 * (S2 - S1) / (d2 - d1)
        else:
            # No sign change? 
            # If diff is always positive, exercise always. S* = inf (or K*2)
            # If diff is always negative, never exercise. S* = 0
            if np.all(diff > 0):
                S_star = K * 1.5
            else:
                S_star = 0.0
                
        critical_prices[t-1] = S_star
        
        # Update V_next for the next step back
        V_next = V_t
        
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, critical_prices):
            self.critical_prices = critical_prices
            self.n_exercise = n_exercise
            self.dt = dt
            self.T = T
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time
            Returns: boolean array
            """
            # Determine the exercise step index (1-based)
            # time is in [0, T]
            # Exercise dates are at t = 1*dt, 2*dt, ..., 12*dt
            # step_idx = round(time / dt)
            step_idx = int(round(time / self.dt))
            
            # Clamp step_idx to 1..12
            step_idx = max(1, min(step_idx, self.n_exercise))
            
            # Get critical price for this step
            # critical_prices array is 0-indexed for steps 1..12
            # So step 1 uses index 0, step 12 uses index 11
            S_star = self.critical_prices[step_idx - 1]
            
            # Exercise if S < S_star
            return current_state < S_star

    return ExercisePolicy(critical_prices)

# EVOLVE-BLOCK-END
