"""Bermudan exercise-policy research: LSMC-based backward induction with smooth thresholds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy import stats


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Least Squares Monte Carlo (LSMC)
    with backward induction and smooth threshold estimation.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_time_steps+1) with simulated GBM paths
    - contract: dict with keys 'K', 'r', 'sigma', 'T', 'n_steps' (exercise dates)
    
    Returns:
    - An object with method exercise(current_state, time) -> boolean vector
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)  # 12 monthly exercise dates
    
    # Training paths: shape (n_paths, n_steps+1) where index 0 is t=0
    # Exercise dates are at steps 1, 2, ..., n_steps (no exercise at t=0)
    n_paths = training_paths.shape[0]
    
    # Time grid: exercise at t_k = k * T / n_steps for k = 1, ..., n_steps
    dt = T / n_steps
    discount_factor = math.exp(-r * dt)
    
    # We'll work backwards from maturity (step n_steps) to step 1
    # At each step, we estimate the continuation value via regression
    # and determine the exercise threshold
    
    # For a put, exercise if K - S > continuation_value
    # We'll estimate continuation value as a function of S using polynomial regression
    
    # Number of polynomial terms for regression
    # Use a reasonable degree - too high causes overfitting, too low underfits
    poly_degree = 8
    
    # Store thresholds for each exercise date
    # threshold[k] = S* such that exercise if S < S*
    thresholds = np.zeros(n_steps)
    
    # Backward induction
    # Start from the last exercise date (maturity)
    # At maturity, continuation value = 0 (no more exercise opportunities)
    # So at step n_steps, exercise if K - S > 0, i.e., S < K
    
    # Actually, let's be more careful:
    # The payoff at maturity is max(K - S, 0)
    # At step n_steps (last exercise date), continuation = 0
    # So exercise if K - S > 0 => S < K
    thresholds[n_steps - 1] = K
    
    # For steps n_steps-1 down to 1:
    # Continuation value = E[discounted value at next step | S_current]
    # We estimate this via regression on the training paths
    
    for step_idx in range(n_steps - 2, -1, -1):
        if time.time() > deadline - 10:
            # Not enough time for more steps, use simple heuristic
            for remaining in range(step_idx + 1, n_steps):
                thresholds[remaining] = K
            break
        
        # Get the asset prices at current step and next step
        S_current = training_paths[:, step_idx + 1]  # step_idx+1 because step 0 is t=0
        S_next = training_paths[:, step_idx + 2]
        
        # Determine exercise decisions at next step using the threshold
        S_star_next = thresholds[step_idx + 1]
        exercise_next = S_next < S_star_next
        
        # Payoff at next step: if exercise, get K - S_next; otherwise, 0 (continuation handled later)
        # Actually, we need to compute the value at next step:
        # V_next = max(K - S_next, 0) if exercise, else continuation from further back
        # But in LSMC, we estimate the continuation value at next step
        
        # The value at next step is:
        # If exercise at next step: K - S_next
        # If not exercise: the continuation value estimated from the next regression
        # For simplicity, we use the fact that at next step, the optimal value is:
        # V_next(S) = max(K - S, C_next(S)) where C_next is the continuation
        
        # In LSMC backward induction:
        # At step j, we know the threshold S*_j
        # The value at step j is: V_j(S) = max(K - S, C_j(S))
        # where C_j(S) = E[discounted V_{j+1}(S_{j+1}) | S_j = S]
        
        # For the regression at step j, we need the realized value at step j+1
        # V_{j+1}(S_{j+1}) = max(K - S_{j+1}, C_{j+1}(S_{j+1}))
        # But we don't have C_{j+1} explicitly; we only have the threshold
        
        # Simpler approach: At step j+1, the value is:
        # V_{j+1} = (K - S_{j+1}) * I(S_{j+1} < S*_{j+1}) + C_{j+1}(S_{j+1}) * I(S_{j+1} >= S*_{j+1})
        # But we approximate C_{j+1} using the regression we'll do at step j+1
        
        # Let's use a cleaner LSMC approach:
        # At each step, we regress the "continuation payoff" on the current state
        # The continuation payoff at step j is:
        # Y_j = discount_factor * V_{j+1}(S_{j+1})
        # where V_{j+1} is the optimal value at step j+1
        
        # To compute Y_j, we need V_{j+1}(S_{j+1}) for each path
        # V_{j+1}(S_{j+1}) = max(K - S_{j+1}, C_{j+1}(S_{j+1}))
        # We approximate C_{j+1} using the polynomial fit from step j+1
        
        # Let's store the polynomial coefficients for each step
        poly_coeffs = [None] * n_steps
        poly_coeffs[n_steps - 1] = np.zeros(poly_degree + 1)  # Continuation = 0 at maturity
        
        # Compute V at next step for each path
        S_next_arr = training_paths[:, step_idx + 2]
        
        # Continuation at next step: polynomial evaluation
        if step_idx + 1 == n_steps - 1:
            C_next = np.zeros(n_paths)
        else:
            # Evaluate the polynomial from step j+1
            coeffs = poly_coeffs[step_idx + 1]
            C_next = np.polyval(coeffs, S_next_arr)
        
        # Optimal value at next step
        V_next = np.maximum(K - S_next_arr, C_next)
        
        # Continuation payoff at current step (discounted)
        Y = discount_factor * V_next
        
        # Regression: fit Y as a function of S_current
        # Use polynomial regression
        X = S_current
        # Center and scale for numerical stability
        X_mean = np.mean(X)
        X_std = np.std(X)
        if X_std < 1e-10:
            X_std = 1.0
        X_scaled = (X - X_mean) / X_std
        
        # Build design matrix
        design_matrix = np.vander(X_scaled, N=poly_degree + 1, increasing=True)
        
        # Solve least squares
        try:
            coeffs_scaled, residuals, rank, sv = np.linalg.lstsq(design_matrix, Y, rcond=None)
        except:
            coeffs_scaled = np.zeros(poly_degree + 1)
        
        # Convert back to original scale
        # We need to express the polynomial in terms of S, not S_scaled
        # P(S) = sum_{k=0}^{degree} c_k * ((S - mean)/std)^k
        # We'll store the scaled coefficients and the mean/std for evaluation
        
        poly_coeffs[step_idx] = (coeffs_scaled, X_mean, X_std)
        
        # Find the exercise threshold: S* such that K - S* = C(S*)
        # where C(S) is the continuation value polynomial
        # K - S = P(S) => P(S) + S - K = 0
        
        # Define the function to find roots of
        def exercise_condition(S):
            """Returns K - S - C(S), we want this to be zero"""
            S_scaled = (S - X_mean) / X_std
            C_val = np.polyval(coeffs_scaled, S_scaled)
            return K - S - C_val
        
        # Find the root in a reasonable range
        # For a put, the threshold should be between 0 and K
        # Use bisection or root-finding
        
        # Evaluate at several points to bracket the root
        S_low = 0.0
        S_high = K
        
        # Check if there's a root
        f_low = exercise_condition(S_low)
        f_high = exercise_condition(S_high)
        
        if f_low * f_high < 0:
            # Root exists in [S_low, S_high], use bisection
            for _ in range(100):
                S_mid = (S_low + S_high) / 2
                f_mid = exercise_condition(S_mid)
                if abs(f_mid) < 1e-10:
                    break
                if f_low * f_mid <= 0:
                    S_high = S_mid
                else:
                    S_low = S_mid
                    f_low = f_mid
            thresholds[step_idx] = (S_low + S_high) / 2
        else:
            # No sign change; use the point where exercise is most favorable
            # If f_low > 0 and f_high > 0: always exercise (threshold = K)
            # If f_low < 0 and f_high < 0: never exercise in [0, K] (threshold = 0)
            if f_low > 0:
                thresholds[step_idx] = K
            else:
                # Try to find where it's closest to zero
                S_grid = np.linspace(0, K, 1000)
                f_grid = np.array([exercise_condition(s) for s in S_grid])
                idx = np.argmin(np.abs(f_grid))
                thresholds[step_idx] = S_grid[idx]
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, thresholds, n_steps):
            self.thresholds = thresholds
            self.n_steps = n_steps
        
        def exercise(self, current_state, time_idx):
            """
            Determine exercise decisions.
            
            Parameters:
            - current_state: array of current asset prices (shape: n_eval_paths,)
            - time_idx: integer index of the exercise date (0-based, where 0 is first exercise date)
            
            Returns:
            - Boolean array: True if exercise at this date
            """
            if time_idx < 0 or time_idx >= self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            
            S_star = self.thresholds[time_idx]
            return current_state < S_star
    
    return ExercisePolicy(thresholds, n_steps)
# EVOLVE-BLOCK-END
