"""Bermudan exercise-policy research: Least Squares Monte Carlo (LSMC) regression."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Least Squares Monte Carlo (LSMC) regression.
    
    Parameters:
    -----------
    training_paths : ndarray of shape (N, 13) where columns are S_0, S_1, ..., S_12
    contract : dict with keys 'K', 'r', 'sigma', 'T', 'num_steps' (optional)
    
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean vector
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin out of 150s

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = contract.get('num_steps', 12)
    
    # Ensure training_paths is numpy array
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 13)
        
    N = paths.shape[0]
    
    # We care about S_1 to S_12 (indices 1 to 12 in the array)
    # S_0 is at index 0.
    S = paths[:, 1:n_exercises+1] # Shape (N, 12)
    
    # Discount factor per step
    dt = T / n_exercises
    disc = math.exp(-r * dt)
    
    # We will store the regression coefficients for the continuation value at each step k
    # V_k is the value at step k.
    # We iterate backwards from k = n_exercises (index 11) down to k = 1 (index 0).
    
    # V_next corresponds to V_{k+1}
    # At the last step (k=12, index 11), the value is the payoff.
    V_next = np.maximum(K - S[:, -1], 0.0)
    
    # Polynomials will be stored as coefficients for np.polyval
    # We use a polynomial of degree 5 or 6 for stability and speed
    poly_degree = 6
    
    # Store coefficients for each step k (1 to 11)
    # coeffs[k] corresponds to the polynomial fitted at step k to predict V_{k+1}
    # Note: We need the continuation value at step k to decide exercise.
    # Cont_k = disc * E[V_{k+1} | S_k]
    # We fit P_k(S_k) ~= E[V_{k+1} | S_k]
    
    coeffs_list = [None] * (n_exercises + 1) # 1-based indexing for clarity
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple ITM exercise
        class SimplePolicy:
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=np.float64)
                return current_state < K
        return SimplePolicy()

    # Backward induction
    # k goes from n_exercises-1 down to 0 (0-based index in S array)
    # This corresponds to exercise date k+1 (1-based)
    for k_idx in range(n_exercises - 1, -1, -1):
        if time.time() > deadline:
            break
            
        S_curr = S[:, k_idx]
        
        # Fit polynomial to S_curr -> V_next
        # We only fit on "continuation" points ideally, but standard LSMC fits on all.
        # For puts, deep ITM points are exercise, so their V_next is intrinsic.
        # To be more accurate, we can mask out deep ITM points for the regression
        # of the *continuation* part, but standard practice is to fit all and take max.
        # Let's fit all.
        
        try:
            # Use polyfit. It's fast for N=4096.
            # Degree 6 is a good balance.
            coeffs = np.polyfit(S_curr, V_next, poly_degree)
        except Exception:
            # If fitting fails (e.g., singular matrix), fall back to lower degree
            try:
                coeffs = np.polyfit(S_curr, V_next, 3)
            except Exception:
                coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
        coeffs_list[k_idx] = coeffs
        
        # Compute predicted continuation value
        V_cont_pred = np.polyval(coeffs, S_curr)
        V_cont = disc * V_cont_pred
        
        # Intrinsic value
        V_int = np.maximum(K - S_curr, 0.0)
        
        # Update V for this step
        V_curr = np.maximum(V_int, V_cont)
        
        # This V_curr becomes V_next for the previous step
        V_next = V_curr

    # Build the policy object
    class ExercisePolicy:
        def __init__(self, coeffs, n_exercises, K, disc, poly_degree):
            self.coeffs = coeffs
            self.n_exercises = n_exercises
            self.K = K
            self.disc = disc
            self.poly_degree = poly_degree
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the Bermudan put.
            
            Parameters:
            -----------
            current_state : ndarray of shape (M,) - current spot prices
            time : int or float - current time step (1-based, 1 to 12) or time in years
            
            Returns:
            --------
            boolean vector of shape (M,) - True if should exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            M = len(current_state)
            
            if time is None:
                return np.zeros(M, dtype=bool)
            
            t = float(time)
            
            # Map time to exercise index k (1-based, 1 to 12)
            # If t is 1.0, it's the first exercise date.
            # If t is 12.0, it's the last.
            # The contract says "12 monthly exercise dates (no time-0 exercise)".
            # Usually, t=1 corresponds to index 0 in our 0-based loop for step 1.
            
            if t <= 0:
                return np.zeros(M, dtype=bool)
                
            # Determine which exercise date this is
            # If t is an integer 1..12, use it directly.
            # If t is continuous, round to nearest or floor?
            # The prompt implies discrete exercise dates.
            # Let's assume t is passed as the integer step index 1..12.
            k = int(round(t))
            
            if k < 1:
                return np.zeros(M, dtype=bool)
            if k > self.n_exercises:
                k = self.n_exercises
            
            # We need the polynomial fitted at step k to predict V_{k+1}
            # In our coeffs_list, index k-1 corresponds to step k (1-based)
            # because we iterated k_idx from n-1 down to 0.
            # coeffs_list[0] is for step 1.
            # coeffs_list[11] is for step 12.
            
            # Wait, at step 12 (last), we don't have a polynomial to predict V_13.
            # At step 12, we always exercise if ITM because it's the last date.
            # So for k = 12, the continuation value is 0 (or we just check intrinsic).
            # Actually, the policy at step 12 is: Exercise if K - S > 0.
            
            if k == self.n_exercises:
                # Last exercise date: exercise if ITM
                return current_state < self.K
            
            # For k < 12:
            # Get coeffs for this step
            coeffs = self.coeffs[k-1]
            
            if coeffs is None:
                # Fallback
                return current_state < self.K
            
            # Compute continuation value
            V_cont_pred = np.polyval(coeffs, current_state)
            V_cont = self.disc * V_cont_pred
            
            # Intrinsic value
            V_int = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if Intrinsic >= Continuation
            return V_int >= V_cont

    return ExercisePolicy(coeffs_list, n_exercises, K, disc, poly_degree)

# EVOLVE-BLOCK-END
