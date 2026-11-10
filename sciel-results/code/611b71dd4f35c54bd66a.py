"""Bermudan exercise-policy research with direct Monte Carlo optimization."""
import numpy as np
import time
import math

class BermudanPolicy:
    """Policy that exercises based on learned thresholds at each exercise date."""
    
    def __init__(self, thresholds, r, T, num_steps=12):
        """
        thresholds: array of shape (num_steps,) where thresholds[j] is the price
                    below which we exercise at exercise date j (0-indexed).
        r: risk-free rate
        T: maturity
        num_steps: number of exercise dates (12)
        """
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.r = r
        self.T = T
        self.num_steps = num_steps
        # Time step
        self.dt = T / num_steps
        # Discount factors for each exercise date
        # Exercise dates are at t = dt, 2*dt, ..., T
        self.times = np.arange(1, num_steps + 1) * self.dt
        self.discount_factors = np.exp(-r * self.times)
    
    def exercise(self, current_state, time_idx):
        """
        current_state: array of spot prices at the current exercise date
        time_idx: index of the current exercise date (0-based, 0..num_steps-1)
        Returns: boolean vector indicating which paths to exercise
        """
        if time_idx < 0 or time_idx >= self.num_steps:
            return np.zeros(len(current_state), dtype=bool)
        threshold = self.thresholds[time_idx]
        # Exercise if spot <= threshold (for a put)
        return current_state <= threshold

def fit_exercise_policy(training_paths, contract):
    """
    Fit the exercise policy using training paths.
    
    training_paths: array of shape (N, num_steps+1) where columns are
                    S_0, S_1, ..., S_12 (12 exercise dates)
    contract: dict with parameters K, r, sigma, T
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = 12
    
    # Determine training path shape
    tp = np.asarray(training_paths, dtype=np.float64)
    if tp.ndim == 1:
        tp = tp.reshape(1, -1)
    
    N, total_cols = tp.shape
    
    # Extract exercise-date spots: columns 1..12 (t=1..12)
    # If total_cols == 13, columns 0..12, exercise dates at 1..12
    # If total_cols == 12, columns 0..11, exercise dates at 0..11
    if total_cols == num_steps + 1:
        exercise_spots = tp[:, 1:num_steps+1]  # shape (N, 12)
    elif total_cols == num_steps:
        exercise_spots = tp[:, :num_steps]  # shape (N, 12)
    else:
        # Fallback: assume last num_steps columns
        exercise_spots = tp[:, -num_steps:]
    
    N_train = exercise_spots.shape[0]
    
    # Payoff at each exercise date if exercised: max(K - S, 0)
    payoffs = np.maximum(K - exercise_spots, 0.0)  # shape (N, 12)
    
    # Discount factors
    dt = T / num_steps
    times = np.arange(1, num_steps + 1) * dt
    disc = np.exp(-r * times)  # shape (12,)
    
    # Discounted payoffs
    disc_payoffs = payoffs * disc[np.newaxis, :]  # shape (N, 12)
    
    # Initial thresholds: use Black-Scholes-style heuristic
    # For a Bermudan put, early exercise is optimal when interest rate effect dominates
    # Simple initialization: threshold = K * (1 - something)
    # Use a simple heuristic: start with thresholds around K * 0.9 to K * 0.7
    # Better: use the intrinsic value threshold as a starting point
    # At each date, the critical price is where continuation value = intrinsic value
    # For initialization, use a linear interpolation
    init_thresholds = np.linspace(K * 0.95, K * 0.70, num_steps)
    
    # Stochastic gradient ascent on thresholds
    # Objective: maximize mean over paths of max over dates of discounted payoff
    # (since we exercise at the first date where we choose to, but in this formulation
    #  we need to be careful about the stopping rule)
    
    # Actually, the correct formulation for Bermudan options:
    # We choose a stopping time tau in {1, 2, ..., 12}
    # Payoff = exp(-r * tau * dt) * max(K - S_tau, 0)
    # We want to maximize E[payoff]
    
    # With threshold policy: at each date j, exercise if S_j <= threshold_j
    # But we must stop at the FIRST date where we exercise
    # So the value for a path is: for j from 0 to 11, if S_j <= threshold_j, 
    #   then payoff_j = disc[j] * max(K - S_j, 0), stop
    #   else continue
    # If never exercised, payoff = 0 (or we could force exercise at last date)
    
    # Let's define the value function properly
    def compute_value(thresholds, spots, disc_payoffs, N_samples):
        """
        Compute the expected value using the threshold policy.
        For each path, find the first date where S <= threshold, take that payoff.
        If no date triggers, take 0 (or we could take the last date's payoff).
        
        Actually, for a Bermudan option, you must be able to exercise at any date.
        If you don't exercise by the last date, you get nothing (or the option expires).
        But typically, you would exercise at the last date if the option is in the money.
        
        Let's handle it as: at each date, if S <= threshold, exercise.
        At the last date, if not yet exercised, exercise if S < K (in the money).
        """
        N = spots.shape[0]
        values = np.zeros(N)
        exercised = np.zeros(N, dtype=bool)
        
        for j in range(num_steps):
            # At this date, which not-yet-exercised paths should exercise?
            mask = (~exercised) & (spots[:, j] <= thresholds[j])
            # At the last date, also exercise if in the money
            if j == num_steps - 1:
                mask_last = (~exercised) & (spots[:, j] < K)
                mask = mask | mask_last
            values[mask] = disc_payoffs[mask, j]
            exercised[mask] = True
        
        return np.mean(values)
    
    # For efficiency, let's vectorize the value computation
    def compute_value_vectorized(thresholds, spots, disc_payoffs):
        """
        Vectorized computation of expected value.
        """
        N = spots.shape[0]
        values = np.zeros(N)
        exercised = np.zeros(N, dtype=bool)
        
        for j in range(num_steps):
            mask = (~exercised) & (spots[:, j] <= thresholds[j])
            if j == num_steps - 1:
                mask = mask | ((~exercised) & (spots[:, j] < K))
            values[mask] = disc_payoffs[mask, j]
            exercised[mask] = True
        
        return np.mean(values)
    
    # Gradient via finite differences (since the objective is nonsmooth due to max)
    # Use a smooth approximation or just finite differences
    
    # Let's use a simple coordinate ascent / stochastic gradient approach
    best_thresholds = init_thresholds.copy()
    best_value = compute_value_vectorized(best_thresholds, exercise_spots, disc_payoffs)
    
    # Stochastic gradient ascent with finite difference gradients
    lr = 0.01  # learning rate
    fd_eps = 0.01  # finite difference step
    
    n_iterations = 0
    max_iterations = 200
    
    while n_iterations < max_iterations:
        if time.time() > deadline - 10.0:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros(num_steps)
        for j in range(num_steps):
            th_plus = best_thresholds.copy()
            th_plus[j] += fd_eps
            th_minus = best_thresholds.copy()
            th_minus[j] -= fd_eps
            
            v_plus = compute_value_vectorized(th_plus, exercise_spots, disc_payoffs)
            v_minus = compute_value_vectorized(th_minus, exercise_spots, disc_payoffs)
            
            grad[j] = (v_plus - v_minus) / (2 * fd_eps)
        
        # Update thresholds
        new_thresholds = best_thresholds + lr * grad
        
        # Clamp thresholds to reasonable range
        new_thresholds = np.clip(new_thresholds, 0.0, K * 1.5)
        
        new_value = compute_value_vectorized(new_thresholds, exercise_spots, disc_payoffs)
        
        if new_value > best_value:
            best_thresholds = new_thresholds
            best_value = new_value
        else:
            # Reduce learning rate if no improvement
            lr *= 0.9
        
        n_iterations += 1
        
        # Early stopping if convergence
        if n_iterations > 10 and abs(new_value - best_value) < 1e-6:
            break
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 5.0:
        lr_polish = 0.001
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            grad = np.zeros(num_steps)
            for j in range(num_steps):
                th_plus = best_thresholds.copy()
                th_plus[j] += 0.005
                th_minus = best_thresholds.copy()
                th_minus[j] -= 0.005
                v_plus = compute_value_vectorized(th_plus, exercise_spots, disc_payoffs)
                v_minus = compute_value_vectorized(th_minus, exercise_spots, disc_payoffs)
                grad[j] = (v_plus - v_minus) / 0.01
            
            new_thresholds = best_thresholds + lr_polish * grad
            new_thresholds = np.clip(new_thresholds, 0.0, K * 1.5)
            new_value = compute_value_vectorized(new_thresholds, exercise_spots, disc_payoffs)
            
            if new_value > best_value:
                best_thresholds = new_thresholds
                best_value = new_value
    
    return BermudanPolicy(best_thresholds, r, T, num_steps)
