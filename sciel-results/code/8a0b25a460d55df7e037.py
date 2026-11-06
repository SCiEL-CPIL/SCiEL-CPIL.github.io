"""Rewritten program: threshold-based Bermudan put exercise policy via backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on training paths.
    
    Args:
        training_paths: array of shape (n_paths, n_dates) with simulated spot prices
        contract: dict with contract parameters
    
    Returns:
        An object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))
    
    # Ensure n_dates is 12 (monthly, no time-0)
    if n_dates is None:
        n_dates = 12
    
    dt = T / n_dates
    
    # Create a finer log-spaced grid for better threshold precision
    n_grid = 600
    S_min = 0.005 * K
    S_max = 4.0 * K
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    n_quad = 96
    z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_norm = z_quad * np.sqrt(2.0)
    w_norm = w_quad / np.sqrt(np.pi)
    
    # Precompute the log-multiplier for GBM step
    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)
    log_mult = drift + vol_step * z_norm  # shape (n_quad,)
    
    # Discount factor
    disc = np.exp(-r * dt)
    
    # Terminal payoff: max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store thresholds
    thresholds = np.zeros(n_dates)
    
    # Backward induction from last date to first
    for t_idx in range(n_dates - 1, -1, -1):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            # Fill remaining with K (conservative)
            for rem in range(t_idx, -1, -1):
                thresholds[rem] = K
            break
        
        # S_next shape: (n_grid, n_quad)
        S_next = S_grid[:, np.newaxis] * np.exp(log_mult[np.newaxis, :])
        
        # Interpolate V_next at S_next
        S_next_clipped = np.clip(S_next, S_min, S_max)
        V_interp = np.interp(S_next_clipped, S_grid, V_next)
        
        # Weighted sum for expectation
        cont_value = disc * np.sum(w_norm[np.newaxis, :] * V_interp, axis=1)
        
        # Immediate payoff
        immediate = np.maximum(K - S_grid, 0.0)
        
        # Value function at this date
        V_current = np.maximum(immediate, cont_value)
        
        # Find threshold: largest S where immediate >= cont_value
        diff = immediate - cont_value
        pos_indices = np.where(diff >= 0)[0]
        if len(pos_indices) > 0:
            last_pos = pos_indices[-1]
            if last_pos + 1 < len(S_grid):
                d1 = diff[last_pos]
                d2 = diff[last_pos + 1]
                if d2 < 0 and d1 >= 0:
                    frac = d1 / (d1 - d2)
                    threshold = S_grid[last_pos] + frac * (S_grid[last_pos + 1] - S_grid[last_pos])
                else:
                    threshold = S_grid[last_pos]
            else:
                threshold = S_grid[last_pos]
        else:
            threshold = 0.0
        
        thresholds[t_idx] = threshold
        V_next = V_current
    
    # Ensure all thresholds are set
    for i in range(n_dates):
        if thresholds[i] == 0.0:
            thresholds[i] = K
    
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time_idx):
            if time_idx < 0 or time_idx >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[time_idx]
            return np.asarray(current_state) < threshold
    
    return ExercisePolicy(thresholds, K)


def _SimpleThresholdPolicy(K, r, sigma, T, n_dates):
    """Fallback simple threshold policy."""
    dt = T / n_dates
    thresholds = np.zeros(n_dates)
    for i in range(n_dates):
        t_remaining = (n_dates - i) * dt
        alpha = 2.0
        thresholds[i] = K * (1.0 - np.exp(-alpha * t_remaining))
    
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time_idx):
            if time_idx < 0 or time_idx >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[time_idx]
            return np.asarray(current_state) < threshold
    
    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
