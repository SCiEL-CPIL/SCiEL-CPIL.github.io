"""Bermudan exercise-policy research with explicit candidate API."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo regression.
    
    Parameters:
    -----------
    training_paths : array of shape (n_paths, n_timesteps+1)
        Simulated price paths under risk-neutral GBM.
    contract : dict with keys for K, r, sigma, T, n_exercise_dates, etc.
    
    Returns:
    --------
    An object with an exercise(current_state, time) method.
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise_dates', 12)
    
    # Exercise times: monthly, no time-0
    exercise_times = np.linspace(T / n_exercise, T, n_exercise)
    
    n_paths = training_paths.shape[0]
    n_steps = training_paths.shape[1] - 1
    dt = T / n_steps
    
    # Determine which column corresponds to each exercise time
    exercise_indices = np.round(exercise_times / dt).astype(int)
    exercise_indices = np.clip(exercise_indices, 0, n_steps)
    
    # Polynomial degree for regression (degree 4 -> 5 basis functions)
    n_basis = 5
    
    # Work backwards through exercise dates
    reg_coeffs = []
    
    # Last exercise date: continuation value = 0
    reg_coeffs.append(None)
    
    # Track which paths are still alive (not yet exercised)
    alive = np.ones(n_paths, dtype=bool)
    
    # Optimal payoffs at each date
    opt_payoffs = [None] * n_exercise
    
    # Last date: always exercise if in the money
    idx_last = exercise_indices[-1]
    S_last = training_paths[:, idx_last]
    opt_payoffs[-1] = np.maximum(K - S_last, 0.0)
    
    # Work backwards from second-to-last to first
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > DEADLINE:
            # Fill remaining with None (fallback to always exercise if in money)
            for j in range(i, -1, -1):
                reg_coeffs.insert(0, None)
            break
        
        idx_i = exercise_indices[i]
        
        S_i = training_paths[:, idx_i]
        
        # Discount factor from date i to date i+1
        dt_exercise = exercise_times[i + 1] - exercise_times[i]
        discount = np.exp(-r * dt_exercise)
        
        # Continuation value = discount * optimal payoff at next date
        cont_val = discount * opt_payoffs[i + 1]
        
        # Fit regression using alive paths
        alive_mask = alive
        n_alive = np.sum(alive_mask)
        
        if n_alive < n_basis + 2:
            # Not enough data, use all paths
            alive_mask = np.ones(n_paths, dtype=bool)
        
        S_norm = S_i[alive_mask] / K
        y = cont_val[alive_mask]
        
        # Filter out non-finite values
        valid = np.isfinite(S_norm) & np.isfinite(y)
        if np.sum(valid) < n_basis:
            coeffs = np.zeros(n_basis)
            if np.sum(valid) > 0:
                coeffs[0] = np.mean(y[valid])
        else:
            S_norm = S_norm[valid]
            y = y[valid]
            X = np.vander(S_norm, n_basis, increasing=True)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                if not np.all(np.isfinite(coeffs)):
                    coeffs = np.zeros(n_basis)
                    coeffs[0] = np.mean(y)
            except:
                coeffs = np.zeros(n_basis)
                coeffs[0] = np.mean(y)
        
        reg_coeffs.insert(0, coeffs)
        
        # Compute optimal payoff at date i
        S_norm_all = S_i / K
        valid_all = np.isfinite(S_norm_all)
        fitted_cont_all = np.zeros(n_paths)
        if np.any(valid_all):
            fitted_cont_all[valid_all] = np.polynomial.polynomial.polyval(S_norm_all[valid_all], coeffs)
        fitted_cont_all = np.maximum(fitted_cont_all, 0.0)
        
        immediate = np.maximum(K - S_i, 0.0)
        exercise_now = immediate > fitted_cont_all
        
        # Optimal payoff: if exercise, get immediate; if continue, get fitted_cont
        opt_payoffs[i] = np.where(exercise_now, immediate, fitted_cont_all)
        
        # Update alive: paths that exercised are no longer alive
        alive[exercise_now & alive] = False
    
    # Create the policy object
    class BermudanPolicy:
        def __init__(self, reg_coeffs, exercise_times, exercise_indices, K, n_exercise, n_steps, T, dt):
            self.reg_coeffs = reg_coeffs
            self.exercise_times = exercise_times
            self.exercise_indices = exercise_indices
            self.K = K
            self.n_exercise = n_exercise
            self.n_steps = n_steps
            self.T = T
            self.dt = dt
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given state and time.
            
            Parameters:
            -----------
            current_state : float or array of floats
                Current spot price(s).
            time : float
                Current time.
            
            Returns:
            --------
            Boolean or boolean array: True if exercise.
            """
            # Find the closest exercise date
            diffs = np.abs(self.exercise_times - time)
            idx = int(np.argmin(diffs))
            
            # Check if we're close enough to an exercise date
            if diffs[idx] > self.dt * 0.6:
                if np.isscalar(current_state):
                    return False
                else:
                    return np.zeros(len(current_state), dtype=bool)
            
            S = np.asarray(current_state, dtype=float)
            scalar_input = np.isscalar(current_state)
            if scalar_input:
                S = S.reshape(1)
            
            immediate = np.maximum(self.K - S, 0.0)
            
            if self.reg_coeffs[idx] is None:
                # Fallback: always exercise if in the money
                result = immediate > 0
            else:
                S_norm = S / self.K
                valid = np.isfinite(S_norm)
                fitted_cont = np.zeros(len(S))
                if np.any(valid):
                    fitted_cont[valid] = np.polynomial.polynomial.polyval(S_norm[valid], self.reg_coeffs[idx])
                fitted_cont = np.maximum(fitted_cont, 0.0)
                result = immediate > fitted_cont
            
            if scalar_input:
                return bool(result[0])
            else:
                return result
    
    return BermudanPolicy(reg_coeffs, exercise_times, exercise_indices, K, n_exercise, n_steps, T, dt)
# EVOLVE-BLOCK-END

