# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm

_DEADLINE = 140.0  # seconds; safety margin under 150 s wall-clock limit

def _bs_put_price(S, K, r, sigma, T):
    """
    Calculate European put price using Black-Scholes formula.
    S: array of spot prices
    K: strike
    r: risk-free rate
    sigma: volatility
    T: time to maturity
    """
    S = np.asarray(S, dtype=float)
    if T <= 0:
        return np.maximum(K - S, 0.0)
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    N = norm.cdf
    price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    return price

def _find_bs_threshold(K, r, sigma, t, grid_low=1.0, grid_high=200.0, n_grid=10000):
    """
    Find the optimal exercise threshold for a Bermudan put at time t
    using Black-Scholes continuation value.
    The threshold S* is where intrinsic value (K-S) equals continuation value.
    """
    S_grid = np.linspace(grid_low, grid_high, n_grid)
    intrinsic = np.maximum(K - S_grid, 0.0)
    continuation = _bs_put_price(S_grid, K, r, sigma, t)
    
    # Find where intrinsic >= continuation
    # For a put, we exercise when S is low enough
    # The threshold is the highest S where intrinsic >= continuation
    diff = continuation - intrinsic
    # We want the point where diff changes from negative to positive (or vice versa)
    # Actually, for a put: exercise when K-S >= cont_value
    # So we want the largest S such that K-S >= cont_value(S)
    
    # Find sign changes in (intrinsic - continuation)
    # Exercise when intrinsic - continuation >= 0
    exercise_region = intrinsic >= continuation
    
    # Find the boundary
    if not np.any(exercise_region):
        return grid_low  # Always exercise
    if np.all(exercise_region):
        return grid_high  # Never exercise (but this shouldn't happen for puts)
    
    # Find the last index where exercise_region is True
    # Then interpolate to find the exact threshold
    indices = np.where(exercise_region)[0]
    last_idx = indices[-1]
    
    if last_idx == len(S_grid) - 1:
        return S_grid[last_idx]
    
    # Interpolate between last_idx and last_idx+1
    S1 = S_grid[last_idx]
    S2 = S_grid[last_idx + 1]
    d1 = intrinsic[last_idx] - continuation[last_idx]
    d2 = intrinsic[last_idx + 1] - continuation[last_idx + 1]
    
    if d2 == d1:
        return S1
    threshold = S1 - d1 * (S2 - S1) / (d2 - d1)
    return threshold

def _binom_tree_reference(K, r, sigma, T, n_steps=1200):
    """
    Compute reference values using a binomial tree.
    Returns array of optimal exercise thresholds at each exercise date.
    """
    dt = T / n_steps
    # For monthly exercise dates, we have 12 dates at T/12, 2T/12, ..., 12T/12
    # But the binomial tree has n_steps steps
    # We need to map exercise dates to tree nodes
    
    # Build binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Number of exercise dates: 12 monthly
    n_exercise = 12
    exercise_times = np.linspace(T, T * (n_exercise - 1) / n_exercise, n_exercise)
    # Actually, exercise dates are at T/12, 2T/12, ..., 12T/12 = T
    # Time 0 is not an exercise date
    exercise_times = np.arange(1, n_exercise + 1) * (T / n_exercise)
    
    # For each exercise date, find the corresponding tree step
    # Tree step k corresponds to time k*dt
    # Exercise date j is at time j*T/12
    # So tree step = j*T/12 / dt = j*T/12 * n_steps/T = j*n_steps/12
    
    thresholds = []
    for j in range(n_exercise):
        t_ex = exercise_times[j]
        # Find BS threshold at this time
        threshold = _find_bs_threshold(K, r, sigma, t_ex)
        thresholds.append(threshold)
    
    return thresholds

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using threshold-based approach.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_time_steps) with simulated paths
    - contract: dictionary with contract parameters
    
    Returns:
    - An object with an exercise(current_state, time) method
    """
    t_start = _time.time()
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Compute optimal exercise thresholds at each exercise date using BS continuation value
    exercise_times = np.arange(1, n_exercise + 1) * (T / n_exercise)
    
    # For each exercise date, compute the threshold where intrinsic = continuation
    thresholds = []
    for t_ex in exercise_times:
        remaining = T - t_ex
        if remaining <= 0:
            # At maturity, exercise if intrinsic > 0
            threshold = K
        else:
            threshold = _find_bs_threshold(K, r, sigma, remaining)
        thresholds.append(threshold)
    
    # Convert to numpy array
    thresholds = np.array(thresholds)
    
    # Define the policy class
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times, K, r, sigma, T, n_exercise):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the put option.
            
            Parameters:
            - current_state: array of current spot prices
            - time: current time
            
            Returns:
            - Boolean array indicating whether to exercise
            """
            current_state = np.asarray(current_state, dtype=float)
            
            # Find which exercise date we are at
            # time is the current time, we need to find the closest exercise date
            # Exercise dates are at T/12, 2T/12, ..., 12T/12
            # Find the index of the exercise date closest to current time
            
            # Calculate the exercise index
            # time / (T/n_exercise) gives the exercise number
            ex_idx = int(round(time / (T / self.n_exercise)))
            
            # Clamp to valid range
            ex_idx = max(0, min(ex_idx, self.n_exercise - 1))
            
            # Exercise if current_state < threshold
            threshold = self.thresholds[ex_idx]
            return current_state < threshold
    
    return ExercisePolicy(thresholds, exercise_times, K, r, sigma, T, n_exercise)
# EVOLVE-BLOCK-END
