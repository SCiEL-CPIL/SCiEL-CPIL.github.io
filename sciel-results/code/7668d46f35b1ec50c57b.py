"""Bermudan put exercise-policy research with backward induction using Gaussian Kernel Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def exercise(self, current_state, time_index):
        """
        current_state: array of stock prices
        time_index: integer index of the exercise date (0 to 11)
        Returns: boolean array
        """
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    
    # Determine shape
    # training_paths is expected to be (n_dates, n_paths)
    n_dates, n_paths = training_paths.shape
    dt = T / n_dates
    discount_factor = np.exp(-r * dt)
    
    # Initialize value array
    V = np.zeros((n_dates, n_paths))
    
    # Terminal condition
    V[-1] = np.maximum(K - training_paths[-1], 0.0)
    
    # Backward induction
    # Use Gaussian Kernel Regression for continuation value estimation
    
    thresholds = np.zeros(n_dates)
    thresholds[-1] = K  # At maturity, exercise if in the money (S < K)

    # Precompute kernel bandwidth based on the scale of prices
    # A reasonable bandwidth is related to the standard deviation of log-prices or price differences
    # Let's use a fraction of the range of prices in the training set
    all_S = training_paths.flatten()
    S_range = np.percentile(all_S, 99) - np.percentile(all_S, 1)
    if S_range == 0:
        S_range = 1.0
    # Bandwidth h: typically sqrt(2 * n) / range? Or simpler: range / 20
    # Let's try h = S_range / 30. This gives a local but smooth average.
    h = S_range / 30.0
    
    # For efficiency, we can sort S_t once per step? No, S_t is different at each step.
    # But we can use a vectorized approach for the kernel weights if n_paths is not too large.
    # n_paths = 4096. n_dates = 12.
    # Computing 4096x4096 kernel matrix is 16M operations per step. 12 steps -> 200M ops.
    # This might be slow in pure Python loops, but NumPy vectorization is fast.
    # However, 4096x4096 matrix is 16MB. This is fine.
    
    for t in range(n_dates - 2, -1, -1):
        if _time.time() > deadline - 5.0:
            # Fallback: use previous threshold
            thresholds[t] = thresholds[t+1]
            # We still need V[t] to be defined, but if we break, we can't compute it properly.
            # However, the loop is fast. Let's assume it finishes.
            # If it doesn't, we return with incomplete thresholds, which is bad.
            # Let's try to make the kernel computation faster.
            pass

        S_t = training_paths[t]
        V_next = V[t + 1]
        
        # Sort by S_t to potentially optimize or just for clarity? 
        # Kernel regression doesn't require sorting, but it helps for threshold finding.
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        # Compute Gaussian Kernel Weights
        # W_ij = exp(-0.5 * ((S_i - S_j) / h)^2)
        # To save memory and time, we can compute the difference matrix.
        # diff = S_i - S_j
        # Using broadcasting: S_i[:, None] - S_j[None, :]
        
        # Optimization: If n_paths is large, full matrix is O(N^2).
        # 4096^2 = 16M. exp is expensive.
        # Alternative: Local neighbor search.
        # Since the kernel decays fast, we only need neighbors within ~3h.
        # We can use a sliding window in the sorted array.
        
        # Let's use the sliding window approach with Gaussian weights.
        # Window size: how many points fit in 3h?
        # Average spacing = S_range / n_paths
        # Window size in points ~ 3h / (S_range/n_paths) = 3 * (S_range/30) * (n_paths/S_range) = n_paths / 10 = 409.
        # So we only need to look at ~400 neighbors.
        
        k_window = 500  # Number of neighbors to consider
        
        # Create continuation values
        cont_sorted = np.zeros(n_paths)
        
        # We iterate over each point i in sorted order
        # This is O(N * K). 4096 * 500 = 2M operations. Very fast.
        
        # Precompute the squared differences for the window?
        # For each i, we take S[i-k:i+k+1]
        
        for i in range(n_paths):
            left = max(0, i - k_window)
            right = min(n_paths, i + k_window + 1)
            
            # Get neighbors
            S_neighbors = sorted_S[left:right]
            V_neighbors = sorted_V_next[left:right]
            
            # Compute weights
            # diff = S_neighbors - sorted_S[i]
            diff = S_neighbors - sorted_S[i]
            # weights = exp(-0.5 * (diff/h)^2)
            weights = np.exp(-0.5 * (diff / h) ** 2)
            
            # Weighted average
            sum_w = np.sum(weights)
            if sum_w > 1e-10:
                cont_sorted[i] = np.sum(weights * V_neighbors) / sum_w
            else:
                # Fallback to simple mean if weights are too small (shouldn't happen with h>0)
                cont_sorted[i] = np.mean(V_neighbors)
        
        # Discount the continuation value
        cont_sorted *= discount_factor
        
        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_sorted
        
        # Immediate payoff
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Optimal value
        V[t] = np.maximum(payoff_now, cont_values)
        
        # Determine threshold
        # Exercise if payoff_now >= cont_values
        exercise_mask = payoff_now >= cont_values
        
        if np.any(exercise_mask):
            # Get S values where exercise is optimal
            S_exercise = S_t[exercise_mask]
            # The threshold is the max S where we exercise.
            # To be robust against noise, we can use a high percentile or the max.
            # Given the smoothness of the kernel, the max should be close to the true boundary.
            # Let's use the max to be aggressive, or 99th percentile to be safe.
            # For a put, exercising slightly early when S is close to K might be suboptimal if continuation is high.
            # The condition S < threshold is a hard cutoff.
            # Using the max S_exercise is the most direct estimate.
            thresholds[t] = np.max(S_exercise)
        else:
            thresholds[t] = 0.0
            
    # Enforce monotonicity: thresholds[t] <= thresholds[t+1]
    # For a put, the exercise boundary should be non-decreasing over time (closer to expiry, you exercise at higher prices).
    # So thresholds[0] <= thresholds[1] <= ... <= thresholds[11]
    for t in range(n_dates - 2, -1, -1):
        if thresholds[t] > thresholds[t+1]:
            thresholds[t] = thresholds[t+1]
            
    # Cap at K
    thresholds = np.minimum(thresholds, K)
    
    return ExercisePolicy(thresholds)

# EVOLVE-BLOCK-END
