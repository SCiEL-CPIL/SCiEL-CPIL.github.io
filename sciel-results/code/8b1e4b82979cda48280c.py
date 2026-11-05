"""Bermudan put exercise-policy research with backward induction on training paths."""
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
    deadline = start_time + 110.0  # Safety margin
    
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
    # We use a local average of k nearest neighbors in the sorted-by-S domain
    k = 50  # Number of neighbors for continuation estimation
    
    for t in range(n_dates - 2, -1, -1):
        if _time.time() > deadline - 5.0:
            # If we run out of time, break and use current V
            # But we need to finish the loop to get thresholds for earlier steps
            # If we break, we might have incomplete V for earlier steps.
            # However, the loop is fast. 12 steps * 4096 paths is small.
            pass
            
        S_t = training_paths[t]
        V_next = V[t + 1]
        
        # Sort indices by S_t
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        # Estimate continuation value E[V_next | S_t] using moving average in sorted domain
        # We compute a cumulative sum to get efficient windowed averages
        cumsum = np.cumsum(sorted_V_next)
        
        # For each index i in sorted order, we want the mean of V_next for indices in [i-k, i+k]
        # Clip indices to [0, n_paths-1]
        # This is a bit tricky with variable window sizes at boundaries.
        # Simpler: use a fixed window and handle boundaries by clipping.
        
        # Let's create an array of continuation values in sorted order
        cont_sorted = np.zeros(n_paths)
        
        # Precompute prefix sums for O(1) range sum
        # prefix_sum[i] = sum(sorted_V_next[0..i-1])
        prefix_sum = np.zeros(n_paths + 1)
        np.cumsum(sorted_V_next, out=prefix_sum[1:])
        
        for i in range(n_paths):
            # Window [i-k, i+k]
            left = max(0, i - k)
            right = min(n_paths, i + k + 1) # exclusive
            
            # Sum in range [left, right)
            sum_val = prefix_sum[right] - prefix_sum[left]
            count = right - left
            
            cont_sorted[i] = sum_val / count
            
        # Discount the continuation value
        cont_sorted *= discount_factor
        
        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_sorted
        
        # Immediate payoff
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Optimal value
        V[t] = np.maximum(payoff_now, cont_values)
        
    # Determine thresholds
    thresholds = np.zeros(n_dates)
    
    for t in range(n_dates):
        S_t = training_paths[t]
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # We need cont_values for t. 
        # If t is the last date, cont is 0 (or irrelevant, V=payoff).
        # For t < n_dates-1, we can re-calculate or store during backward induction.
        # To save time, we can just check where V[t] == payoff_now.
        # But due to floating point, we check if payoff_now >= cont_values.
        # We didn't store cont_values. Let's re-calculate for threshold determination?
        # That doubles the time. 
        # Alternative: During backward induction, store the boolean mask of exercise.
        
        # Let's re-do the logic to store exercise decisions or just use V vs Payoff.
        # If V[t] is very close to payoff_now, it's likely an exercise.
        # But for threshold, we want the boundary.
        
        # Let's re-compute cont for the threshold step only?
        # Or, we can store the cont_values in the backward loop.
        
        # Let's modify the backward loop to store exercise decisions or thresholds directly.
        
    # Re-implementing backward induction to store thresholds is safer.
    
    V = np.zeros((n_dates, n_paths))
    thresholds = np.zeros(n_dates)
    
    V[-1] = np.maximum(K - training_paths[-1], 0.0)
    # At the last date, we always exercise if in the money.
    # Threshold is K (or slightly less due to noise, but effectively K).
    # Actually, if S < K, we exercise. If S >= K, we don't.
    # So threshold is K.
    thresholds[-1] = K
    
    for t in range(n_dates - 2, -1, -1):
        if _time.time() > deadline - 5.0:
            # Fallback: use previous threshold or K
            thresholds[t] = thresholds[t+1]
            continue
            
        S_t = training_paths[t]
        V_next = V[t + 1]
        
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        prefix_sum = np.zeros(n_paths + 1)
        np.cumsum(sorted_V_next, out=prefix_sum[1:])
        
        cont_sorted = np.zeros(n_paths)
        k = 50
        
        for i in range(n_paths):
            left = max(0, i - k)
            right = min(n_paths, i + k + 1)
            sum_val = prefix_sum[right] - prefix_sum[left]
            count = right - left
            cont_sorted[i] = sum_val / count
            
        cont_sorted *= discount_factor
        
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_sorted
        
        payoff_now = np.maximum(K - S_t, 0.0)
        
        V[t] = np.maximum(payoff_now, cont_values)
        
        # Determine threshold
        # Exercise if payoff_now >= cont_values
        exercise_mask = payoff_now >= cont_values
        
        if np.any(exercise_mask):
            # Get S values where exercise is optimal
            S_exercise = S_t[exercise_mask]
            # Use a high percentile to avoid outliers
            # The threshold is the max S where we exercise.
            # Using 95th percentile is robust.
            thresholds[t] = np.percentile(S_exercise, 95)
        else:
            thresholds[t] = 0.0
            
    # Enforce monotonicity: thresholds[t] <= thresholds[t+1]
    for t in range(n_dates - 2, -1, -1):
        if thresholds[t] > thresholds[t+1]:
            thresholds[t] = thresholds[t+1]
            
    # Cap at K
    thresholds = np.minimum(thresholds, K)
    
    return ExercisePolicy(thresholds)

# EVOLVE-BLOCK-END
