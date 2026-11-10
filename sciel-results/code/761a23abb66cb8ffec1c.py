"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import bisect

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using Monte Carlo Backward Induction.
    
    Strategy:
    1. Parse contract parameters (K, r, sigma, T, n_exercises).
    2. Identify exercise times from training_paths or contract.
    3. Perform backward induction on the training_paths:
       - Start from the last exercise time (T-1) where continuation value is 0.
       - For each preceding time t, calculate immediate exercise payoff.
       - Estimate continuation value by regressing future discounted payoffs on current state S_t.
         We use a simple piecewise linear interpolation on the sorted training states for robustness
         and speed, which acts as a non-parametric regression.
    4. Determine the optimal exercise threshold for each time step:
       Exercise if Immediate Payoff >= Estimated Continuation Value.
    5. Return an object that uses these thresholds to decide exercise.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Determine number of exercise dates
    # The contract says "12 monthly exercise dates (no time-0 exercise)"
    # Usually, this means exercise at t=1/12, 2/12, ..., 12/12
    # Let's verify if training_paths contain time info or just states.
    # Standard interface: training_paths is likely (n_paths, n_times)
    if training_paths is None or len(training_paths) == 0:
        # Fallback if no training data provided (shouldn't happen per contract)
        return _ThresholdPolicy(np.zeros(12), K)
        
    # training_paths shape: (N, M) where M is number of time steps
    # We assume the last column is T, and columns correspond to exercise dates.
    # If there are 12 exercise dates, and no time-0 exercise, the paths likely have 12 columns
    # representing S at t_1, ..., t_12.
    
    n_paths, n_times = training_paths.shape
    
    # Infer dt
    # If n_times is 12, dt = T/12
    # If n_times is 13, it might include t=0. Contract says "no time-0 exercise".
    # Let's assume the columns provided are the exercise dates.
    # If n_times == 12, they are t_1..t_12.
    # If n_times == 13, they might be t_0..t_12. We need to be careful.
    # The contract says "12 monthly exercise dates".
    # Let's assume the input paths correspond exactly to the exercise dates.
    
    # To be safe, let's check the first path's last element.
    # Actually, we don't know the exact times. But we know T.
    # If n_times == 12, dt = T/12.
    # If n_times > 12, we might have intermediate steps. 
    # However, the standard "Bermudan" setup in these tasks usually provides states at exercise dates only.
    
    n_exercises = n_times
    
    # Backward Induction
    # V_t is the value at time t.
    # We need to estimate E[V_{t+1} | S_t].
    
    # We will store the threshold for each time step.
    # Threshold condition: Exercise if (K - S_t) >= ContinuationValue(S_t)
    # Since (K - S_t) is decreasing in S_t, and ContinuationValue is increasing in S_t (for a put),
    # there exists a threshold S*_t such that we exercise if S_t <= S*_t.
    
    thresholds = np.zeros(n_exercises)
    
    # Start from the last time step (index n_exercises - 1)
    # At the last exercise date, continuation value is 0 (option expires).
    # So we always exercise if K - S > 0.
    # Threshold at last step: S* = K (exercise if S < K).
    # Actually, if S=K, payoff is 0. Continuation is 0. Indifferent.
    # Let's set the "continuation value" array for the last step to be 0 for all S.
    
    # We will maintain arrays of S and V for the *next* time step to regress on.
    # At step t, we have S_t and we want to predict E[ V_{t+1} | S_t ].
    
    # V_{t+1} is known for all paths at time t+1.
    # We need to map S_t -> E[V_{t+1} | S_t].
    
    # Let's work backwards.
    # V_last is the payoff at the last step.
    # For t = n_exercises - 2 down to 0:
    
    # Initialize V for the last step (index n_exercises - 1)
    S_last = training_paths[:, n_exercises - 1]
    V_last = np.maximum(K - S_last, 0.0)
    
    # We need to regress V_last on S_last? No.
    # At step t, we decide based on S_t.
    # The continuation value is E[ V_{t+1} | S_t ].
    
    # Let's iterate from t = n_exercises - 2 down to 0.
    
    # We will keep the sorted S_{t+1} and corresponding V_{t+1} from the previous iteration (which is the "next" step).
    # Wait, the regression is: given S_t, what is the expected V_{t+1}?
    # The pairs are (S_t, V_{t+1}).
    # So we need to sort S_t and interpolate V_{t+1}.
    
    # Let's prepare the data for the last regression (t = n_exercises - 2).
    # Pairs: (S_{n-2}, V_{n-1})
    
    # We'll maintain:
    # next_S: states at time t+1
    # next_V: values at time t+1
    
    next_S = S_last
    next_V = V_last
    
    # We need to compute thresholds for t = n_exercises - 2, ..., 0.
    # Note: The contract says "no time-0 exercise".
    # This implies the first exercise is at t=1.
    # If n_times=12, the indices are 0..11 corresponding to t_1..t_12.
    # So we determine thresholds for all 12 dates.
    
    for t in range(n_exercises - 1, -1, -1):
        if t == n_exercises - 1:
            # Last date: always exercise if ITM.
            # Threshold is K.
            thresholds[t] = K
            continue
            
        # Current states S_t
        curr_S = training_paths[:, t]
        
        # We want to estimate C(S_t) = E[ V_{t+1} | S_t ]
        # We have pairs (curr_S, next_V).
        
        # Sort curr_S and next_V together
        # Note: next_V is the value *after* discounting?
        # No, V_{t+1} is the value at time t+1.
        # The continuation value at time t is exp(-r*dt) * E[ V_{t+1} | S_t ].
        
        # Let's compute the regression function g(S) = E[ V_{t+1} | S ].
        # Then Continuation Value CV(S_t) = exp(-r*dt) * g(S_t).
        # We exercise if (K - S_t) >= CV(S_t).
        
        # Sort by curr_S
        sort_idx = np.argsort(curr_S)
        sorted_S = curr_S[sort_idx]
        sorted_V_next = next_V[sort_idx]
        
        # To estimate E[V_next | S], we can use a simple local average or interpolation.
        # Given the large N (4096), a simple linear interpolation on the sorted data
        # (treating it as a step function or linear) is a reasonable non-parametric estimator.
        # However, simple interpolation of (S, V_next) assumes V_next is a function of S.
        # Due to MC noise, V_next is noisy.
        # A better approach: Bin the S values and take the mean V_next in each bin.
        # Or use a low-order polynomial regression.
        # Given the time constraint and need for robustness, let's use a binned mean approach.
        # This reduces noise and provides a monotonic-ish estimate.
        
        n_bins = 50
        # Create bins
        min_S = np.min(sorted_S)
        max_S = np.max(sorted_S)
        if max_S == min_S:
            bin_edges = np.array([min_S - 1, min_S, max_S + 1])
        else:
            bin_edges = np.linspace(min_S, max_S, n_bins + 1)
            
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_means = np.zeros(n_bins)
        
        # Digitize
        bin_indices = np.digitize(sorted_S, bin_edges) - 1
        # Handle edges
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices >= n_bins] = n_bins - 1
        
        # Sum V_next for each bin
        bin_sums = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, sorted_V_next)
        
        # Count
        bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
        
        # Avoid division by zero
        valid = bin_counts > 0
        bin_means[valid] = bin_sums[valid] / bin_counts[valid]
        
        # For empty bins, interpolate from neighbors
        if np.any(~valid):
            # Simple forward/backward fill
            last_valid = 0
            for i in range(n_bins):
                if valid[i]:
                    bin_means[i] = bin_means[i]
                    last_valid = i
                else:
                    if i == 0:
                        # Use first valid
                        first_valid = np.argmax(valid)
                        bin_means[i] = bin_means[first_valid]
                    else:
                        bin_means[i] = bin_means[last_valid]
                        
            # Backward fill for trailing empty bins
            last_valid = n_bins - 1
            for i in range(n_bins - 1, -1, -1):
                if valid[i]:
                    bin_means[i] = bin_means[i]
                    last_valid = i
                else:
                    if i == n_bins - 1:
                        last_valid_idx = np.where(valid)[0][-1]
                        bin_means[i] = bin_means[last_valid_idx]
                    else:
                        bin_means[i] = bin_means[last_valid]
                        
        # Now bin_means is an estimate of E[V_{t+1} | S] at bin_centers.
        # We can interpolate this to get C(S_t).
        
        # Check time
        if time.time() > deadline - 5.0:
             # Not enough time, use current thresholds for remaining steps (default to K)
             # and break
             for tt in range(t, -1, -1):
                 if tt < n_exercises - 1:
                     thresholds[tt] = K
             break
        
        # Interpolate Continuation Value for all curr_S
        # CV(S) = exp(-r*dt) * interp(S, bin_centers, bin_means)
        
        # Estimate dt
        # If n_times == 12, dt = T/12
        # If n_times == 13, dt = T/12? 
        # Let's assume uniform spacing.
        dt = T / n_exercises
        
        discount_factor = np.exp(-r * dt)
        
        # Interpolate
        # np.interp requires increasing x. bin_centers is increasing.
        # curr_S might have values outside [min_S, max_S]? Unlikely if from same dist, but possible.
        # np.interp handles this by clamping to endpoints.
        
        expected_V_next = np.interp(curr_S, bin_centers, bin_means)
        continuation_values = discount_factor * expected_V_next
        
        # Exercise if (K - S_t) >= continuation_values
        # (K - S_t) - continuation_values >= 0
        # K - S_t - C(S_t) >= 0
        
        # We need the threshold S*_t such that K - S >= C(S).
        # Let H(S) = K - S - C(S).
        # H(S) is decreasing in S (since -S is decreasing, -C(S) is decreasing).
        # We want the largest S such that H(S) >= 0.
        
        # We can find this by evaluating H on the sorted curr_S and finding the crossover.
        # Or simply, since we have the discrete set of S_t, we can find the max S_t that satisfies the condition.
        # But we need a continuous threshold for the `exercise` method.
        
        # Let's find the threshold by solving K - S = C(S).
        # We can approximate this by checking the sorted S values.
        
        # Calculate H for all curr_S
        H = K - curr_S - continuation_values
        
        # We want the max S where H >= 0.
        # Sort S and H together
        sort_idx_H = np.argsort(curr_S)
        sorted_S_H = curr_S[sort_idx_H]
        sorted_H = H[sort_idx_H]
        
        # Find the last index where sorted_H >= 0
        valid_exercise = sorted_H >= 0
        if np.any(valid_exercise):
            last_valid_idx = np.where(valid_exercise)[0][-1]
            # The threshold is somewhere between sorted_S_H[last_valid_idx] and sorted_S_H[last_valid_idx+1]
            # If last_valid_idx is the max index, threshold is max_S_H
            if last_valid_idx == len(sorted_S_H) - 1:
                threshold = sorted_S_H[last_valid_idx]
            else:
                # Linear interpolation of the root
                # H(x) = 0
                # S1, H1 (H1 >= 0)
                # S2, H2 (H2 < 0)
                S1 = sorted_S_H[last_valid_idx]
                H1 = sorted_H[last_valid_idx]
                S2 = sorted_S_H[last_valid_idx + 1]
                H2 = sorted_H[last_valid_idx + 1]
                
                if H2 == H1:
                    threshold = S1
                else:
                    # Root = S1 - H1 * (S2 - S1) / (H2 - H1)
                    threshold = S1 - H1 * (S2 - S1) / (H2 - H1)
        else:
            # No exercise even at lowest S?
            # This implies continuation value is always higher than payoff.
            # Threshold should be 0 (or very small).
            threshold = 0.0
            
        thresholds[t] = threshold
        
        # Update next_S and next_V for the next iteration (t-1)
        # V_t is the max of Exercise and Continuation
        # V_t = max(K - S_t, C(S_t))
        V_curr = np.maximum(K - curr_S, continuation_values)
        
        next_S = curr_S
        next_V = V_curr
        
    # The thresholds array is now filled.
    # thresholds[0] corresponds to the first exercise date (t=1 in time, index 0 in array).
    # thresholds[n_exercises-1] corresponds to the last exercise date.
    
    return _ThresholdPolicy(thresholds, K)

class _ThresholdPolicy:
    def __init__(self, thresholds, K):
        self.thresholds = thresholds
        self.K = K
        
    def exercise(self, current_state, time):
        """
        current_state: array of stock prices for all paths at the current time.
        time: the current time step index or actual time?
        The contract says "time" is passed.
        Usually in these loops, `time` is the index of the current step (0 to n-1).
        Or it could be the actual time t.
        
        If `time` is an integer index (0-based) into the exercise dates:
        We use thresholds[time].
        
        If `time` is a float (actual time):
        We need to map it to the nearest exercise index.
        
        Given "12 monthly exercise dates", and standard implementations, `time` is likely the index.
        Let's assume `time` is the index (0 to 11).
        """
        # Robustness: if time is a float, convert to index
        if isinstance(time, float):
             # Assume T=1.0, dt=1/12. Index = int(time / dt)
             # But we don't have dt here directly, just T from contract? 
             # We stored thresholds. Let's assume the caller passes the index.
             # If it's a float, we round to nearest index.
             # However, the contract says "time" is passed. 
             # Let's check the seed program's interface or common patterns.
             # In many of these tasks, `time` is the index.
             
             # If time is a float like 0.0833, and T=1, n=12, index = 1.
             # Let's try to infer.
             # If we assume time is the index, this is simple.
             
             # Let's handle both. If time is int, use it. If float, convert.
             # But we don't know T in this method. 
             # We can store n_exercises and T in the policy object.
             
             # Let's modify the class to store n_exercises.
             pass
             
        # Assuming `time` is the integer index of the exercise date (0-based)
        # If time is out of bounds, return False (don't exercise)
        
        # To be safe, let's store n_exercises in the object.
        
        idx = int(time)
        if idx < 0 or idx >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
            
        threshold = self.thresholds[idx]
        
        # Exercise if S <= threshold
        return current_state <= threshold

# Refine the class to handle time better
class _ThresholdPolicy:
    def __init__(self, thresholds, K, n_exercises, T):
        self.thresholds = thresholds
        self.K = K
        self.n_exercises = n_exercises
        self.T = T
        self.dt = T / n_exercises if n_exercises > 0 else 1.0

    def exercise(self, current_state, time):
        # Determine the index
        # If time is an integer, it's the index.
        # If time is a float, it's the actual time.
        if isinstance(time, int):
            idx = time
        elif isinstance(time, float):
            # Map time to index
            # t = idx * dt
            # idx = t / dt
            idx = int(round(time / self.dt))
            # Clamp
            if idx < 0:
                idx = 0
            if idx >= self.n_exercises:
                idx = self.n_exercises - 1
        else:
            idx = int(time)
            
        if idx < 0 or idx >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
            
        threshold = self.thresholds[idx]
        return current_state <= threshold

# Re-define fit_exercise_policy to use the updated class
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 100.0
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    if training_paths is None or len(training_paths) == 0:
        n_ex = 12
        return _ThresholdPolicy(np.full(n_ex, K), K, n_ex, T)
        
    n_paths, n_times = training_paths.shape
    n_exercises = n_times
    
    thresholds = np.zeros(n_exercises)
    
    S_last = training_paths[:, n_exercises - 1]
    V_last = np.maximum(K - S_last, 0.0)
    
    next_S = S_last
    next_V = V_last
    
    dt = T / n_exercises
    
    for t in range(n_exercises - 1, -1, -1):
        if t == n_exercises - 1:
            thresholds[t] = K
            continue
            
        curr_S = training_paths[:, t]
        
        sort_idx = np.argsort(curr_S)
        sorted_S = curr_S[sort_idx]
        sorted_V_next = next_V[sort_idx]
        
        n_bins = 50
        min_S = np.min(sorted_S)
        max_S = np.max(sorted_S)
        if max_S == min_S:
            bin_edges = np.array([min_S - 1, min_S, max_S + 1])
        else:
            bin_edges = np.linspace(min_S, max_S, n_bins + 1)
            
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_means = np.zeros(n_bins)
        
        bin_indices = np.digitize(sorted_S, bin_edges) - 1
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices >= n_bins] = n_bins - 1
        
        bin_sums = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, sorted_V_next)
        
        bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
        
        valid = bin_counts > 0
        bin_means[valid] = bin_sums[valid] / bin_counts[valid]
        
        if np.any(~valid):
            last_valid = 0
            for i in range(n_bins):
                if valid[i]:
                    last_valid = i
                else:
                    if i == 0:
                        first_valid = np.argmax(valid)
                        bin_means[i] = bin_means[first_valid]
                    else:
                        bin_means[i] = bin_means[last_valid]
                        
            last_valid = n_bins - 1
            for i in range(n_bins - 1, -1, -1):
                if valid[i]:
                    last_valid = i
                else:
                    if i == n_bins - 1:
                        last_valid_idx = np.where(valid)[0][-1]
                        bin_means[i] = bin_means[last_valid_idx]
                    else:
                        bin_means[i] = bin_means[last_valid]
                        
        if time.time() > deadline - 5.0:
            for tt in range(t, -1, -1):
                if tt < n_exercises - 1:
                    thresholds[tt] = K
            break
            
        discount_factor = np.exp(-r * dt)
        expected_V_next = np.interp(curr_S, bin_centers, bin_means)
        continuation_values = discount_factor * expected_V_next
        
        H = K - curr_S - continuation_values
        
        sort_idx_H = np.argsort(curr_S)
        sorted_S_H = curr_S[sort_idx_H]
        sorted_H = H[sort_idx_H]
        
        valid_exercise = sorted_H >= 0
        if np.any(valid_exercise):
            last_valid_idx = np.where(valid_exercise)[0][-1]
            if last_valid_idx == len(sorted_S_H) - 1:
                threshold = sorted_S_H[last_valid_idx]
            else:
                S1 = sorted_S_H[last_valid_idx]
                H1 = sorted_H[last_valid_idx]
                S2 = sorted_S_H[last_valid_idx + 1]
                H2 = sorted_H[last_valid_idx + 1]
                
                if H2 == H1:
                    threshold = S1
                else:
                    threshold = S1 - H1 * (S2 - S1) / (H2 - H1)
        else:
            threshold = 0.0
            
        thresholds[t] = threshold
        
        V_curr = np.maximum(K - curr_S, continuation_values)
        
        next_S = curr_S
        next_V = V_curr
        
    return _ThresholdPolicy(thresholds, K, n_exercises, T)
# EVOLVE-BLOCK-END
