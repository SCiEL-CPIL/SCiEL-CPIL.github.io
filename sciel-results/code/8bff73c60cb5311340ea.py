"""Bermudan put exercise-policy research with backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds, r, dt):
        self.thresholds = thresholds
        self.r = r
        self.dt = dt

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

    # Extract parameters from contract
    # Assuming contract has keys like 'S0', 'K', 'r', 'sigma', 'T', 'n_dates'
    # Based on task description: Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1, 12 dates
    # The 'training_paths' likely contains the simulated paths.
    
    # We need to know the shape of training_paths.
    # Usually (n_dates, n_paths) or (n_paths, n_dates).
    # Task says "12 monthly exercise dates".
    # Let's assume training_paths is (n_dates, n_paths).
    
    n_dates, n_paths = training_paths.shape
    
    # Get parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    dt = T / n_dates
    discount_factor = np.exp(-r * dt)

    # Initialize values
    # V is the value of the option at each step for each path
    V = np.zeros((n_dates, n_paths))
    
    # Terminal payoff
    V[-1] = np.maximum(K - training_paths[-1], 0.0)
    
    # Backward Induction
    # We will store the exercise decisions to derive thresholds later
    # Or we can just derive thresholds from the final V calculation
    
    # We need to estimate continuation values.
    # We'll use binning on the current state S_t.
    n_bins = 50
    
    for t in range(n_dates - 2, -1, -1):
        S_t = training_paths[t]
        V_next = V[t + 1]
        
        # Estimate E[V_next | S_t] using binning
        # Sort indices by S_t
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V_next = V_next[sorted_indices]
        
        # Create bin edges
        # Use quantiles to ensure roughly equal number of points per bin
        bin_edges = np.quantile(sorted_S, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges for digitize
        bin_edges = np.unique(bin_edges)
        
        # Assign each sorted point to a bin
        bin_indices = np.digitize(sorted_S, bin_edges) - 1
        # Clip to valid range
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        # Calculate mean V_next for each bin
        bin_sums = np.zeros(len(bin_edges) - 1)
        bin_counts = np.zeros(len(bin_edges) - 1)
        
        np.add.at(bin_sums, bin_indices, sorted_V_next)
        np.add.at(bin_counts, bin_indices, 1)
        
        # Avoid division by zero
        bin_counts[bin_counts == 0] = 1
        bin_means = bin_sums / bin_counts
        
        # For each path, get the bin mean corresponding to its S_t
        # We need to map back to original indices
        # Create an array of continuation values for sorted paths
        cont_values_sorted = bin_means[bin_indices]
        
        # Map back to original order
        cont_values = np.zeros(n_paths)
        cont_values[sorted_indices] = cont_values_sorted
        
        # Discount the continuation value
        cont_values *= discount_factor
        
        # Payoff if exercised now
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # Optimal value
        V[t] = np.maximum(payoff_now, cont_values)
        
        # Check deadline
        if _time.time() > deadline - 10.0:
            # If we are running out of time, we can break and use current thresholds
            # But the loop is fast, so this is unlikely.
            break

    # Determine thresholds
    # For each time t, the policy is: Exercise if S_t < threshold
    # We find the threshold by looking at the exercise decisions in the training set.
    # Exercise happened if payoff_now >= cont_values (with tie-breaking to exercise)
    # Actually, V[t] = max(payoff, cont). If payoff > cont, we exercise.
    # Let's re-evaluate exercise decisions for threshold determination.
    
    thresholds = np.zeros(n_dates)
    
    # We need to re-calculate or store the exercise decisions.
    # Let's redo the logic to store decisions, or just infer from V and S.
    # If V[t] == payoff, it means we exercised (or were indifferent).
    # For a put, the region is S < K.
    
    for t in range(n_dates):
        S_t = training_paths[t]
        payoff_now = np.maximum(K - S_t, 0.0)
        
        # If V[t] is close to payoff, we exercised.
        # To be precise, let's find the max S where V[t] == payoff_now (and payoff_now > 0)
        # Or simply, find the cutoff.
        
        # In the optimal policy for a put, there is a threshold.
        # Let's find the highest S_t where we would exercise.
        # We can approximate this by finding where V[t] transitions from "continuation-dominated" to "payoff-dominated".
        
        # A robust way:
        # Sort S_t
        sorted_indices = np.argsort(S_t)
        sorted_S = S_t[sorted_indices]
        sorted_V = V[t][sorted_indices]
        sorted_Payoff = payoff_now[sorted_indices]
        
        # Find the last index where V[t] is determined by payoff
        # i.e., V[t] == payoff (within tolerance)
        # And payoff > 0
        
        # Actually, if V[t] > payoff, we continue. If V[t] == payoff, we exercise.
        # Due to floating point, we check if V[t] is "close" to payoff.
        
        # Let's assume the threshold is the maximum S among exercised paths.
        # We need to identify exercised paths.
        # Let's re-compute cont_values for this t to be sure, or just use the fact that
        # for S < K, if V[t] == payoff, it's an exercise.
        
        # Simpler heuristic:
        # The threshold is the point where the value function "kinks".
        # For a put, it's the max S such that Exercise is optimal.
        
        # Let's find the max S_t where V[t] == payoff_now
        # We'll use a tolerance
        tol = 1e-8
        is_exercise = np.abs(V[t] - payoff_now) < tol
        # Also require payoff > 0
        is_exercise = is_exercise & (payoff_now > 0)
        
        if np.any(is_exercise):
            thresholds[t] = np.max(S_t[is_exercise])
        else:
            thresholds[t] = 0.0
            
        # If we never exercise, threshold is 0.
        # If we always exercise (for S<K), threshold is K (or max S).
        
    return ExercisePolicy(thresholds, r, dt)

# EVOLVE-BLOCK-END
