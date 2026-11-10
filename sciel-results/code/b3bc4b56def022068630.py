"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using a non-parametric,
    piecewise-constant continuation value estimator via backward induction on
    training paths.
    
    Strategy:
    1. Discretize the state space into bins at each exercise date.
    2. Backward induction: for each date j from last to first, estimate the
       continuation value C_j(S) as the mean of discounted optimal values at
       j+1, grouped by the bin of S_j.
    3. Exercise if intrinsic value >= estimated continuation value.
    4. The policy is a lookup table: for each (time, bin), store a boolean
       exercise decision.
    
    This avoids parametric assumptions (polynomials) and coordinate ascent,
    using pure empirical estimation from the training paths.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin within 120s limit
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    dt = T / n_exercises
    disc = np.exp(-r * dt)
    
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
        
    n_paths, n_cols = paths.shape
    
    # Determine offset for exercise dates
    if n_cols == 13:
        exercise_paths = paths[:, 1:13]
    elif n_cols == 12:
        exercise_paths = paths[:, 0:12]
    else:
        exercise_paths = paths[:, -12:]
    
    # State space discretization
    # For a put with K=100, relevant range is roughly [0, 1.5*K]
    # Use log-spaced bins for better resolution at low prices
    n_bins = 200
    S_min = 1e-4
    S_max = 1.5 * K
    bin_edges = np.geomspace(S_min, S_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Precompute bin indices for all paths at all steps
    # bin_idx[p, j] = bin index of S[p, j]
    S_clipped = np.clip(exercise_paths, S_min, S_max)
    bin_idx = np.zeros((n_paths, n_exercises), dtype=np.int32)
    for j in range(n_exercises):
        bin_idx[:, j] = np.searchsorted(bin_edges, S_clipped[:, j], side='right') - 1
        bin_idx[:, j] = np.clip(bin_idx[:, j], 0, n_bins - 1)
    
    # Backward induction
    # optimal_val[p, j] = max(K - S[p,j], disc * E[optimal_val[p, j+1] | S[p,j]])
    # We estimate E[optimal_val[p, j+1] | S[p,j]] by grouping on bin_idx[:, j]
    
    optimal_val = np.zeros((n_paths, n_exercises))
    
    # Last step: optimal value is intrinsic
    optimal_val[:, -1] = np.maximum(K - exercise_paths[:, -1], 0.0)
    
    # For each step j from n_exercises-2 down to 0:
    #   target[p] = disc * optimal_val[p, j+1]
    #   For each bin b: cont_val[b] = mean(target[p] for p with bin_idx[p,j] == b)
    #   For paths with no data in their bin, use nearest bin or 0
    #   optimal_val[p, j] = max(K - S[p,j], cont_val[bin_idx[p,j]])
    
    # Store continuation value per bin per step for the policy
    # cont_per_bin[j, b] = estimated continuation value at step j, bin b
    cont_per_bin = np.zeros((n_exercises, n_bins))
    
    # For the last step, continuation is 0
    cont_per_bin[-1, :] = 0.0
    
    for j in range(n_exercises - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Emergency: fill remaining with intrinsic (exercise always)
            for jj in range(j, -1, -1):
                cont_per_bin[jj, :] = 0.0
                optimal_val[:, jj] = np.maximum(K - exercise_paths[:, jj], 0.0)
            break
        
        S_j = exercise_paths[:, j]
        intrinsic_j = np.maximum(K - S_j, 0.0)
        target = disc * optimal_val[:, j + 1]
        
        # Compute mean target per bin
        bin_j = bin_idx[:, j]
        cont_vals = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=np.int32)
        
        # Use np.bincount for efficiency
        # We need sum(target) and count per bin
        # np.bincount with weights
        sums = np.bincount(bin_j, weights=target, minlength=n_bins)
        counts = np.bincount(bin_j, minlength=n_bins)
        
        # Mean where count > 0, else 0
        nonzero = counts > 0
        cont_vals[nonzero] = sums[nonzero] / counts[nonzero]
        
        # For bins with no data, interpolate from neighbors or use 0
        # Simple approach: forward-fill from nearest bin with data
        if not np.all(nonzero):
            # Find bins with data
            data_bins = np.where(nonzero)[0]
            if len(data_bins) > 0:
                # For each empty bin, find nearest data bin
                # This is O(n_bins * n_data_bins) but n_bins=200, so fine
                for b in range(n_bins):
                    if not nonzero[b]:
                        # Find nearest data bin
                        dists = np.abs(data_bins - b)
                        nearest = data_bins[np.argmin(dists)]
                        cont_vals[b] = cont_vals[nearest]
        
        cont_per_bin[j, :] = cont_vals
        
        # Compute optimal value for each path at step j
        cont_for_paths = cont_vals[bin_j]
        optimal_val[:, j] = np.maximum(intrinsic_j, cont_for_paths)
    
    # Now build the policy: for each (time, bin), exercise if intrinsic >= continuation
    # Intrinsic at bin center b: K - bin_centers[b] (clipped to 0)
    intrinsic_per_bin = np.maximum(K - bin_centers, 0.0)
    
    # exercise_decision[j, b] = True if intrinsic_per_bin[b] >= cont_per_bin[j, b]
    exercise_decision = np.zeros((n_exercises, n_bins), dtype=bool)
    for j in range(n_exercises):
        exercise_decision[j, :] = intrinsic_per_bin >= cont_per_bin[j, :]
    
    # For the last step, always exercise if ITM (intrinsic > 0)
    exercise_decision[-1, :] = intrinsic_per_bin > 0
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, exercise_decision, bin_edges, bin_idx_func):
            self.exercise_decision = exercise_decision
            self.bin_edges = bin_edges
            self.bin_idx_func = bin_idx_func
            self.n_exercises = exercise_decision.shape[0]
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices for all paths at this time step.
            time: the time step index (0-based, 0 to 11 for 12 dates)
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            S = np.asarray(current_state)
            S_clipped = np.clip(S, self.bin_edges[0], self.bin_edges[-1])
            bin_j = np.searchsorted(self.bin_edges, S_clipped, side='right') - 1
            bin_j = np.clip(bin_j, 0, len(self.bin_edges) - 2)
            
            return self.exercise_decision[time, bin_j]
    
    return ExercisePolicy(exercise_decision, bin_edges, None)
# EVOLVE-BLOCK-END
