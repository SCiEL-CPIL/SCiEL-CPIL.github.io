"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Leave 5s buffer for evaluation overhead
    
    # Parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # training_paths shape: (n_paths, n_time_steps) or similar
    # Assume training_paths is array of shape (n_paths, n_dates) where each row is a path
    # with prices at each exercise date (excluding time 0)
    paths = np.asarray(training_paths, dtype=np.float64)
    
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
    
    n_paths = paths.shape[0]
    n_dates = paths.shape[1]
    
    # We have 12 exercise dates. paths[:, i] gives spot at exercise date i+1 (1-indexed)
    # Payoff if exercised at date i (0-indexed, i=0..11): max(K - S_i, 0)
    # Discounting: payoff at date i is discounted by exp(-r * (i+1) * dt)
    
    # Backward induction
    # V[i, s] = max(K - s, 0) if s <= threshold_i, else continuation value
    # Continuation value at date i = discount * E[V[i+1, S_{i+1}] | S_i = s]
    
    # Use quantile-based binning for the value function
    n_bins = 200
    
    # Store thresholds for each exercise date
    thresholds = np.zeros(n_dates)
    
    # Value function at the last exercise date
    # At the last date, always exercise (no continuation)
    # V[11, s] = max(K - s, 0)
    
    # For earlier dates, we need to estimate continuation value
    # Use the training paths to estimate E[S_{i+1} | S_i]
    
    # For each date i from n_dates-1 down to 0:
    #   Compute continuation value as function of S_i
    #   Find threshold where exercise payoff = continuation value
    
    # Build value function representation using piecewise linear interpolation
    # on a grid of S values
    
    # Determine S range from training data
    all_s = paths.flatten()
    s_min = max(np.percentile(all_s, 1), 1.0)
    s_max = min(np.percentile(all_s, 99), K * 3)
    
    # Create grid
    s_grid = np.linspace(s_min, s_max, n_bins)
    
    # V_next[s] will store the value function at the next exercise date
    # Initialize for the last date
    V_next = np.maximum(K - s_grid, 0.0)
    
    # Store thresholds
    thresholds = np.zeros(n_dates)
    
    # Backward induction
    for i in range(n_dates - 1, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Get S_i and S_{i+1} from training paths
        S_i = paths[:, i]
        
        if i < n_dates - 1:
            S_next = paths[:, i + 1]
        else:
            # At the last date, no continuation
            thresholds[i] = K  # Always exercise
            continue
        
        # For each grid point s, estimate E[V_next[S_{i+1}] | S_i = s]
        # Use kernel regression / local linear
        # For efficiency, use binned approach
        
        # Bin S_i into quantiles
        n_train = len(S_i)
        # Use all training paths
        
        # Estimate conditional expectation using local averaging
        # For each grid point, find nearby S_i values and average V_next[S_next]
        
        # Efficient approach: sort by S_i and use cumulative sums
        order = np.argsort(S_i)
        S_i_sorted = S_i[order]
        S_next_sorted = S_next[order]
        
        # For each grid point, find the fraction of paths with S_i near that point
        # and average V_next[S_next] for those paths
        
        # Use a simple approach: for each grid point, take the median of V_next[S_next]
        # for paths where S_i is within a small window
        
        # More efficient: use np.searchsorted to find indices
        continuation_vals = np.zeros(n_bins)
        
        # Window size for local averaging
        window_frac = 0.05  # 5% of paths on each side
        
        for j in range(n_bins):
            s_j = s_grid[j]
            # Find paths where S_i is close to s_j
            # Use a bandwidth based on local density
            idx_left = np.searchsorted(S_i_sorted, s_j - 0.5 * (s_max - s_min) / n_bins)
            idx_right = np.searchsorted(S_i_sorted, s_j + 0.5 * (s_max - s_min) / n_bins)
            
            if idx_right > idx_left:
                # Average V_next[S_next] for these paths
                V_next_at_next = np.maximum(K - S_next_sorted[idx_left:idx_right], 0.0)
                continuation_vals[j] = np.mean(V_next_at_next)
            else:
                # No data, use interpolation
                continuation_vals[j] = np.interp(s_j, s_grid, V_next)
        
        # Discount the continuation value
        continuation_vals = discount * continuation_vals
        
        # Exercise payoff at grid points
        exercise_payoff = np.maximum(K - s_grid, 0.0)
        
        # Find threshold where exercise_payoff = continuation_vals
        # For a put, exercise when K - S >= continuation, i.e., S <= K - continuation
        # So threshold_i = K - continuation at the crossing point
        
        # Find the crossing point
        diff = exercise_payoff - continuation_vals
        # We want the largest s where diff >= 0 (exercise region)
        
        # The optimal threshold is where it's optimal to exercise
        # For a put, exercise when S is low enough
        # Find the point where diff changes sign from + to -
        
        # Find all indices where diff >= 0
        exercise_mask = diff >= 0
        if np.any(exercise_mask):
            # The threshold is the maximum s where we still exercise
            threshold_idx = np.max(np.where(exercise_mask)[0])
            # Interpolate to find exact threshold
            if threshold_idx < n_bins - 1:
                # Linear interpolation
                d0 = diff[threshold_idx]
                d1 = diff[threshold_idx + 1]
                if d1 != d0:
                    frac = d0 / (d0 - d1)
                    thresholds[i] = s_grid[threshold_idx] + frac * (s_grid[threshold_idx + 1] - s_grid[threshold_idx])
                else:
                    thresholds[i] = s_grid[threshold_idx]
            else:
                thresholds[i] = s_grid[-1]
        else:
            # Never exercise (shouldn't happen for a put with K=100)
            thresholds[i] = s_min
        
        # Update V_next for the next iteration
        V_next = np.where(s_grid <= thresholds[i], exercise_payoff, continuation_vals / discount if discount > 0 else continuation_vals)
        # Actually, V at this date = max(exercise, continuation)
        V_next = np.maximum(exercise_payoff, continuation_vals)
    
    # Coordinate descent refinement of thresholds
    # For each threshold, try small perturbations and check if value improves
    # This is done by simulating on training paths
    
    def compute_policy_value(thr):
        """Compute the mean discounted payoff using the given thresholds."""
        total_value = np.zeros(n_paths)
        exercised = np.zeros(n_paths, dtype=bool)
        
        for i in range(n_dates):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            S_i = paths[:, i]
            # Exercise if S_i <= threshold[i] and not yet exercised
            do_exercise = (S_i <= thr[i]) & (~exercised)
            if np.any(do_exercise):
                payoff = np.maximum(K - S_i[do_exercise], 0.0)
                disc = np.exp(-r * (i + 1) * dt)
                total_value[do_exercise] += disc * payoff
                exercised[do_exercise] = True
        
        return np.mean(total_value)
    
    # Refine thresholds via coordinate descent
    n_cd_iters = 3
    for cd_iter in range(n_cd_iters):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
        
        current_val = compute_policy_value(thresholds)
        
        for i in range(n_dates):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Try small perturbations
            best_thr = thresholds[i]
            best_val = current_val
            
            # Perturbation sizes
            pert_sizes = [0.5, 1.0, 2.0, 5.0]
            for ps in pert_sizes:
                for sign in [1, -1]:
                    new_thr = thresholds[i] + sign * ps
                    if new_thr < s_min or new_thr > s_max:
                        continue
                    old_thr = thresholds[i]
                    thresholds[i] = new_thr
                    val = compute_policy_value(thresholds)
                    thresholds[i] = old_thr
                    
                    if val > best_val:
                        best_val = val
                        best_thr = new_thr
            
            thresholds[i] = best_thr
            current_val = best_val
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, s_min, s_max, K, r, dt):
            self.thresholds = thresholds
            self.s_min = s_min
            self.s_max = s_max
            self.K = K
            self.r = r
            self.dt = dt
            self.n_dates = len(thresholds)
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices for active paths
            time: current time (continuous) or exercise date index
            
            Returns: boolean vector indicating whether to exercise
            """
            # Determine which exercise date we're at
            # time is in [0, T], exercise dates are at dt, 2*dt, ..., 12*dt
            # time might be the continuous time, so we need to figure out the date index
            
            # If time is exactly at an exercise date, use that date
            # time = i * dt for exercise date i (1-indexed: date 1 is at dt)
            
            # Convert time to exercise date index
            # Exercise dates are at t = dt, 2*dt, ..., 12*dt
            # If time is close to (i+1)*dt, we're at exercise date i (0-indexed)
            
            if isinstance(time, (int, np.integer)):
                # time might be the date index directly
                date_idx = int(time)
                if date_idx < 0 or date_idx >= self.n_dates:
                    return np.zeros(len(current_state), dtype=bool)
            else:
                # time is continuous
                t = float(time)
                # Find which exercise date
                date_idx = int(np.round(t / self.dt)) - 1
                if date_idx < 0 or date_idx >= self.n_dates:
                    return np.zeros(len(current_state), dtype=bool)
            
            S = np.asarray(current_state, dtype=np.float64)
            return S <= self.thresholds[date_idx]
    
    return ExercisePolicy(thresholds, s_min, s_max, K, r, dt)
# EVOLVE-BLOCK-END
