"""Bermudan exercise-policy research: backward induction with antithetic MC for continuation values."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import interp1d

class BermudanExercisePolicy:
    """
    Bermudan put exercise policy using backward induction with Monte Carlo
    estimation of continuation values on a state grid.
    
    The policy is fitted using training paths and evaluated on independent paths.
    """
    
    def __init__(self, training_paths, contract):
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s limit
        
        # Parse contract parameters
        self.S0 = contract.get('S0', 100.0)
        self.K = contract.get('K', 100.0)
        self.r = contract.get('r', 0.05)
        self.sigma = contract.get('sigma', 0.2)
        self.T = contract.get('T', 1.0)
        self.n_exercise = contract.get('n_exercise', 12)
        self.n_paths = contract.get('n_paths', 4096)
        
        # Training paths shape: (n_paths, n_timesteps+1)
        self.training_paths = np.asarray(training_paths)
        n_paths, n_steps = self.training_paths.shape
        
        # Exercise times (monthly, no time-0)
        self.dt = self.T / self.n_exercise
        self.exercise_times = np.arange(1, self.n_exercise + 1) * self.dt
        # Map exercise dates to indices in the path array
        # If paths have n_steps+1 points (time 0 to T), exercise at step i
        self.exercise_indices = np.arange(1, self.n_exercise + 1)
        
        # Build state grid for continuation value estimation
        # Use a reasonable range of stock prices
        S_min = max(0.01, self.S0 * 0.1)
        S_max = self.S0 * 3.0
        n_grid = 200
        self.S_grid = np.linspace(S_min, S_max, n_grid)
        
        # Number of MC samples for continuation estimation
        n_mc = min(n_paths, 4096)
        
        # Backward induction
        # continuation_values[i] is a function of state S at exercise date i
        # We store it as values on the grid
        self.cont_values = []  # list of arrays, one per exercise date (from last to first)
        
        # Start from the last exercise date
        # At the last exercise date, continuation = 0 (no more exercise opportunities)
        # So we exercise if payoff > 0, which for a put means always exercise if S < K
        
        # For each exercise date from last to first, compute continuation value
        # continuation at date i = E[discounted max(exercise at i+1, continue)]
        
        # We'll store continuation values on the grid for each exercise date
        # cont_grid[i] = continuation value at exercise date i as function of S
        # Exercise at date i if payoff(S) > cont_grid[i](S)
        
        n_ex = self.n_exercise
        cont_grid = [None] * n_ex  # cont_grid[i] for exercise date i (0-indexed)
        
        # At the last exercise date (index n_ex-1), continuation = 0
        # So we always exercise if S < K (for a put)
        # But we need to be more careful: the continuation value at date n_ex-1
        # is the expected discounted payoff from future exercise dates, which is 0
        # since there are no more exercise dates after the last one.
        
        # Actually, let me reconsider. We have n_exercise exercise dates.
        # Let's index them 0 to n_exercise-1.
        # At exercise date j (0-indexed), if we don't exercise, we can exercise at dates j+1, ..., n_exercise-1.
        # So the continuation value at date j is the expected discounted value of optimal stopping from j+1 onwards.
        
        # Backward induction:
        # V(n_exercise-1, S) = max(K - S, 0)  [exercise is optimal if S < K, else 0]
        #   Actually, at the last date, we must decide: exercise or not.
        #   If we don't exercise, we get 0 (no more dates).
        #   So V(last, S) = max(K - S, 0)
        
        # V(j, S) = max(K - S, 0, E[exp(-r*dt) * V(j+1, S_{j+1}) | S_j = S])
        
        # Let's compute this using MC on the training paths.
        
        # For efficiency, we'll use the training paths directly.
        # For each exercise date j, we need to estimate:
        #   E[exp(-r*dt) * V(j+1, S_{j+1}) | S_j]
        
        # We'll use a grid-based approach with interpolation.
        
        # Initialize: V at last exercise date
        # V_last(S) = max(K - S, 0)
        V_last = np.maximum(self.K - self.S_grid, 0.0)
        self.cont_grid = [None] * n_ex
        self.cont_grid[n_ex - 1] = V_last  # This is the value if we continue from last-1, but actually...
        
        # Wait, let me restructure.
        # Let value[j][i] = optimal value at exercise date j given state S_grid[i]
        # value[n_ex-1][i] = max(K - S_grid[i], 0)
        # value[j][i] = max(K - S_grid[i], E[exp(-r*dt) * value[j+1] | S_j = S_grid[i]])
        
        # For the expectation, we use MC: for each path, at exercise date j,
        # if S_j is near S_grid[i], we look at S_{j+1} and evaluate value[j+1] at S_{j+1}.
        
        # This is computationally expensive. Let's use a simpler approach:
        # For each exercise date j, compute the continuation value as a function of S
        # using the training paths.
        
        # Alternative: use the training paths to estimate the conditional expectation.
        # For each exercise date j, group paths by S_j value, then average
        # exp(-r*dt) * value[j+1](S_{j+1}).
        
        # Let's implement this step by step.
        
        # First, compute value at the last exercise date
        value_next = np.maximum(self.K - self.S_grid, 0.0)
        
        # Store values for each exercise date
        self.values = [None] * n_ex
        self.values[n_ex - 1] = value_next
        
        # Now go backwards
        for j in range(n_ex - 2, -1, -1):
            if time.time() - self.start_time > self.deadline:
                # Time up, use simple heuristic for remaining dates
                for k in range(j, -1, -1):
                    self.values[k] = np.maximum(self.K - self.S_grid, 0.0)
                break
            
            # Compute continuation value at exercise date j
            # For each point on the grid, estimate E[exp(-r*dt) * value_next(S_{j+1}) | S_j = S_grid[i]]
            
            # Use the training paths
            S_j = self.training_paths[:, j + 1]  # state at exercise date j (1-indexed in path)
            S_j1 = self.training_paths[:, j + 2]  # state at next exercise date
            
            # For each grid point, find paths where S_j is close to S_grid[i]
            # Use binning for efficiency
            
            n_bins = n_grid
            bin_edges = np.linspace(S_min, S_max, n_bins + 1)
            
            # Assign paths to bins based on S_j
            bin_indices = np.digitize(S_j, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # For each bin, compute the average of exp(-r*dt) * value_next(S_j1)
            disc = np.exp(-self.r * self.dt)
            
            # Interpolate value_next at S_j1
            # Use linear interpolation
            from scipy.interpolate import interp1d
            interp_fn = interp1d(self.S_grid, value_next, kind='linear', 
                                  bounds_error=False, fill_value=(value_next[0], value_next[-1]))
            value_next_at_Sj1 = interp_fn(S_j1)
            
            # Compute continuation value for each bin
            cont_values_bin = np.zeros(n_bins)
            counts = np.zeros(n_bins)
            
            for b in range(n_bins):
                mask = (bin_indices == b)
                if np.any(mask):
                    cont_values_bin[b] = disc * np.mean(value_next_at_Sj1[mask])
                    counts[b] = np.sum(mask)
                else:
                    # No paths in this bin, use nearest non-empty bin
                    # For simplicity, use 0 or interpolate later
                    cont_values_bin[b] = 0.0
            
            # Interpolate to grid points
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            interp_cont = interp1d(bin_centers, cont_values_bin, kind='linear',
                                    bounds_error=False, fill_value=(cont_values_bin[0], cont_values_bin[-1]))
            cont_value_grid = interp_cont(self.S_grid)
            
            # Value at exercise date j = max(immediate payoff, continuation value)
            payoff = np.maximum(self.K - self.S_grid, 0.0)
            value_j = np.maximum(payoff, cont_value_grid)
            
            self.values[j] = value_j
            value_next = value_j  # For next iteration
        
        # Precompute interpolation functions for each exercise date
        self.interp_funcs = []
        for j in range(n_ex):
            if self.values[j] is not None:
                func = interp1d(self.S_grid, self.values[j], kind='linear',
                                 bounds_error=False, fill_value=(self.values[j][0], self.values[j][-1]))
            else:
                func = interp1d(self.S_grid, np.maximum(self.K - self.S_grid, 0.0), kind='linear',
                                 bounds_error=False, fill_value=(0.0, 0.0))
            self.interp_funcs.append(func)
        
        # Also store the grid and values for direct lookup
        self.S_grid_arr = self.S_grid
        self.values_arr = self.values
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time and state.
        
        Parameters:
        -----------
        current_state : array of stock prices
        time : current time (or index)
        
        Returns:
        --------
        Boolean array indicating whether to exercise
        """
        # Determine which exercise date we're at
        # time could be a float in [0, T] or an integer index
        if isinstance(time, (int, np.integer)):
            # Integer index into exercise dates
            j = int(time)
            if j < 0:
                j = 0
            elif j >= self.n_exercise:
                j = self.n_exercise - 1
        else:
            # Float time, map to exercise date index
            t = float(time)
            j = int(round(t / self.dt)) - 1
            if j < 0:
                j = 0
            elif j >= self.n_exercise:
                j = self.n_exercise - 1
        
        # Get the value function at this exercise date
        if self.values_arr[j] is not None:
            value_fn = self.interp_funcs[j]
        else:
            value_fn = self.interp_funcs[-1]
        
        S = np.asarray(current_state, dtype=float)
        
        # Immediate payoff
        payoff = np.maximum(self.K - S, 0.0)
        
        # Continuation value (value if we don't exercise now)
        # This is the value function minus the immediate payoff if we were to exercise
        # Actually, the value function already includes the max, so:
        # We exercise if payoff >= continuation_value
        # continuation_value = E[exp(-r*dt) * V(next)]
        # But we stored V(j) = max(payoff, cont)
        # So we need to recover cont from V(j)
        
        # Simpler: we exercise if S is below a threshold
        # The threshold is where payoff = continuation value
        # Let's just compare payoff to the value function
        # If value_fn(S) > payoff, we don't exercise (continuation is better)
        # If value_fn(S) <= payoff, we exercise
        
        val = value_fn(S)
        exercise_mask = payoff >= val - 1e-10
        
        return exercise_mask.astype(bool)


def fit_exercise_policy(training_paths, contract):
    """
    Fit the Bermudan exercise policy using training paths.
    
    Parameters:
    -----------
    training_paths : array of shape (n_paths, n_timesteps+1)
    contract : dict with contract parameters
    
    Returns:
    --------
    BermudanExercisePolicy object
    """
    return BermudanExercisePolicy(training_paths, contract)
# EVOLVE-BLOCK-END
