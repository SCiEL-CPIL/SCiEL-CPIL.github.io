"""
Strategy: Non-parametric backward induction using local weighted regression.
Instead of assuming a parametric boundary (like a constant strike or linear decay) 
or a fixed grid (binomial tree), we use the training paths to estimate the 
continuation value at each exercise date via kernel regression (Nadaraya-Watson).
This allows the policy to capture complex state-dependencies directly from the 
sample paths, which is particularly useful if the optimal boundary has non-trivial 
curvature or if the training distribution differs from the standard GBM assumption 
in higher moments.

The algorithm:
1. Parse training paths to extract spot prices at each of the 12 exercise dates.
2. For each date t from T-1 down to 0:
   a. Calculate the intrinsic payoff at date t: max(K - S_t, 0).
   b. Estimate the continuation value C_t(S_t) using the known values at t+1.
      - We have the values V_{t+1}(S_{t+1}) from the previous step.
      - We regress V_{t+1} on S_t using a Gaussian kernel.
      - The predicted value is the discounted expected continuation.
   c. The optimal value at t is max(Intrinsic, Continuation).
   d. Store the exercise decision: Exercise if Intrinsic > Continuation.
3. During evaluation, for a given state S_t at date t, we need to decide whether to exercise.
   - We can't easily re-run the regression for every evaluation point quickly enough if we do it naively.
   - Instead, we fit a simple monotonic approximation (e.g., piecewise linear or polynomial) 
     to the "exercise boundary" or the "continuation value surface" derived from the training data.
   - Alternatively, since the state space is 1D (spot price), we can pre-compute the 
     continuation value function on a dense grid of S values using the kernel regression 
     on the training data, and then use interpolation during evaluation.

Given the 150s time limit and 4096 paths:
- 12 steps * 4096 points is small.
- Kernel regression with O(N^2) complexity per step might be slow if done naively for a dense grid.
- Optimization: Use a fixed grid of spot prices for the value function approximation.
  Let S_grid be a vector of e.g., 200 points ranging from 0.5*K to 2*K.
  For each step t:
    1. We have the training spots S_train_t (shape N).
    2. We have the values V_train_{t+1} (shape N) from the next step.
    3. We want to estimate V_cont(s) for s in S_grid.
    4. V_cont(s) = discount * (sum_i K(s, S_train_t[i]) * V_train_{t+1}[i]) / (sum_i K(s, S_train_t[i]))
    5. This is O(|S_grid| * N) per step. 200 * 4096 * 12 ~ 10^7 operations. Very fast in NumPy.
    6. Then V_train_t[i] = max(K - S_train_t[i], V_cont(S_train_t[i])).
    7. We also store the exercise decision for the training points, but for evaluation, 
       we use the grid: Exercise if (K - S) > V_cont_interp(S).

This approach is structurally different from:
- Binomial tree: Uses empirical paths, not a theoretical lattice.
- Parametric boundary: Uses data-driven regression, not a fixed formula.
- Neural Network: Uses kernel regression, which is simpler, deterministic, and doesn't require 
  hyperparameter tuning (bandwidth can be fixed or Silverman's rule) or gradient descent.
"""

import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = 12
    
    # Handle input shape
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)
    
    N, total_cols = training_paths.shape
    
    # Determine if column 0 is t=0 or t=1
    # Standard Bermudan: 12 exercise dates.
    # If total_cols == 13, it includes t=0. Exercise dates are indices 1..12.
    # If total_cols == 12, exercise dates are indices 0..11.
    if total_cols == num_steps + 1:
        exercise_indices = list(range(1, num_steps + 1))
    elif total_cols == num_steps:
        exercise_indices = list(range(0, num_steps))
    else:
        # Fallback: assume last 12 columns
        exercise_indices = list(range(total_cols - num_steps, total_cols))
        
    # Extract spots at exercise dates
    # Shape: (num_steps, N) -> transpose to (N, num_steps)
    spots = training_paths[:, exercise_indices].T  # (12, N)
    
    # Discount factor per step
    dt = T / num_steps
    disc = np.exp(-r * dt)
    
    # Define a grid of spot prices for value function approximation
    # Range: 0.5 * min_spot to 1.5 * max_spot, clipped to reasonable bounds
    min_s = np.min(spots)
    max_s = np.max(spots)
    s_min = max(0.1, 0.5 * min_s)
    s_max = 1.5 * max_s
    grid_size = 300
    S_grid = np.linspace(s_min, s_max, grid_size)
    
    # Initialize value at the final step (T)
    # At the final exercise date, the option is European (or just payoff).
    # Payoff = max(K - S_T, 0)
    V_next = np.maximum(K - spots[-1], 0.0)
    
    # Store exercise boundaries/decisions for each step
    # We will store the continuation value function on the grid for each step
    # and the intrinsic value.
    # Actually, we just need to know if we exercise.
    # We can store the "continuation value" grid for each step t=0..10.
    # For step t, we compare intrinsic at S vs continuation_value_grid_t[S].
    
    cont_grids = []  # List of grids for steps 0 to 10
    
    # Bandwidth for kernel regression
    # Silverman's rule of thumb for 1D
    # h = 1.06 * sigma * N^(-1/5)
    # We estimate sigma from the spots at each step, or use a global estimate.
    # Since it's GBM, sigma of log S is constant. Let's use a fixed bandwidth 
    # relative to the scale of S.
    # A safer approach: adaptive bandwidth based on local density? 
    # For simplicity and speed, use a fixed bandwidth that is a fraction of the range.
    # Or compute per-step bandwidth.
    
    for t in reversed(range(num_steps - 1)):  # t = 10 down to 0
        if time.time() > deadline:
            break
            
        S_t = spots[t]
        S_next = spots[t + 1]
        
        # Calculate intrinsic value at current step
        intrinsic = np.maximum(K - S_t, 0.0)
        
        # Estimate continuation value on the grid S_grid
        # V_cont(s) = disc * E[V_next | S_t = s]
        # Using Nadaraya-Watson estimator:
        # V_cont(s) = disc * sum_i K(s, S_t[i]) * V_next[i] / sum_i K(s, S_t[i])
        
        # Kernel: Gaussian
        # Bandwidth h: Let's use a fraction of the standard deviation of S_t
        std_s = np.std(S_t)
        if std_s < 1e-6:
            std_s = 1.0
        h = 0.5 * std_s  # Adjust factor as needed
        
        # Vectorized computation for all grid points
        # S_grid shape: (G,)
        # S_t shape: (N,)
        # Diff shape: (G, N)
        diff = S_grid[:, None] - S_t[None, :]
        # Weights: exp(-0.5 * (diff/h)^2)
        weights = np.exp(-0.5 * (diff / h) ** 2)
        
        # Numerator: sum(weights * V_next, axis=1)
        num = np.sum(weights * V_next[None, :], axis=1)
        # Denominator: sum(weights, axis=1)
        den = np.sum(weights, axis=1)
        
        # Avoid division by zero
        den[den < 1e-10] = 1e-10
        
        V_cont_grid = disc * num / den
        
        # Store the continuation value grid for this step
        # We need to interpolate this grid during evaluation
        cont_grids.append(V_cont_grid)
        
        # Update V_next for the current step's training points
        # V_t[i] = max(intrinsic[i], V_cont(S_t[i]))
        # Interpolate V_cont_grid to S_t
        V_cont_at_train = np.interp(S_t, S_grid, V_cont_grid)
        V_next = np.maximum(intrinsic, V_cont_at_train)
        
        # Optional: Prune or adjust grid if time is tight? No, 300x4096 is fast.
        
    # Reverse cont_grids to be in chronological order (0 to 10)
    cont_grids.reverse()
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, K, S_grid, cont_grids):
            self.K = K
            self.S_grid = S_grid
            self.cont_grids = cont_grids  # List of arrays, index 0 is step 0, etc.
            self.num_steps = len(cont_grids)
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices for the active paths at this time step.
            time: integer index of the current exercise date (0-indexed? or 1-indexed?)
                   The contract says "12 monthly exercise dates".
                   Usually time in these APIs is the step index 0..11.
                   We assume time is 0-indexed corresponding to the first exercise date.
            """
            # Ensure current_state is a numpy array
            S = np.asarray(current_state, dtype=float)
            
            # If time is out of bounds, return False (no exercise)
            if time < 0 or time >= self.num_steps:
                return np.zeros_like(S, dtype=bool)
            
            # Get intrinsic value
            intrinsic = np.maximum(self.K - S, 0.0)
            
            # Get continuation value by interpolating from the grid
            V_cont = np.interp(S, self.S_grid, self.cont_grids[time])
            
            # Exercise if intrinsic > continuation
            # Strictly greater to avoid unnecessary exercise if equal
            # (though for puts, exercise is often optimal at the boundary)
            return intrinsic > V_cont

    return ExercisePolicy(K, S_grid, cont_grids)
