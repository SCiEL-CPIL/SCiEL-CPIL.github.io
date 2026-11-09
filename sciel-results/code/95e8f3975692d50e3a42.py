"""
Bermudan Put Exercise Policy via Parameterized Threshold Optimization.

Strategy:
1. Model the exercise policy as a set of 12 thresholds $b_0, \dots, b_{11}$.
   The rule is: Exercise at exercise date $i$ (time $t_i$) if $S_t \le b_i$.
2. The objective is to maximize the mean discounted payoff across the training paths.
3. Use the training paths to estimate the expected payoff for a given set of thresholds.
4. Optimize the thresholds using numerical optimization (Nelder-Mead or L-BFGS-B).
5. The resulting policy is a simple threshold check, which is fast and robust.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Determine number of exercise dates
    if 'n_steps' in contract:
        n_exercise = contract['n_steps']
    elif 'n_exercise' in contract:
        n_exercise = contract['n_exercise']
    else:
        try:
            n_cols = training_paths.shape[1]
            n_exercise = n_cols - 1
        except:
            n_exercise = 12

    dt = T / n_exercise
    disc = np.exp(-r * dt)
    
    # Training paths shape: (n_paths, n_exercise + 1)
    # Columns 0..n_exercise correspond to t=0..T
    # Exercise dates are at indices 1..n_exercise
    n_paths = training_paths.shape[0]
    
    # Precompute discount factors for each exercise date
    # Payoff at time i (index i in paths) is discounted by exp(-r * t_i)
    # t_i = i * dt
    t_times = np.arange(1, n_exercise + 1) * dt
    discount_factors = np.exp(-r * t_times)
    
    # Objective function: minimize negative mean discounted payoff
    def objective(b):
        # b: array of shape (n_exercise,)
        # Ensure b is within [0, K]
        b = np.clip(b, 0.0, K)
        
        # We simulate the payoff for all paths
        # payoff_sum = 0
        # For each path, we iterate through time.
        # This is vectorizable!
        
        # S_t is training_paths[:, t]
        # We need to determine the first time t where S_t <= b[t-1].
        # If no such time, payoff is at T.
        
        # Let's create a matrix of shape (n_paths, n_exercise)
        # S_exercise = training_paths[:, 1:n_exercise+1]
        S_ex = training_paths[:, 1:n_exercise+1]
        
        # Exercise decision matrix: True if S_ex <= b
        # b has shape (n_exercise,), S_ex has shape (n_paths, n_exercise)
        # We need to broadcast b across paths.
        exercise_decision = S_ex <= b[None, :]
        
        # Find the first exercise time for each path.
        # If a path exercises at time i (0-indexed in S_ex), we take payoff (K - S_ex[i])
        # discounted by discount_factors[i].
        # If a path never exercises, we take payoff (K - S_ex[-1]) discounted by discount_factors[-1].
        
        # np.argmax returns the first True index. If all False, returns 0?
        # Actually, if all False, argmax returns 0 by default if we use any?
        # We need to be careful.
        
        # Let's find the index of the first True for each row.
        # If no True, we want to treat it as the last time step.
        
        # Create a mask where we have exercised.
        # first_exercise_idx = np.argmax(exercise_decision, axis=1)
        # has_exercised = exercise_decision.any(axis=1)
        
        # If not has_exercised, use last index.
        # last_idx = n_exercise - 1
        
        # This logic is tricky to vectorize perfectly for the payoff calculation
        # because the payoff value depends on the index.
        
        # Alternative: Loop over time steps in reverse? No.
        # Let's just loop over time steps forward, but vectorized over paths.
        # We can maintain a "still_active" mask.
        
        active = np.ones(n_paths, dtype=bool)
        total_payoff = np.zeros(n_paths)
        
        for i in range(n_exercise):
            # Check exercise condition for active paths
            # S_ex[:, i] <= b[i]
            ex_mask = active & (S_ex[:, i] <= b[i])
            
            # Calculate payoff for those who exercised now
            if np.any(ex_mask):
                payoffs = K - S_ex[ex_mask, i]
                total_payoff[ex_mask] = payoffs * discount_factors[i]
                active[ex_mask] = False
            
            # If time runs out, the remaining active paths take the terminal payoff
            # This happens after the loop if we don't handle it inside.
            # Actually, the last step (i = n_exercise - 1) is maturity.
            # If they didn't exercise before, they MUST take the payoff (or exercise, same thing).
            # So after the loop, any path still active takes the payoff at T.
        
        # Handle terminal payoff for paths that never exercised
        if np.any(active):
            terminal_payoffs = K - S_ex[active, -1]
            total_payoff[active] = terminal_payoffs * discount_factors[-1]
            
        # Mean payoff
        mean_payoff = np.mean(total_payoff)
        
        return -mean_payoff

    # Initial guess for thresholds
    # A good initial guess is a linear interpolation from some low value to K.
    # Or use a simple binomial tree approximation for the first step?
    # Let's start with a geometric progression or linear.
    # For a put, boundary is between 0 and K.
    # At t=0 (first exercise), boundary is low. At t=T, boundary is K.
    # Let's try b_i = K * (i / n_exercise)
    b0 = np.linspace(0.5 * K, K, n_exercise)
    
    # Optimization
    # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness (from argmax/clip) better.
    # However, Nelder-Mead can be slow for 12 dimensions if not careful.
    # L-BFGS-B might be faster if we smooth the objective?
    # The objective is piecewise linear/constant, so gradients are not well-defined.
    # Nelder-Mead is safer.
    
    # To speed up, we can limit the number of function evaluations.
    # Each evaluation is ~ n_paths * n_exercise operations.
    # 4096 * 12 = ~50k ops. Very fast.
    # Nelder-Mead with 12 dims might need ~100-200 evaluations.
    # 200 * 50k = 10M ops. Trivial.
    
    try:
        # Set time limit for optimization
        # We have ~110s. Optimization should take < 5s.
        res = minimize(objective, b0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
        
        if res.success or res.fun < objective(b0):
            b_opt = res.x
        else:
            b_opt = b0
            
    except Exception as e:
        # Fallback to initial guess if optimization fails
        b_opt = b0
        # print(f"Optimization failed: {e}")

    # Ensure b_opt is within bounds
    b_opt = np.clip(b_opt, 0.0, K)
    
    # Enforce monotonicity? 
    # Theoretically, the boundary should be non-decreasing.
    # Nelder-Mead might produce non-monotonic results.
    # We can enforce monotonicity by taking the cumulative maximum.
    # b_mono[i] = max(b[0], ..., b[i])
    # This ensures b_0 <= b_1 <= ...
    b_final = np.maximum.accumulate(b_opt)
    
    # Final check on time
    if time.time() > deadline:
        # Return a simple fallback policy if we are too late
        # Just use K as threshold for all steps (exercise if ITM)
        b_final = np.full(n_exercise, K)

    class ExercisePolicy:
        def __init__(self, thresholds):
            self.thresholds = thresholds
            self.n_exercise = len(thresholds)
            
        def exercise(self, current_state, time):
            # time is the index of the exercise date (0-based).
            # current_state is an array of S values.
            if time < 0 or time >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            threshold = self.thresholds[time]
            return current_state <= threshold

    return ExercisePolicy(b_final)

# EVOLVE-BLOCK-END
