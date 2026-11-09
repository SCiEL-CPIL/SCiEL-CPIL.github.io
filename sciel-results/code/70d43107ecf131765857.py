import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Infer structure from training_paths
    # training_paths should be (N_paths, N_times) where N_times includes t=0
    # 12 monthly exercise dates means 13 time points (0 to 12)
    N_paths, N_times = training_paths.shape
    
    # Number of exercise dates is N_times - 1 (no exercise at t=0)
    n_exercise = N_times - 1
    dt = T / n_exercise
    
    # Precompute discount factors
    disc = np.exp(-r * np.arange(N_times) * dt)
    
    # We want to learn a policy that maximizes expected discounted payoff
    # For a Bermudan put: payoff = max(K - S, 0)
    # The optimal policy at each time t is to exercise if the immediate payoff
    # is greater than the continuation value
    
    # Strategy: Use a threshold-based policy learned from the training data
    # For each exercise date, find the optimal threshold S* such that
    # we exercise if S <= S*
    
    # Better approach: Use dynamic programming on a grid
    # But first, let's try a simpler approach using quantile-based thresholds
    
    # For each exercise date j (from 1 to n_exercise), we want to find
    # the threshold below which we should exercise
    
    # Method: For each path and each exercise date, compute the "optimal"
    # decision by looking at whether the path would have benefited from exercising
    # earlier vs. waiting
    
    # Simpler approach: Use the fact that for a put, the optimal exercise region
    # is S <= S*(t). We can estimate S*(t) by finding the critical stock price
    # where the immediate payoff equals the estimated continuation value.
    
    # Let's use a regression-based approach:
    # For each time step, fit a model that predicts the continuation value
    # as a function of S, then exercise if K-S > continuation_value(S)
    
    # Actually, let's use a simpler but effective approach:
    # Use the training paths to estimate the value function via regression
    
    # First, let's compute the "naive" value: assume we never exercise until maturity
    # Then improve iteratively
    
    # Better: Use a grid-based approach with interpolation
    
    # Let's use a piecewise-linear value function approximation
    # For each exercise date, create a grid of stock prices and estimate
    # the continuation value at each grid point
    
    # Determine grid
    # Stock prices in training data
    S_all = training_paths[:, 1:]  # Exclude t=0
    S_min = np.percentile(S_all, 1)
    S_max = np.percentile(S_all, 99)
    
    # Create grid
    n_grid = 500
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Initialize value function (continuation value) at maturity
    # At maturity (last time step), continuation value = 0 (no future)
    # Payoff at maturity = max(K - S, 0)
    
    # Work backwards from maturity
    # V_j(S) = max(K - S, discount * E[V_{j+1}(S_{j+1}) | S_j = S])
    
    # For simplicity, let's use a threshold-based policy
    # Estimate the optimal threshold for each exercise date
    
    # Approach: For each exercise date j, find the stock price S* such that
    # paths with S < S* would have had higher payoff by exercising at j
    # than by continuing
    
    # Let's use a simple iterative method:
    # 1. Start with a naive policy (exercise at maturity only)
    # 2. For each time step backwards, estimate the continuation value
    # 3. Determine the exercise threshold
    
    # Even simpler: Use the training data to directly estimate optimal decisions
    
    # For each path and each exercise date, we can compute:
    # - The payoff if we exercise now: K - S_t
    # - The best payoff we could get from continuing: max over future exercise dates
    
    # But this is circular. Let's use a different approach.
    
    # Use a neural network or random forest? No, let's keep it simple.
    
    # Approach: Quantile-based thresholds
    # For each exercise date j, the optimal threshold is approximately
    # where the density of paths that would benefit from exercising changes.
    
    # Let's try: For each exercise date j, compute the median stock price
    # of paths that end up with high payoff if exercised at j vs. continuing.
    
    # Actually, let's use a more principled approach:
    # Estimate the value function using local regression
    
    # For each exercise date j (from n_exercise down to 1):
    #   For each grid point S in S_grid:
    #     Find paths with S_j close to S
    #     Compute the average discounted continuation value
    #     The continuation value is the average of max(K-S_{j+1}, V_{j+1}(S_{j+1}))
    
    # This is computationally expensive. Let's simplify.
    
    # Use a parametric model: Assume the value function is of the form
    # V_j(S) = a_j * S + b_j for S < S*_j, and 0 for S >= S*_j
    # (This is an approximation for a put)
    
    # Actually, for a put, the value function is convex and decreasing in S.
    # Let's use a simple two-parameter model: V_j(S) = max(c_j - S, 0) * d_j
    
    # This is getting complicated. Let's use a simpler heuristic.
    
    # Heuristic: For a Bermudan put, the optimal exercise threshold
    # increases as we get closer to maturity (because the time value decreases).
    
    # Let's estimate the threshold for each exercise date using the training data.
    
    # For each exercise date j, define:
    #   benefit_of_exercising = K - S_j
    #   benefit_of_continuing = expected future payoff if we don't exercise now
    
    # We can estimate benefit_of_continuing by looking at what happens to paths
    # that don't exercise at j.
    
    # Simple approach: Assume that if we continue, the best we can do is
    # the maximum payoff over all future exercise dates.
    
    # For each path i and exercise date j:
    #   future_max_payoff_i_j = max_{k > j} (K - S_i_k)
    #   exercise_now_payoff_i_j = K - S_i_j
    #   optimal_decision_i_j = exercise if exercise_now_payoff_i_j >= future_max_payoff_i_j
    
    # This is a myopic approximation but might work reasonably well.
    
    # Let's compute this for all paths and exercise dates
    
    S_future = training_paths[:, 1:]  # (N_paths, n_exercise)
    K_arr = np.full_like(S_future, K)
    
    # Payoff if exercised at each date
    payoff_if_exercise = K_arr - S_future  # (N_paths, n_exercise)
    
    # For each path and each exercise date, compute the max future payoff
    # future_max_payoff[:, j] = max(payoff_if_exercise[:, j+1:])
    
    future_max = np.zeros_like(payoff_if_exercise)
    # For the last exercise date, future_max = 0 (no future)
    for j in range(n_exercise - 1, -1, -1):
        if j == n_exercise - 1:
            future_max[:, j] = 0.0
        else:
            future_max[:, j] = np.max(payoff_if_exercise[:, j+1:], axis=1)
    
    # Optimal decision: exercise if payoff_if_exercise >= future_max
    # But this is not quite right because future_max doesn't account for discounting
    
    # Let's include discounting
    # The discounted future max payoff at time j is:
    #   disc[j] * max_{k > j} (disc[k] / disc[j] * (K - S_k))
    # = max_{k > j} (disc[k] * (K - S_k)) / disc[j]
    
    # Let's recompute with discounting
    
    # Discounted payoff if exercised at date k: disc[k] * (K - S_k)
    disc_payoff = disc[1:] * payoff_if_exercise  # (N_paths, n_exercise), disc[1:] for dates 1..n_exercise
    
    # For each path and exercise date j, the max discounted future payoff
    future_max_disc = np.zeros_like(disc_payoff)
    for j in range(n_exercise - 1, -1, -1):
        if j == n_exercise - 1:
            future_max_disc[:, j] = 0.0
        else:
            future_max_disc[:, j] = np.max(disc_payoff[:, j+1:], axis=1)
    
    # Exercise if disc[j] * (K - S_j) >= future_max_disc[:, j]
    # i.e., payoff_if_exercise[:, j] >= future_max_disc[:, j] / disc[j]
    
    thresholds = np.zeros(n_exercise)
    for j in range(n_exercise):
        if j == n_exercise - 1:
            # At maturity, always exercise if in the money
            thresholds[j] = K
        else:
            # Find the threshold where payoff_if_exercise = future_max_disc / disc[j]
            # payoff_if_exercise[:, j] = K - S_future[:, j]
            # future_max_disc[:, j] / disc[j] is the continuation value in undiscounted terms
            cont_val = future_max_disc[:, j] / disc[j]
            # Exercise if K - S_future[:, j] >= cont_val
            # i.e., S_future[:, j] <= K - cont_val
            # The threshold is the median or some quantile of (K - cont_val)
            # Let's use the median
            thresh = np.median(K - cont_val)
            thresholds[j] = thresh
    
    # Create the policy object
    class Policy:
        def __init__(self, thresholds, n_exercise, T, dt):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
            self.T = T
            self.dt = dt
            # Precompute time-to-step mapping
            # The exercise method receives current_state and time
            # time is likely the current time in [0, T]
            # We need to map time to the exercise date index
            
        def exercise(self, current_state, time):
            # current_state: array of stock prices at the current time
            # time: scalar, current time
            # Return: boolean array, True if we should exercise
            
            # Determine which exercise date we're at
            # Exercise dates are at t = dt, 2*dt, ..., T
            # time should be one of these values (or close to it)
            
            # Find the exercise date index
            # time / dt gives the step number
            step = int(round(time / self.dt))
            
            # Clamp to valid range
            step = max(0, min(step, self.n_exercise - 1))
            
            # Get the threshold for this exercise date
            thresh = self.thresholds[step]
            
            # Exercise if S <= thresh
            return current_state <= thresh
    
    # Check time
    if time.time() > deadline:
        # Return a simple policy
        class SimplePolicy:
            def __init__(self, K, n_exercise, dt, T):
                self.K = K
                self.n_exercise = n_exercise
                self.dt = dt
                self.T = T
            
            def exercise(self, current_state, time):
                step = int(round(time / self.dt))
                step = max(0, min(step, self.n_exercise - 1))
                # Exercise if in the money
                return current_state <= self.K
        return SimplePolicy(K, n_exercise, dt, T)
    
    return Policy(thresholds, n_exercise, T, dt)
