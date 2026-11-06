"""Bermudan exercise policy using Binomial Tree Backward Induction."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin (limit is 150s)
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # We don't strictly need training_paths for the tree method, 
    # but we must accept it.
    
    # Binomial Tree Parameters
    # N steps should be high enough for accuracy. 2000 is fast and accurate.
    N = 2000
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    # Precompute exercise date indices in the tree
    # Exercise dates are at t = 1/12, 2/12, ..., 12/12
    # Tree time step i corresponds to t = i * dt
    # We need to map exercise dates to tree indices
    # exercise_date_k = k * T / n_exercise
    # tree_index_k = round(exercise_date_k / dt) = round(k * N / n_exercise)
    
    exercise_indices = []
    for k in range(1, n_exercise + 1):
        idx = int(round(k * N / n_exercise))
        # Ensure indices are unique and within bounds
        if not exercise_indices or idx > exercise_indices[-1]:
            exercise_indices.append(idx)
        
    # Reverse the list so we iterate backwards from maturity
    # The last exercise date is at N (or close to it)
    # We need to handle the final step carefully.
    # If N is not divisible by 12, the last index might be N-1 or N.
    # Let's force the last exercise date to be exactly at N (maturity) if possible, 
    # or just use the closest index.
    # For a Bermudan, the last exercise is at T. So index N.
    # Let's adjust the last index to be N.
    exercise_indices[-1] = N
    
    # Initialize value array at maturity (step N)
    # Number of nodes at step N is N+1
    V = np.zeros(N + 1)
    
    # Stock prices at step N
    S_N = K * (u ** np.arange(N + 1)) * (d ** np.arange(N, -1, -1))
    # Wait, standard binomial: S_{i,j} = S_0 * u^j * d^(i-j)
    # At step N, j ranges from 0 to N.
    # S_{N,j} = S_0 * u^j * d^(N-j)
    # We don't have S_0 explicitly in contract, but we can infer or just use relative prices?
    # Actually, the critical price is an absolute value. We need S_0.
    # The contract doesn't provide S_0 directly, but training_paths do.
    # training_paths[:, 0] gives S_0 for each path. They should be identical or very close.
    S_0 = np.median(training_paths[:, 0])
    
    # Recalculate S_N with S_0
    S_N = S_0 * (u ** np.arange(N + 1)) * (d ** np.arange(N, -1, -1))
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(n_exercise)
    critical_prices[-1] = K # At maturity, exercise if S < K
    
    # Backward induction
    # We only need to update at exercise dates, but we must propagate values through all steps.
    # However, we can optimize by only storing the value array.
    # We iterate i from N down to 1.
    
    # To save time, we can check if we are at an exercise date.
    # We need to know which tree indices are exercise dates.
    # Let's create a set of exercise indices for O(1) lookup.
    exercise_idx_set = set(exercise_indices)
    # Map tree index to exercise date index (1..12) for storing critical price
    # We need to know which critical price array slot to fill.
    # exercise_indices is sorted ascending. 
    # Let's create a dict: tree_idx -> exercise_k (1-based)
    idx_to_ex_k = {}
    for k, idx in enumerate(exercise_indices):
        idx_to_ex_k[idx] = k + 1
        
    # Iterate backwards
    # V represents values at step i+1
    # We compute values at step i
    
    # Optimization: 
    # The tree has N+1 nodes at step N, N nodes at step N-1, etc.
    # V_current length = i+1 (for step i)
    # V_next length = i+2 (for step i+1)
    
    # Start with V at step N
    # V is length N+1
    # We step down to i = N-1 ... 1
    
    # We need S at each step to evaluate exercise.
    # S_{i,j} = S_0 * u^j * d^(i-j)
    
    for i in range(N, 0, -1):
        if time.time() > DEADLINE:
            break
            
        # V currently holds values for step i
        # We need to compute values for step i-1
        
        # Number of nodes at step i is i+1
        # Number of nodes at step i-1 is i
        
        # Continuation value at step i-1, node j:
        # C_{i-1,j} = discount * (p * V_{i,j} + (1-p) * V_{i,j+1})
        
        V_prev = np.zeros(i)
        V_prev[:] = discount_factor * (p * V[0:i] + (1-p) * V[1:i+1])
        
        # Check if step i-1 is an exercise date
        # Note: Exercise dates are at t = k*T/12.
        # Tree step j corresponds to t = j*dt.
        # So we check if i-1 is in exercise_idx_set.
        
        if (i-1) in exercise_idx_set:
            k = idx_to_ex_k[i-1]
            # Compute Stock prices at step i-1
            # j ranges from 0 to i-2
            j = np.arange(i)
            S_i_minus_1 = S_0 * (u ** j) * (d ** (i-1 - j))
            
            # Payoff
            payoff = np.maximum(K - S_i_minus_1, 0.0)
            
            # Decide exercise
            # Exercise if payoff > continuation
            exercise_mask = payoff > V_prev
            
            # Update V_prev
            V_prev = np.where(exercise_mask, payoff, V_prev)
            
            # Find critical price
            # We want S* where payoff = continuation.
            # Find the index where the switch happens.
            # If all exercise, S* = max(S)
            # If all hold, S* = min(S)
            # Otherwise, find boundary.
            
            # Since S is increasing with j, and for a put, we exercise for low S,
            # the mask should be True for low j and False for high j.
            
            if np.all(exercise_mask):
                S_star = S_i_minus_1[-1]
            elif not np.any(exercise_mask):
                S_star = S_i_minus_1[0]
            else:
                # Find the last index where exercise_mask is True
                # idx_switch = max j such that mask[j] is True
                # The switch is between j and j+1
                # mask[j] is True, mask[j+1] is False
                # S_star is between S[j] and S[j+1]
                
                # Use np.where to find transitions
                # We want the last True before a False
                # Since it's monotonic, there is only one transition.
                # Find indices where mask changes from True to False
                # diff = mask[:-1] - mask[1:]
                # We want diff > 0 (True - False = 1)
                
                # Simpler: find the index of the last True
                # But we need the boundary.
                # Let's find the first False index from the right?
                # Or just find the index j where mask[j] is True and mask[j+1] is False.
                
                # Vectorized find:
                # transitions = np.where(exercise_mask[:-1] & ~exercise_mask[1:])[0]
                # If found, j = transitions[0]
                
                trans_idx = np.where(exercise_mask[:-1] & ~exercise_mask[1:])[0]
                if len(trans_idx) > 0:
                    j = trans_idx[0]
                    S1 = S_i_minus_1[j]
                    S2 = S_i_minus_1[j+1]
                    # Linear interpolation is a good approximation
                    # We can also just take the midpoint or the lower bound.
                    # Interpolation assumes linearity of the difference, which is approximate.
                    # Given the high N, the step size is small, so S1 is a good estimate.
                    # Let's use the geometric mean or just S1.
                    # S1 is the highest price where we exercise.
                    # The true critical price is slightly higher.
                    # Let's interpolate.
                    # Value at S1: payoff(S1) > cont(S1)
                    # Value at S2: payoff(S2) < cont(S2)
                    # We don't have the exact values of the *difference* function at the boundary,
                    # but we have the values.
                    # diff1 = payoff(S1) - V_prev[j] (positive)
                    # diff2 = payoff(S2) - V_prev[j+1] (negative)
                    # S_star = S1 - diff1 * (S2 - S1) / (diff2 - diff1)
                    
                    diff1 = payoff[j] - V_prev[j]
                    diff2 = payoff[j+1] - V_prev[j+1]
                    if diff2 != diff1: # avoid div by 0
                        S_star = S1 - diff1 * (S2 - S1) / (diff2 - diff1)
                    else:
                        S_star = S1
                else:
                    # Fallback
                    S_star = S_i_minus_1[0]
            
            critical_prices[k-1] = S_star
        
        V = V_prev
        
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, critical_prices):
            self.critical_prices = critical_prices
            self.n_exercise = n_exercise
            self.dt = T / n_exercise
            self.T = T
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time
            Returns: boolean array
            """
            # Determine the exercise step index (1-based)
            # time is in [0, T]
            # Exercise dates are at t = 1*dt, 2*dt, ..., 12*dt
            # step_idx = round(time / dt)
            step_idx = int(round(time / self.dt))
            
            # Clamp step_idx to 1..12
            step_idx = max(1, min(step_idx, self.n_exercise))
            
            # Get critical price for this step
            # critical_prices array is 0-indexed for steps 1..12
            # So step 1 uses index 0, step 12 uses index 11
            S_star = self.critical_prices[step_idx - 1]
            
            # Exercise if S < S_star
            return current_state < S_star

    return ExercisePolicy(critical_prices)

# EVOLVE-BLOCK-END
