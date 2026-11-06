"""Bermudan exercise policy using binomial tree backward induction with critical-price interpolation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 130.0  # safety margin
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # Binomial tree parameters
    N = 1200  # total steps
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Exercise dates in terms of steps
    exercise_steps = np.arange(1, n_exercise + 1) * (N // n_exercise)
    # N // n_exercise = 1200 // 12 = 100
    # So exercise at steps 100, 200, ..., 1200
    
    S0_tree = 100.0
    log_S0 = np.log(S0_tree)
    log_u = np.log(u)
    
    # At maturity (step N):
    k_mat = np.arange(N + 1)
    log_S_mat = log_S0 + (2 * k_mat - N) * log_u
    S_mat = np.exp(log_S_mat)
    values_N = np.maximum(K - S_mat, 0.0)
    
    # Backward induction
    # We'll do backward induction and record critical prices at exercise dates.
    # For a put, there's a critical price below which you exercise.
    
    current_values = values_N.copy()
    
    # Store critical prices for each exercise date
    # critical_price[step] = price below which we exercise
    critical_prices = {}
    
    # Go backward from step N-1 to step 0
    for n in range(N - 1, -1, -1):
        # At step n, there are n+1 nodes.
        # For node k at step n:
        #   up goes to node k+1 at step n+1
        #   down goes to node k at step n+1
        # current_values has length n+2 (for step n+1)
        up_val = current_values[1:n+2]  # length n+1
        down_val = current_values[0:n+1]  # length n+1
        
        cont_val = discount * (p * up_val + (1 - p) * down_val)
        
        # Check if this is an exercise date
        if n in exercise_steps:
            k_nodes = np.arange(n + 1)
            log_S_nodes = log_S0 + (2 * k_nodes - n) * log_u
            S_nodes = np.exp(log_S_nodes)
            ex_val = np.maximum(K - S_nodes, 0.0)
            
            exercise_mask = ex_val > cont_val
            
            # Find the critical price: the highest S where we exercise
            # For a put, exercise when S < critical_price
            exercising_indices = np.where(exercise_mask)[0]
            if len(exercising_indices) > 0:
                # The critical price is at the boundary between exercise and no-exercise
                # Find the largest k where we exercise
                max_exercise_k = exercising_indices[-1]
                # Interpolate between max_exercise_k and max_exercise_k+1
                if max_exercise_k < n:
                    # Linear interpolation in log-space
                    S_ex = S_nodes[max_exercise_k]
                    S_no_ex = S_nodes[max_exercise_k + 1]
                    ex_val_at_k = ex_val[max_exercise_k]
                    cont_val_at_k = cont_val[max_exercise_k]
                    ex_val_at_k1 = ex_val[max_exercise_k + 1]
                    cont_val_at_k1 = cont_val[max_exercise_k + 1]
                    
                    # Find where ex_val = cont_val
                    # ex_val = K - S (for S < K)
                    # We want K - S = cont_val(S)
                    # Approximate: find S where the difference changes sign
                    diff_k = ex_val_at_k - cont_val_at_k
                    diff_k1 = ex_val_at_k1 - cont_val_at_k1
                    
                    if diff_k > 0 and diff_k1 < 0:
                        # Linear interpolation
                        frac = diff_k / (diff_k - diff_k1)
                        critical_price = S_ex + frac * (S_no_ex - S_ex)
                    elif diff_k > 0:
                        critical_price = S_no_ex
                    else:
                        critical_price = S_ex
                else:
                    critical_price = S_nodes[max_exercise_k]
                
                critical_prices[n] = critical_price
            else:
                critical_prices[n] = 0.0
            
            # The value at this node is max(ex_val, cont_val)
            new_values = np.maximum(ex_val, cont_val)
        else:
            new_values = cont_val
        
        current_values = new_values
        
        # Check deadline
        if time.time() > DEADLINE:
            break
    
    # Now we have critical_prices for each exercise step.
    # For the policy, given a state S and time t:
    # 1. Determine which exercise date this is.
    # 2. Find the critical price for that exercise date.
    # 3. Exercise if S < critical_price.
    
    # Map exercise steps to times
    exercise_step_to_time = {n_ex: n_ex * dt for n_ex in exercise_steps}
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self):
            self.critical_prices = critical_prices
            self.exercise_steps_set = set(exercise_steps)
            self.N = N
            self.dt = dt
            self.K = K
            self.r = r
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time (should be one of the exercise dates)
            Returns: boolean array, True if exercise
            """
            # Determine the step
            n = int(round(time / self.dt))
            n = max(0, min(n, self.N))
            
            # Check if this is an exercise date
            if n not in self.critical_prices:
                # Not an exercise date, don't exercise
                return np.zeros(len(current_state), dtype=bool)
            
            critical_price = self.critical_prices[n]
            
            # Exercise if S < critical_price
            return current_state < critical_price
    
    policy = ExercisePolicy()
    return policy

# EVOLVE-BLOCK-END
