import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option.
    Uses a 1200-step binomial tree with boundary interpolation for high precision.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Use 1200 steps for high precision
    N_steps = 1200
    dt = T / N_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    
    # Reference S_0 for tree construction
    S_0_ref = 100.0
    log_S_0 = math.log(S_0_ref)
    log_u = math.log(u)
    
    # Initialize at maturity (step N_steps)
    i_terminal = np.arange(N_steps + 1)
    exponents = 2 * i_terminal - N_steps
    log_S_terminal = log_S_0 + exponents * log_u
    S_terminal = np.exp(log_S_terminal)
    V_next = np.maximum(K - S_terminal, 0.0)
    
    # Store critical prices for each exercise date k=1..12
    # s_crit[k] is the maximum S at which we exercise at time t_k
    s_crit = [0.0] * (n_exercises + 1)
    
    # Iterate backwards from j = N_steps - 1 down to 0
    for j in range(N_steps - 1, -1, -1):
        # Check deadline
        if time.time() > deadline:
            # Fallback: fill remaining s_crit with K (exercise if ITM)
            for k in range(1, n_exercises + 1):
                if s_crit[k] == 0.0:
                    s_crit[k] = K
            break
            
        # Current step j has nodes i = 0 .. j
        i_curr = np.arange(j + 1)
        exponents_curr = 2 * i_curr - j
        log_S_curr = log_S_0 + exponents_curr * log_u
        S_curr = np.exp(log_S_curr)
        
        # Intrinsic Value at current step
        V_int = np.maximum(K - S_curr, 0.0)
        
        # Continuation Value
        # V_cont[i] = e^(-r dt) * (p * V_next[i+1] + (1-p) * V_next[i])
        V_cont = disc * (p * V_next[1:] + (1 - p) * V_next[:-1])
        
        # Determine if this step is an exercise date
        if j > 0 and j % 100 == 0:
            k = j // 100
            # Find the boundary where V_int >= V_cont
            diff = V_int - V_cont
            exercise_mask = diff >= 0
            
            if np.any(exercise_mask):
                # Find the last index where exercise is optimal
                last_true_idx = np.where(exercise_mask)[0][-1]
                
                # Interpolate the boundary for higher precision
                # We want the S where diff(S) = 0
                # We have diff[last_true_idx] >= 0 and diff[last_true_idx+1] < 0 (if it exists)
                if last_true_idx < j:
                    # Interpolate between last_true_idx and last_true_idx + 1
                    d1 = diff[last_true_idx]
                    d2 = diff[last_true_idx + 1]
                    s1 = S_curr[last_true_idx]
                    s2 = S_curr[last_true_idx + 1]
                    
                    if abs(d2 - d1) > 1e-12:
                        # Linear interpolation
                        alpha = d1 / (d1 - d2)
                        s_crit[k] = s1 + alpha * (s2 - s1)
                    else:
                        s_crit[k] = s1
                else:
                    # Boundary is at the highest node
                    s_crit[k] = S_curr[last_true_idx]
            else:
                # Never exercise (boundary below lowest node)
                s_crit[k] = 0.0
                
        # Update V for this step
        V_curr = np.where(V_int > V_cont, V_int, V_cont)
        V_next = V_curr

    # Fallback for any uncomputed boundaries
    for k in range(1, n_exercises + 1):
        if s_crit[k] == 0.0:
             s_crit[k] = K

    class ExercisePolicy:
        def __init__(self, s_crits, n_exercises, T):
            self.s_crits = s_crits
            self.n_exercises = n_exercises
            self.T = T
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices
            time: float, current time
            returns: boolean array
            """
            if time <= 0:
                return np.zeros_like(current_state, dtype=bool)
            
            # Map time to exercise index k
            # time is in [0, T]
            # k = time / (T/n_exercises)
            dt_exercise = self.T / self.n_exercises
            k_float = time / dt_exercise
            k = int(round(k_float))
            
            if k < 1:
                return np.zeros_like(current_state, dtype=bool)
            if k > self.n_exercises:
                k = self.n_exercises
            
            crit = self.s_crits[k]
            
            if crit == 0.0:
                return np.zeros_like(current_state, dtype=bool)
                
            return current_state <= crit
        
    return ExercisePolicy(s_crit, n_exercises, T)
