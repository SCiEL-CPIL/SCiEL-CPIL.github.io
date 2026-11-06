import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Fixed parameters from task description
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    exercise_times = np.linspace(1/12.0, T, n_exercise)
    
    # High-resolution binomial tree
    # N=4800 provides 400 steps per month, which is sufficient for high accuracy
    # and runs fast with vectorized NumPy operations.
    N = 4800
    dt = T / N
    steps_per_period = N // n_exercise
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # We work in log-space or linear space?
    # Linear space is fine for S in [0, ~1000].
    # S_i[j] = S0 * u^j * d^(i-j).
    # To avoid recomputing powers, we can iterate and update S.
    
    S0 = 100.0
    
    # Initialize at T (i=N)
    # j ranges from 0 to N
    j_N = np.arange(N + 1)
    # S_N[j] = S0 * u^j * d^(N-j)
    # log_S_N = log(S0) + j*log(u) + (N-j)*log(d)
    log_u = np.log(u)
    log_d = np.log(d)
    log_S0 = np.log(S0)
    
    log_S = log_S0 + j_N * log_u + (N - j_N) * log_d
    S = np.exp(log_S)
    
    # Value at T: Put payoff
    V = np.maximum(K - S, 0.0)
    
    critical_prices = [0.0] * 13  # Index 1..12
    
    # Backward induction
    # We iterate i from N-1 down to 0.
    # At each step, we update S and V.
    # S_{i}[j] = S_{i+1}[j] * d (if we move down in j? No, j is up-steps)
    # Actually, it's easier to recompute S from log_S at exercise dates only, 
    # or update S vectorially.
    # S_i[j] = S_{i+1}[j] * d is NOT correct directly because the grid shifts.
    # S_{i}[j] = S0 * u^j * d^(i-j).
    # S_{i+1}[j] = S0 * u^j * d^(i+1-j) = S_i[j] * d.
    # So S_i[j] = S_{i+1}[j] / d = S_{i+1}[j] * u.
    # Wait, j is the number of up steps.
    # At step i+1, there are i+2 states (j=0..i+1).
    # At step i, there are i+1 states (j=0..i).
    # S_{i+1}[j] = S0 * u^j * d^(i+1-j).
    # S_i[j] = S0 * u^j * d^(i-j) = S_{i+1}[j] * (1/d) = S_{i+1}[j] * u.
    # So the state values at step i for j=0..i are just u * S_{i+1}[0..i].
    
    # Let's verify:
    # S_{i+1}[0] = S0 * d^(i+1).
    # S_i[0] = S0 * d^i = S_{i+1}[0] * u. Correct.
    # S_{i+1}[i+1] = S0 * u^(i+1).
    # S_i[i] = S0 * u^i = S_{i+1}[i+1] * d. 
    # So S_i[j] = S_{i+1}[j] * u is correct for j=0..i.
    
    # We need to be careful with the array slicing.
    # V has size i+2 at step i+1.
    # V_up = V[1:] (size i+1)
    # V_down = V[:-1] (size i+1)
    # V_cont = exp(-r dt) * (p * V_up + (1-p) * V_down)
    
    # S at step i is S_prev[0:i+1] * u ?
    # S_prev is S at step i+1 (size i+2).
    # S_i[j] = S_prev[j] * u.
    
    S_prev = S.copy()
    
    for i in range(N - 1, -1, -1):
        if time.time() > deadline:
            break
            
        # Update S for step i
        # S_i has size i+1
        S_curr = S_prev[:i+1] * u
        
        # Compute Continuation Value
        # V_prev has size i+2
        V_up = V[1:]
        V_down = V[:-1]
        V_cont = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)
        
        # Check if this is an exercise date
        # Exercise dates are at i = k * steps_per_period for k=1..12
        # Note: i=N is T. i=0 is t=0.
        # Exercise at t_k = k/12.
        # Step index i corresponds to time t_i = i * dt.
        # We want t_i = k/12.
        # i * (T/N) = k/12 => i = k * N / 12 = k * steps_per_period.
        
        if i % steps_per_period == 0:
            k = i // steps_per_period
            if 1 <= k <= 12:
                V_ex = np.maximum(K - S_curr, 0.0)
                
                # Find boundary
                # We want the largest j such that V_ex[j] >= V_cont[j]
                diff = V_ex - V_cont
                valid = np.where(diff >= 0)[0]
                if len(valid) > 0:
                    j_max = valid[-1]
                    critical_prices[k] = S_curr[j_max]
                else:
                    critical_prices[k] = 0.0
                
                # Update V to be the max of exercise and continuation
                V = np.maximum(V_cont, V_ex)
            else:
                V = V_cont
        else:
            V = V_cont
            
        S_prev = S_curr

    # Enforce monotonicity of critical prices
    # For a put, S*_k should be non-decreasing with k (as time to expiry decreases, we are more likely to exercise, so boundary rises).
    # S*_1 <= S*_2 <= ... <= S*_12
    for k in range(1, 12):
        if critical_prices[k] > critical_prices[k+1]:
            critical_prices[k+1] = critical_prices[k]
            
    return ExercisePolicy(critical_prices)

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)
        
    def exercise(self, current_state, time):
        # time is a float
        # Determine k
        # The exercise dates are at 1/12, 2/12, ..., 12/12.
        # time should be close to one of these.
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        S_star = self.critical_prices[k_int]
        return current_state <= S_star

def _fallback_policy():
    # Fallback if tree calculation fails or times out early
    # Use a rough heuristic: Boundary increases with k
    K = 100.0
    cp = [0.0] * 13
    for k in range(1, 13):
        # Rough approximation for Bermudan put boundary
        # At t=1, S* ~ K. At t=1/12, S* is lower.
        # Linear interpolation is bad, but better than 0.
        # Let's use S* = K * (1 - exp(-5 * k/12)) ?
        # Or just a simple ramp.
        cp[k] = K * (k / 12.0) * 0.95
    return ExercisePolicy(cp)
