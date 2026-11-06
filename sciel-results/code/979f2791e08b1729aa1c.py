import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = np.asarray(critical_prices)
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return current_state <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    # 1. High-Resolution Binomial Tree for Initial Critical Prices
    steps_per_month = 100
    N = n_ex * steps_per_month 
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    S0_ref = 100.0 
    
    # Initialize value array at maturity
    V = np.zeros(N + 1)
    i_vals = np.arange(N + 1)
    
    # Use log-space for stability and speed
    log_d = np.log(d)
    log_u = np.log(u)
    log_S_N = np.log(S0_ref) + N * log_d + i_vals * (log_u - log_d)
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(12)
    S_curr = S_N.copy()
    
    for j in range(N, 0, -1):
        V_cont = discount_factor * (p * V[0:j] + (1 - p) * V[1:j+1])
        S_prev = S_curr[0:j] / d
        V_ex = np.maximum(K - S_prev, 0.0)
        V_new = np.maximum(V_cont, V_ex)
        
        step_num = j - 1
        if step_num % steps_per_month == 0 and step_num > 0:
            k_idx = step_num // steps_per_month
            exercise_mask = V_ex >= V_cont
            if np.any(exercise_mask):
                last_true_idx = np.where(exercise_mask)[0][-1]
                critical_prices[k_idx - 1] = S_prev[last_true_idx]
            else:
                critical_prices[k_idx - 1] = 0.0
            
        V = V_new
        S_curr = S_prev
        
        if time.time() > deadline - 90.0:
            break

    # 2. Longstaff-Schwartz Regression-Based Refinement
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12)
        
    M = paths.shape[0]
    
    # Discount factors for each step
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    # Initialize critical prices from binomial tree
    c_prices = critical_prices.copy()
    c_prices = np.clip(c_prices, 0, K)
    c_prices[11] = K  # At maturity, exercise if ITM
    
    # Longstaff-Schwartz style: iterate backwards from last exercise date
    # For each exercise date k, fit a regression to estimate continuation value
    # and update the critical price
    
    # We'll do multiple passes to allow for refinement
    n_passes = 3
    if time.time() > deadline - 60.0:
        n_passes = 1
    
    for pass_num in range(n_passes):
        if time.time() > deadline - 30.0:
            break
            
        # Work backwards from exercise date 12 to 1
        # V_next[k] = continuation value at step k given optimal exercise from k+1 onwards
        # We'll compute this iteratively
        
        # Initialize: at maturity (step 12), continuation value is 0 (option expires)
        # For step 11, continuation = discounted expected value of max(exercise at 12, continue)
        
        # We'll store the fitted continuation value function for each step
        # and use it to determine exercise decisions
        
        # For simplicity and speed, we'll use the following approach:
        # 1. Start with current critical prices
        # 2. For each step k from 11 down to 1:
        #    a. Determine which paths are still "alive" (not exercised) at step k
        #    b. For alive paths, compute the continuation value using the fitted model
        #       from step k+1 (or the immediate payoff if k=11)
        #    c. Fit a regression to estimate continuation value as a function of S_k
        #    d. Update critical price for step k based on where exercise payoff >= continuation
        
        # Let's implement a simpler but effective version:
        # Use the training paths to directly optimize each critical price
        # with a Monte Carlo estimate of the value
        
        def compute_value(c_prices):
            """Compute the mean discounted payoff given critical prices"""
            W = paths <= c_prices[np.newaxis, :]
            
            exercised = np.zeros(M, dtype=bool)
            exercise_step = np.full(M, -1, dtype=int)
            
            for k in range(12):
                not_exercised = ~exercised
                cond = W[:, k] & not_exercised
                exercised[cond] = True
                exercise_step[cond] = k
                
                if np.all(exercised):
                    break
                    
            payoffs = np.zeros(M)
            if np.any(exercised):
                idx = np.arange(M)
                ex_idx = idx[exercised]
                ex_steps = exercise_step[exercised]
                
                S_ex = paths[ex_idx, ex_steps]
                disc_ex = discount_factors[ex_steps]
                
                payoffs[ex_idx] = disc_ex * (K - S_ex)
                
            return np.mean(payoffs)
        
        # Optimize each critical price independently using the current estimate
        # for the other prices
        for k in range(11):  # 0 to 10
            if time.time() > deadline - 25.0:
                break
                
            # Bounds based on neighboring critical prices for monotonicity
            lower = c_prices[k-1] if k > 0 else 0
            upper = c_prices[k+1] if k < 11 else K
            
            if lower > upper:
                lower, upper = upper, lower
                
            # Add small margin to allow for exploration
            margin = 0.5
            lower = max(0, lower - margin)
            upper = min(K, upper + margin)
            
            if lower >= upper:
                continue
                
            def obj_k(c_k_val):
                c_temp = c_prices.copy()
                c_temp[k] = c_k_val
                return -compute_value(c_temp)
                
            try:
                res = minimize_scalar(obj_k, bounds=(lower, upper), method='bounded', 
                                      options={'xatol': 1e-4, 'maxiter': 50})
                if res.success:
                    c_prices[k] = res.x
            except Exception:
                pass
        
        # Enforce monotonicity: c_1 <= c_2 <= ... <= c_12
        for k in range(1, 12):
            if c_prices[k] < c_prices[k-1]:
                c_prices[k] = c_prices[k-1]
                
        # Ensure c_12 = K
        c_prices[11] = K
    
    # Final high-precision polish pass
    if time.time() < deadline - 15.0:
        for k in range(11):
            if time.time() > deadline - 10.0:
                break
                
            lower = c_prices[k-1] if k > 0 else 0
            upper = c_prices[k+1] if k < 11 else K
            
            if lower > upper:
                lower, upper = upper, lower
                
            # Tighter bounds for polish
            polish_margin = 0.1
            lower = max(0, lower - polish_margin)
            upper = min(K, upper + polish_margin)
            
            if lower >= upper:
                continue
                
            def obj_k_polish(c_k_val):
                c_temp = c_prices.copy()
                c_temp[k] = c_k_val
                return -compute_value(c_temp)
                
            try:
                res = minimize_scalar(obj_k_polish, bounds=(lower, upper), method='bounded', 
                                      options={'xatol': 1e-6, 'maxiter': 100})
                if res.success:
                    c_prices[k] = res.x
            except Exception:
                pass
        
        # Final monotonicity enforcement
        for k in range(1, 12):
            if c_prices[k] < c_prices[k-1]:
                c_prices[k] = c_prices[k-1]
                
        c_prices[11] = K
    
    return ExercisePolicy(c_prices)
