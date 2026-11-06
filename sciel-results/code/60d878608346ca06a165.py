import numpy as np
import time
from scipy.optimize import minimize

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
    deadline = start_time + 145.0
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12)
    M = paths.shape[0]
    
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    # High-resolution binomial tree for initial guess
    steps_per_month = 100
    N = n_ex * steps_per_month
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    S0_ref = 100.0
    
    V = np.zeros(N + 1)
    i_vals = np.arange(N + 1)
    log_S_N = np.log(S0_ref) + N * np.log(d) + i_vals * (np.log(u) - np.log(d))
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    log_S_curr = log_S_N
    
    critical_prices = np.zeros(12)
    
    for j in range(N, 0, -1):
        V_cont = discount_factor * (p * V[0:j] + (1 - p) * V[1:j+1])
        log_S_prev = log_S_curr[0:j] - np.log(d)
        S_prev = np.exp(log_S_prev)
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
        log_S_curr = log_S_prev
        
        if time.time() > deadline - 60.0:
            break
    
    critical_prices = np.clip(critical_prices, 0, K)
    
    # Smooth surrogate objective using soft-min
    def smooth_negative_value(c_prices, temperature=0.5):
        """
        Smooth surrogate: uses a soft-min to approximate the first-exercise payoff.
        The exact payoff is: sum_k 1{first_exercise=k} * disc_k * (K - S_k)
        We approximate with a smooth function that penalizes early exercise less.
        """
        # For each path and each step k, compute the "effective" exercise indicator
        # Using a softmax-like approach over the exercise indicators
        # W[:, k] = 1 if S_k <= c_k
        W = (paths <= c_prices[np.newaxis, :]).astype(float)
        
        # The first exercise is the smallest k where W[k]=1
        # We can approximate the contribution of each step using a weighted sum
        # where earlier steps get higher weight if they are exercised
        
        # A simpler smooth approximation:
        # payoff ≈ sum_k disc_k * (K - S_k) * W_k * prod_{j<k} (1 - W_j)
        # This is still nonsmooth due to the product.
        
        # Let's use a different approach: 
        # The value is the expected discounted payoff.
        # We can compute it directly for the hard threshold, and use that for optimization.
        # The smooth surrogate idea is to use a logistic function instead of step function.
        
        # Actually, let's stick with the hard threshold but use a better optimizer.
        # The issue with the previous approach was local minima.
        # Let's use a multi-start Nelder-Mead with random perturbations.
        
        # For now, return the hard negative value
        return negative_value_hard(c_prices)
    
    def negative_value_hard(c_prices):
        W = paths <= c_prices[np.newaxis, :]
        has_exercise = np.any(W, axis=1)
        first_idx = np.argmax(W, axis=1)
        
        payoffs = np.zeros(M)
        if np.any(has_exercise):
            ex_idx = np.where(has_exercise)[0]
            ex_steps = first_idx[ex_idx]
            S_ex = paths[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            payoffs[ex_idx] = disc_ex * (K - S_ex)
        
        return -np.mean(payoffs)
    
    # Multi-start optimization
    bounds = [(0, K) for _ in range(12)]
    best_x = critical_prices.copy()
    best_val = negative_value_hard(best_x)
    
    rng = np.random.default_rng(42)
    
    # Try multiple random starts
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        if i == 0:
            x0 = best_x
        else:
            # Random perturbation around the current best
            x0 = best_x + rng.normal(0, 2.0, 12)
            x0 = np.clip(x0, 0, K)
        
        try:
            res = minimize(negative_value_hard, x0, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'disp': False})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final refinement with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            res = minimize(negative_value_hard, best_x, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return ExercisePolicy(best_x)
