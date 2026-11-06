import numpy as np
import time
from scipy.optimize import minimize

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = np.asarray(critical_prices)
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        # current_state: array-like
        # time: float
        # Determine exercise index
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        # Validate index
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return np.asarray(current_state) <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    # 1. High-Resolution Binomial Tree for Initial Critical Prices
    # Use 200 steps per month -> 2400 total steps.
    # This provides a very accurate initial estimate of the boundary.
    steps_per_month = 200
    N = n_ex * steps_per_month 
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    S0_ref = 100.0 
    
    # Initialize value array at maturity
    # We work with log-prices for stability, but convert for payoff.
    # Step N has N+1 nodes.
    # Node i at step j: S = S0 * u^i * d^(j-i)
    # log(S) = log(S0) + i*log(u) + (j-i)*log(d)
    
    V = np.zeros(N + 1)
    
    # Maturity (Step N)
    i_vals = np.arange(N + 1)
    log_S_N = np.log(S0_ref) + N * np.log(d) + i_vals * (np.log(u) - np.log(d))
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    # Current log-S for step N
    log_S_curr = log_S_N
    
    critical_prices = np.zeros(12)
    
    # Backward Induction
    # We need to find critical prices at steps: 100, 200, ..., 2400 (which is N)
    # Wait, exercise dates are at t=1/12, 2/12, ..., 12/12.
    # Step j corresponds to time t = j * dt = j * T / N.
    # Exercise date k (1..12) is at t_k = k/12.
    # Step index j_k such that j_k * dt = k/12 => j_k = k/12 * N/T = k * (N/12) = k * steps_per_month.
    # So exercise dates are at steps: steps_per_month, 2*steps_per_month, ..., 12*steps_per_month.
    
    for j in range(N, 0, -1):
        # Calculate values for step j-1
        # V_cont[i] = df * (p * V[i] + (1-p) * V[i+1]) for i=0..j-1
        V_cont = discount_factor * (p * V[0:j] + (1 - p) * V[1:j+1])
        
        # Update log-S for step j-1
        # log_S_prev[i] = log_S_curr[i] - log(d)  (since S_prev = S_curr / d? No.)
        # S_curr[i] = S0 * u^i * d^(j-i)
        # S_prev[i] = S0 * u^i * d^(j-1-i) = S_curr[i] * d / u? No.
        # S_prev[i] = S_curr[i] * d / (u * d^(j-i) * u^-i ... )
        # Actually: S_prev[i] = S0 * u^i * d^(j-1-i)
        # S_curr[i] = S0 * u^i * d^(j-i)
        # So S_prev[i] = S_curr[i] / d
        # log_S_prev[i] = log_S_curr[i] - log(d)
        
        log_S_prev = log_S_curr[0:j] - np.log(d)
        S_prev = np.exp(log_S_prev)
        
        V_ex = np.maximum(K - S_prev, 0.0)
        V_new = np.maximum(V_cont, V_ex)
        
        step_num = j - 1
        # Check if we are at an exercise date
        if step_num % steps_per_month == 0 and step_num > 0:
            k_idx = step_num // steps_per_month  # 1..12
            
            # Find critical price: largest S where V_ex >= V_cont
            exercise_mask = V_ex >= V_cont
            if np.any(exercise_mask):
                # Since S increases with i, find the last True
                last_true_idx = np.where(exercise_mask)[0][-1]
                critical_prices[k_idx - 1] = S_prev[last_true_idx]
            else:
                critical_prices[k_idx - 1] = 0.0
            
        V = V_new
        log_S_curr = log_S_prev
        
        # Time check
        if time.time() > deadline - 30.0:
            break

    # Ensure critical prices are monotonic? 
    # For a put, as time decreases (k decreases), the critical price should generally increase?
    # Actually, for a put, you exercise when S is low. 
    # At maturity (k=12), crit = K.
    # As t -> 0, the option value approaches intrinsic value if deep ITM, but time value is higher.
    # The boundary S_crit(t) for a put is decreasing as t increases? 
    # Let's check: At t=T, S_crit=K. At t=0, S_crit is lower? 
    # Actually, for a European put, no exercise. For Bermudan, you might exercise earlier if deep ITM.
    # The critical price S_c(t) satisfies V(S_c, t) = K - S_c.
    # Typically S_c(t) < K for t < T. And S_c(t) increases as t approaches T? 
    # No, S_c(T)=K. As t goes back to 0, S_c(t) decreases.
    # So critical_prices[11] (t=1) >= critical_prices[10] (t=11/12) ... >= critical_prices[0] (t=1/12).
    
    # 2. Optimization using Training Paths
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12)
        
    M = paths.shape[0]
    
    # Pre-compute discount factors
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    def negative_value(c_prices):
        # Vectorized first-exercise detection
        # c_prices shape (12,)
        # paths shape (M, 12)
        
        # Condition: S[:, k] <= c_prices[k]
        W = paths <= c_prices[np.newaxis, :]
        
        # We want the first k where W is True.
        # If no k, payoff is 0.
        
        # Trick: Flip W? No.
        # Use cumulative product of ~W?
        # If ~W is True for all previous steps, and W is True at current, it's the first.
        
        # Efficient way:
        # 1. Compute "not exercised before" mask.
        #    not_exercised_before[:, 0] = True
        #    not_exercised_before[:, k] = ~W[:, 0] & ... & ~W[:, k-1]
        
        # Vectorized cumulative AND of ~W along axis 1
        # cum_not_ex = np.cumprod(~W, axis=1)  # This is 1 if all previous are False
        # But we need ~W[:, k-1] ... 
        # Let's use a different trick.
        # First exercise index is the argmax of W if any True, else -1.
        # But we need the *first* True.
        
        # Since M can be 4096 and 12 steps, a small loop is often faster than complex vectorization in Python/NumPy due to overhead,
        # but let's try to vectorize properly.
        
        # Method:
        # Create a matrix of indices 0..11.
        # Mask out False entries with a large number.
        # Then argmin gives the first True index.
        
        W_float = W.astype(float)
        # Set False to 100, True to 0?
        # If all False, min is 100.
        
        # Actually, let's use the cumulative product trick correctly.
        # not_exercised_yet[:, k] = True if no exercise in 0..k-1.
        # not_exercised_yet[:, 0] = True
        # not_exercised_yet[:, k] = not_exercised_yet[:, k-1] & ~W[:, k-1]
        
        # This is inherently sequential. However, we can do:
        # cum_not = np.cumprod(~W, axis=1)
        # This gives 1 if ~W[0..k] are all 1.
        # We want the first k where W[k] is 1 AND cum_not[k-1] is 1.
        # Let's shift cum_not.
        
        # Simpler:
        # If W has no True, payoff 0.
        # If W has True, we want the index of the first True.
        
        # Using argmax on W gives the *last* True if we don't invert? No, argmax on bool gives first True.
        # Wait, np.argmax([0,1,1]) returns 1 (first True).
        # But we need to handle the case where no True exists.
        
        has_exercise = np.any(W, axis=1)
        
        # Get first True index
        # Replace False with -1? No.
        # Use:
        # idx = np.argmax(W, axis=1)
        # But if no True, argmax returns 0. We need to mask those.
        
        first_idx = np.argmax(W, axis=1)
        
        # For paths with no exercise, set payoff to 0.
        # For paths with exercise, payoff = discount[first_idx] * (K - S[first_idx])
        
        payoffs = np.zeros(M)
        
        if np.any(has_exercise):
            # Advanced indexing
            # We need to index paths[range(M), first_idx] only for those has_exercise
            # And discount_factors[first_idx]
            
            ex_idx = np.where(has_exercise)[0]
            ex_steps = first_idx[ex_idx]
            
            S_ex = paths[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            
            payoffs[ex_idx] = disc_ex * (K - S_ex)
            
        return -np.mean(payoffs)
    
    # Optimization
    try:
        # Bounds: Critical price should be between 0 and K
        bounds = [(0, K) for _ in range(12)]
        x0 = critical_prices
        x0 = np.clip(x0, 0, K)
        
        # Check time
        if time.time() < deadline - 10.0:
            # Use Nelder-Mead. It's robust for small dimensions.
            # maxiter 500 is safe.
            res = minimize(negative_value, x0, method='Nelder-Mead', bounds=bounds, 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            
            # Compare with initial
            val_init = negative_value(x0)
            val_opt = res.fun
            
            if val_opt < val_init: # Remember negative value, so lower is better
                critical_prices = res.x
            else:
                critical_prices = x0
        else:
            critical_prices = x0
            
    except Exception:
        # Fallback to tree
        pass

    return ExercisePolicy(critical_prices)
