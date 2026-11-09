import numpy as np
from scipy.optimize import brentq
import time
import threading

# Cache for precomputed tables
_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories.
    Returns:
        opt_bid: np.array of shape (steps, 2*Q+1)
        opt_ask: np.array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    b_upper = 10.0 / k
    
    # Precompute constants
    p_0 = 1.0 - np.exp(-A * dt)
    p_0_prime = -k * A * dt * (1.0 - p_0)
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Return what we have so far, fill remaining with a reasonable default
            depth_default = 1.0 / k
            for n_fill in range(n, N):
                opt_bid[n_fill, :] = depth_default
                opt_ask[n_fill, :] = depth_default
            return opt_bid, opt_ask
        
        V_curr = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_idx - Q
            
            V_q = V_next[q_idx]
            
            if q < max_q:
                delta_plus = V_next[q_idx + 1] - V_q
            else:
                delta_plus = 0.0 

            if q > min_q:
                delta_minus = V_next[q_idx - 1] - V_q
            else:
                delta_minus = 0.0

            b = 0.0
            a = 0.0
            
            for _iter in range(40):
                if q < max_q:
                    p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    C_b = delta_plus * (1.0 - p_a_curr)
                    
                    val_0 = p_0_prime * C_b + p_0
                    
                    if val_0 <= 0:
                        b_new = 0.0
                    else:
                        def f_b(x):
                            if x == 0:
                                p_b = p_0
                                dp_db = p_0_prime
                            else:
                                u = np.exp(-k * x)
                                p_b = 1.0 - np.exp(-A * u * dt)
                                dp_db = -k * A * dt * u * (1.0 - p_b)
                            return dp_db * (x + C_b) + p_b
                        
                        try:
                            b_new = brentq(f_b, 0, b_upper, xtol=1e-10)
                        except ValueError:
                            if f_b(b_upper) > 0:
                                b_new = b_upper
                            else:
                                b_new = 0.0
                else:
                    b_new = 0.0

                if q > min_q:
                    p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
                    C_a = delta_minus * (1.0 - p_b_curr)
                    
                    val_0 = p_0_prime * C_a + p_0
                    
                    if val_0 <= 0:
                        a_new = 0.0
                    else:
                        def f_a(x):
                            if x == 0:
                                p_a = p_0
                                dp_da = p_0_prime
                            else:
                                u = np.exp(-k * x)
                                p_a = 1.0 - np.exp(-A * u * dt)
                                dp_da = -k * A * dt * u * (1.0 - p_a)
                            return dp_da * (x + C_a) + p_a
                        
                        try:
                            a_new = brentq(f_a, 0, b_upper, xtol=1e-10)
                        except ValueError:
                            if f_a(b_upper) > 0:
                                a_new = b_upper
                            else:
                                a_new = 0.0
                else:
                    a_new = 0.0

                if abs(b_new - b) < 1e-9 and abs(a_new - a) < 1e-9:
                    b = b_new
                    a = a_new
                    break
                b = b_new
                a = a_new
            
            opt_bid[n, q_idx] = b
            opt_ask[n, q_idx] = a
            
            p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
            
            if q == max_q:
                prob_q_plus = 0.0
                prob_q_minus = p_a
                prob_q = 1.0 - p_a
                reward = p_a * a
            elif q == min_q:
                prob_q_plus = p_b
                prob_q_minus = 0.0
                prob_q = 1.0 - p_b
                reward = p_b * b
            else:
                prob_q_plus = p_b * (1.0 - p_a)
                prob_q_minus = (1.0 - p_b) * p_a
                prob_q = 1.0 - prob_q_plus - prob_q_minus
                reward = p_b * b + p_a * a
            
            exp_val = prob_q_plus * (V_next[q_idx+1] if q < max_q else 0.0) + \
                      prob_q_minus * (V_next[q_idx-1] if q > min_q else 0.0) + \
                      prob_q * V_q
            
            risk = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            V_curr[q_idx] = reward - risk + exp_val

        V_next = V_curr

    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            # Precompute
            start_time = time.time()
            opt_bid, opt_ask = _precompute_regime(
                params['sigma'], params['A'], params['k'], params['gamma'], 
                params['T'], params['steps'], params['Q']
            )
            _TABLE_CACHE[key] = (opt_bid, opt_ask)
            
            # Time guard (safety)
            if time.time() - start_time > 290:
                 depth = 1.0 / params["k"]
                 return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    # Ensure valid outputs: no NaN, no -inf, no values below -3/k
    min_allowed = -3.0 / params['k']
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths < min_allowed), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths < min_allowed), np.inf, ask_depths)
    
    return bid_depths, ask_depths
