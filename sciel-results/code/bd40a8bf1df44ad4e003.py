# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Global cache for pre-computed policies
_policy_cache = {}

def _compute_policy(params):
    """
    Compute the optimal policy using backward induction with continuous depth optimization
    using gradient-based methods.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    num_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    V_next = np.zeros(num_q)
    opt_bid = np.zeros((steps, num_q))
    opt_ask = np.zeros((steps, num_q))
    
    max_d = 10.0
    
    for t in range(steps - 1, -1, -1):
        V_current = np.zeros(num_q)
        
        for q_idx in range(num_q):
            q = q_indices[q_idx]
            
            V_q = V_next[q_idx]
            V_q_plus = V_next[q_idx + 1] if q_idx < num_q - 1 else 0.0
            V_q_minus = V_next[q_idx - 1] if q_idx > 0 else 0.0
            
            def objective_and_grad(x):
                d_b, d_a = x
                d_b = np.clip(d_b, 0, max_d)
                d_a = np.clip(d_a, 0, max_d)
                
                if q == Q:
                    p_b = 0.0
                    dp_b_dd_b = 0.0
                else:
                    exp_term_b = np.exp(-k * d_b)
                    rate_b = A * exp_term_b * dt
                    p_b = 1.0 - np.exp(-rate_b)
                    # dp_b/dd_b = -exp(-rate_b) * d(rate_b)/dd_b
                    # d(rate_b)/dd_b = A * exp(-k d_b) * (-k) * dt = -k * rate_b
                    # dp_b/dd_b = -exp(-rate_b) * (-k * rate_b) = k * rate_b * exp(-rate_b)
                    dp_b_dd_b = k * rate_b * np.exp(-rate_b)
                
                if q == -Q:
                    p_a = 0.0
                    dp_a_dd_a = 0.0
                else:
                    exp_term_a = np.exp(-k * d_a)
                    rate_a = A * exp_term_a * dt
                    p_a = 1.0 - np.exp(-rate_a)
                    dp_a_dd_a = k * rate_a * np.exp(-rate_a)
                
                E_R = p_b * d_b + p_a * d_a
                EV = ((1 - p_b) * (1 - p_a) * V_q + 
                      p_b * (1 - p_a) * V_q_plus + 
                      (1 - p_b) * p_a * V_q_minus + 
                      p_b * p_a * V_q)
                
                J = E_R + EV - 0.5 * gamma * sigma**2 * q**2 * dt
                
                # Gradient of J w.r.t d_b
                # d(E_R)/dd_b = dp_b/dd_b * d_b + p_b
                # d(EV)/dd_b = dp_b_dd_b * [ (1-p_a)V_q_plus - (1-p_a)V_q + p_a V_q - p_a V_q_minus ]
                #           = dp_b_dd_b * (1-p_a)(V_q_plus - V_q) + dp_b_dd_b * p_a (V_q - V_q_minus)
                #           = dp_b_dd_b * [ (1-p_a)(V_q_plus - V_q) + p_a(V_q - V_q_minus) ]
                
                grad_b = dp_b_dd_b * d_b + p_b + dp_b_dd_b * ((1 - p_a) * (V_q_plus - V_q) + p_a * (V_q - V_q_minus))
                
                grad_a = dp_a_dd_a * d_a + p_a + dp_a_dd_a * ((1 - p_b) * (V_q_minus - V_q) + p_b * (V_q - V_q_plus))
                
                # We minimize -J
                return -J, np.array([-grad_b, -grad_a])
            
            x0 = np.array([1.0/k, 1.0/k])
            
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True, bounds=[(0, max_d), (0, max_d)],
                           options={'maxiter': 20, 'gtol': 1e-4})
            
            d_b_opt, d_a_opt = res.x
            opt_bid[t, q_idx] = d_b_opt
            opt_ask[t, q_idx] = d_a_opt
            V_current[q_idx] = -res.fun
            
        V_next = V_current
        
    return {
        'bid': opt_bid,
        'ask': opt_ask
    }

def quote(t, inventory, mid, params):
    cache_key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if cache_key not in _policy_cache:
        _policy_cache[cache_key] = _compute_policy(params)
    
    policy = _policy_cache[cache_key]
    
    steps = params["steps"]
    Q = params["Q"]
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    inv = np.asarray(inventory, dtype=np.int64)
    inv_idx = np.clip(inv + Q, 0, 2 * Q)
    
    bid_depths = policy['bid'][n, inv_idx]
    ask_depths = policy['ask'][n, inv_idx]
    
    bid_depths = np.where(inv >= Q, np.inf, bid_depths)
    ask_depths = np.where(inv <= -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
