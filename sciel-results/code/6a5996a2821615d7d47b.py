"""Optimal market making with inventory risk - Continuous DP solution."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}

def _solve_lambda_1d(C, k, dt, A):
    """
    Solve for optimal lambda maximizing:
    f(lambda) = (1 - exp(-lambda)) * (K + C - (1/k) * ln(lambda))
    where K = (1/k) * ln(A * dt)
    Note: d = K - (1/k) ln(lambda)
    f(lambda) = (1 - exp(-lambda)) * d(lambda) + (1 - exp(-lambda)) * C
    Wait, the C term in the prompt derivation was W.
    Let's stick to the derivative condition:
    e^{-lambda} * (K + W - (1/k) ln(lambda)) - (1/k/lambda) * (1 - e^{-lambda}) = 0
    """
    K = (1.0 / k) * np.log(A * dt)
    # We want to solve for lambda > 0.
    # If K + W is very small, lambda might be small.
    # Use Newton's method on g(lambda) = f'(lambda)
    
    # Initial guess
    # If W is positive, lambda is larger. If W is negative, lambda is smaller.
    # Rough guess: lambda ~ exp(-(K+W) * k) ? No.
    # Let's try lambda = 1.0
    lam = 1.0
    
    for _ in range(10):
        exp_lam = np.exp(-lam)
        # g(lam) = exp_lam * (K + W - (1/k) * ln(lam)) - (1/k) * (1 - exp_lam) / lam
        # f''(lam) for Newton step? Or just use a robust root finder.
        # Let's use scipy.optimize.brentq on the derivative if available, 
        # but manual Newton is faster if we have f''.
        
        # g(lam)
        val = exp_lam * (K + C - (1.0/k) * np.log(lam)) - (1.0/k) * (1.0 - exp_lam) / lam
        
        # g'(lam) = f''(lam)
        # f'(lam) = e^{-l}(C + K - 1/k ln l) - 1/(k l) (1 - e^{-l})
        # f''(lam) = -e^{-l}(C + K - 1/k ln l) + e^{-l}(-1/(k l)) - [ -1/(k l^2)(1 - e^{-l}) + 1/(k l) e^{-l} ]
        #          = -e^{-l}(C + K - 1/k ln l) - e^{-l}/(k l) + 1/(k l^2) - e^{-l}/(k l^2) - e^{-l}/(k l)
        #          = -e^{-l}(C + K - 1/k ln l) - 2 e^{-l}/(k l) + e^{-l}/(k l^2) + 1/(k l^2)
        
        term1 = -exp_lam * (K + C - (1.0/k) * np.log(lam))
        term2 = -2.0 * exp_lam / (k * lam)
        term3 = exp_lam / (k * lam**2)
        term4 = 1.0 / (k * lam**2)
        g_prime = term1 + term2 + term3 + term4
        
        if abs(g_prime) < 1e-12:
            break
            
        step = val / g_prime
        # Dampen if step is too large
        if abs(step) > lam * 0.5:
            step = np.sign(step) * lam * 0.5
            
        lam_new = lam - step
        if lam_new <= 0:
            lam_new = lam * 0.5
        if abs(lam_new - lam) < 1e-8 * lam:
            lam = lam_new
            break
        lam = lam_new
        
    return lam

def _solve_dp_continuous(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    V = np.zeros((n_steps + 1, n_inv))
    bid_depths_table = np.zeros((n_steps, n_inv))
    ask_depths_table = np.zeros((n_steps, n_inv))
    
    K_const = (1.0 / k) * np.log(A * dt)
    
    for t in range(n_steps - 1, -1, -1):
        V_next = V[t + 1]
        cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        for i in range(n_inv):
            q = q_vals[i]
            
            # Get next values
            V_0 = V_next[i]
            
            # Handle boundaries
            if i < n_inv - 1:
                V_plus = V_next[i + 1]
            else:
                V_plus = V_0 # No bid fill, so V_plus effectively V_0 for the term? 
                             # Actually if i = n_inv-1 (q=Q), bid fill is impossible.
                             # p_b must be 0.
                             # We can enforce this by setting W_b such that lambda_b -> 0.
                             # Or just skip optimization and set d_b = inf (or large).
                             # The problem says "no bid fill at q=Q".
                             # So we should not quote bid. d_b = inf.
                             # In our lambda formulation, lambda -> 0 means d -> inf.
                             # So we can just let the solver find lambda ~ 0.
                             # To be safe, if at boundary, we can fix lambda to 0.
            
            if i > 0:
                V_minus = V_next[i - 1]
            else:
                V_minus = V_0
            
            # If at boundary, we force no quote on that side.
            # lambda = 0 => d = inf.
            
            if i == n_inv - 1:
                lam_b = 0.0
            else:
                # We need W_b which depends on lam_a.
                # We will do coordinate ascent.
                lam_a = 1.0
                lam_b = 1.0
                
                # Initialize p_a
                p_a = 1.0 - np.exp(-lam_a)
                
                for _ in range(5): # Iterate a few times
                    # Solve for lambda_b given lambda_a
                    if i < n_inv - 1:
                        # W_b = (V_plus - V_0) - p_a * (V_plus + V_minus - 2*V_0)
                        W_b = (V_plus - V_0) - p_a * (V_plus + V_minus - 2.0 * V_0)
                        lam_b = _solve_lambda_1d(W_b, k, dt, A)
                        p_b = 1.0 - np.exp(-lam_b)
                    else:
                        lam_b = 0.0
                        p_b = 0.0
                        
                    # Solve for lambda_a given lambda_b
                    if i > 0:
                        # W_a = (V_minus - V_0) - p_b * (V_plus + V_minus - 2*V_0)
                        W_a = (V_minus - V_0) - p_b * (V_plus + V_minus - 2.0 * V_0)
                        lam_a = _solve_lambda_1d(W_a, k, dt, A)
                        p_a = 1.0 - np.exp(-lam_a)
                    else:
                        lam_a = 0.0
                        p_a = 0.0

            # Compute depths
            # d = -1/k * ln(lambda / (A dt)) = 1/k * ln(A dt / lambda)
            if lam_b > 0:
                d_b = (1.0 / k) * np.log(A * dt / lam_b)
            else:
                d_b = np.inf
                
            if lam_a > 0:
                d_a = (1.0 / k) * np.log(A * dt / lam_a)
            else:
                d_a = np.inf
                
            bid_depths_table[t, i] = d_b
            ask_depths_table[t, i] = d_a
            
            # Compute V_t(q)
            # E[Depth] = p_b * d_b - p_a * d_a
            # E[V_next] = V_0 + p_b(V_plus - V_0) + p_a(V_minus - V_0) - p_b p_a (V_plus + V_minus - 2 V_0)
            
            if i < n_inv - 1 and i > 0:
                E_V = V_0 + p_b * (V_plus - V_0) + p_a * (V_minus - V_0) - p_b * p_a * (V_plus + V_minus - 2.0 * V_0)
            elif i < n_inv - 1:
                # No ask fill
                E_V = V_0 + p_b * (V_plus - V_0)
            elif i > 0:
                # No bid fill
                E_V = V_0 + p_a * (V_minus - V_0)
            else:
                E_V = V_0
                
            E_Depth = p_b * d_b - p_a * d_a
            V[t, i] = E_Depth + E_V - cost[i]
            
    return bid_depths_table, ask_depths_table

def _get_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _policy_cache:
        _policy_cache[key] = _solve_dp_continuous(sigma, A, k, gamma, T, steps, Q)
    return _policy_cache[key]

def quote(t, inventory, mid, params):
    t_start = time.time()
    deadline = t_start + 290
    
    try:
        bid_table, ask_table = _get_policy(params)
        
        steps = params['steps']
        Q = params['Q']
        
        dt = params['T'] / steps
        t_step = int(round(t / dt))
        t_step = max(0, min(t_step, steps - 1))
        
        inv_idx = inventory + Q
        inv_idx = np.clip(inv_idx, 0, 2 * Q)
        
        bid_depths = bid_table[t_step, inv_idx]
        ask_depths = ask_table[t_step, inv_idx]
        
        # Ensure validity
        # The problem says values below -3/k are invalid.
        # Our depths are always >= 0 (since lambda <= A*dt usually? 
        # If lambda > A*dt, log(A*dt/lambda) < 0, so depth < 0.
        # Is lambda bounded by A*dt?
        # p = 1 - exp(-lambda). Max p is 1.
        # If lambda is large, p->1.
        # The optimal lambda shouldn't be huge because d becomes negative.
        # We should clip depths to be >= 0.
        
        bid_depths = np.maximum(bid_depths, 0.0)
        ask_depths = np.maximum(ask_depths, 0.0)
        
        return bid_depths, ask_depths
    except Exception:
        k = params['k']
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
# EVOLVE-BLOCK-END
