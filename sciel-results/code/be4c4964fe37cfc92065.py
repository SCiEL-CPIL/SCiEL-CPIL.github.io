import numpy as np
from scipy.optimize import brentq
import time

# Cache for precomputed tables
_TABLE_CACHE = {}
_DEADLINE = 325.0  # Safety margin before 330s kill
_START_TIME = time.time()

def _check_deadline():
    """Check if we're approaching the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE

def _solve_depth(C, A, k, dt):
    """
    Solve for optimal depth given constant C where:
    J(d) = p(d)*d + C*p(d), p(d) = 1 - exp(-A*exp(-k*d)*dt)
    Optimal d satisfies: p'(d)*(d+C) + p(d) = 0
    
    Uses closed-form approximation for small A*dt, with brentq fallback.
    """
    p_0 = 1.0 - np.exp(-A * dt)
    if p_0 <= 0:
        return 0.0
    
    # Check if d=0 is optimal
    p_0_prime = -k * A * dt * (1 - p_0)
    val_0 = p_0_prime * C + p_0
    if val_0 <= 0:
        return 0.0
    
    # Closed-form approximation: for small A*dt, p(d) ≈ A*exp(-k*d)*dt
    # Then f(d) ≈ A*dt*exp(-k*d)*(1 - k*(d+C))
    # Root: d = 1/k - C
    d_approx = 1.0/k - C
    if d_approx <= 0:
        return 0.0
    
    # Verify with exact function
    def f(d):
        u = np.exp(-k * d)
        p = 1.0 - np.exp(-A * u * dt)
        dp = -k * A * dt * u * (1 - p)
        return dp * (d + C) + p
    
    # Check if approximation is good
    f_approx = f(d_approx)
    if abs(f_approx) < 1e-6:
        return d_approx
    
    # Use brentq with bracket
    d_upper = max(5.0/k, d_approx * 2 + 1.0/k)
    try:
        d_sol = brentq(f, 0, d_upper, xtol=1e-10)
        return max(0.0, d_sol)
    except ValueError:
        # Fallback to approximation
        return max(0.0, d_approx)

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories.
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Precompute p_0 for reuse
    p_0 = 1.0 - np.exp(-A * dt)
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Fill remaining with default depths
            default_depth = 1.0 / k
            for n_rem in range(n, -1, -1):
                opt_bid[n_rem, :] = default_depth
                opt_ask[n_rem, :] = default_depth
            break
        
        V_curr = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_idx - Q
            
            # Get deltas from V_next
            V_q = V_next[q_idx]
            
            if q < max_q:
                V_q_plus = V_next[q_idx + 1]
                delta_plus = V_q_plus - V_q
            else:
                delta_plus = 0.0 

            if q > min_q:
                V_q_minus = V_next[q_idx - 1]
                delta_minus = V_q_minus - V_q
            else:
                delta_minus = 0.0

            # Solve for optimal b and a with fixed-point iteration
            b = 0.0
            a = 0.0
            
            for _iter in range(15):
                # Solve for b given a
                if q < max_q:
                    p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    C_b = delta_plus * (1.0 - p_a_curr)
                    b_new = _solve_depth(C_b, A, k, dt)
                else:
                    b_new = 0.0

                # Solve for a given b
                if q > min_q:
                    p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
                    C_a = delta_minus * (1.0 - p_b_curr)
                    a_new = _solve_depth(C_a, A, k, dt)
                else:
                    a_new = 0.0

                # Check convergence
                if abs(b_new - b) < 1e-7 and abs(a_new - a) < 1e-7:
                    b = b_new
                    a = a_new
                    break
                b = b_new
                a = a_new
            
            opt_bid[n, q_idx] = b
            opt_ask[n, q_idx] = a
            
            # Calculate V_curr[q_idx]
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
            
            # Risk term
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
    
    # Check cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _TABLE_CACHE:
        # Check deadline before starting precomputation
        if _check_deadline():
            depth = 1.0 / params["k"]
            return np.full(len(inventory), depth), np.full(len(inventory), depth)
        
        opt_bid, opt_ask = _precompute_regime(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
        _TABLE_CACHE[key] = (opt_bid, opt_ask)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    # Map inventory to indices
    q_indices = np.asarray(inventory, dtype=np.int64) + params['Q']
    
    # Look up depths
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    # Ensure valid outputs
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
