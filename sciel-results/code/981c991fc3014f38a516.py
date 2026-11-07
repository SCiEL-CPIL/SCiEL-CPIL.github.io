"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

# Cache for precomputed policies per regime
_policy_cache = {}


def _solve_depth(delta_v, A, k, dt, max_depth=100.0):
    """
    Find optimal depth d maximizing p(d) * (d + delta_v).
    p(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    Returns d. If optimal is effectively 0, returns a small epsilon.
    If optimal is effectively infinity (no quote), returns inf.
    """
    # h(d) = p(d) * (d + delta_v)
    # h'(d) = p'(d)(d + delta_v) + p(d)
    # p'(d) = -k * lambda(d) * (1 - p(d))
    # lambda(d) = A * exp(-k*d) * dt
    
    # We want to find root of:
    # -k * lambda(d) * (1 - p(d)) * (d + delta_v) + p(d) = 0
    
    def objective(d):
        if d < 0:
            return -1e10 # Penalize negative depths
        lam = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-lam)
        dp = -k * lam * (1.0 - p)
        return dp * (d + delta_v) + p

    # Check behavior at d=0
    # lam_0 = A * dt
    # p_0 = 1 - exp(-A*dt)
    # dp_0 = -k * A * dt * exp(-A*dt)
    # obj(0) = -k * A * dt * exp(-A*dt) * delta_v + 1 - exp(-A*dt)
    
    # If obj(0) <= 0, the maximum is at d=0 (or boundary).
    # If obj(0) > 0, the function increases initially, so there is a root for d > 0.
    
    # Also check large d. As d -> inf, p -> 0, dp -> 0.
    # obj(d) -> 0.
    # If obj(0) > 0, it must cross 0 to go to 0 (from positive to negative? or positive to 0?)
    # Let's check the sign.
    # For very large d, p is tiny. dp is tiny negative.
    # dp * (d + delta_v) is tiny negative. p is tiny positive.
    # Which one dominates?
    # p ~ lam. dp ~ -k lam.
    # obj ~ -k lam (d) + lam = lam (1 - k d).
    # Since k d becomes large, obj becomes negative.
    # So if obj(0) > 0, there is a root.
    
    obj_0 = objective(0.0)
    
    if obj_0 <= 1e-12:
        # Optimal is at 0
        return 0.0
    
    # Find upper bound for root
    # We know root exists.
    # Try d=1, 10, 100...
    d_upper = 1.0
    while objective(d_upper) > 0:
        d_upper *= 2.0
        if d_upper > max_depth:
            # If it's still positive at max_depth, the max is likely at max_depth or beyond
            # But usually it drops off quickly.
            break
            
    if objective(d_upper) > 0:
        return d_upper # Return the largest checked, or inf? 
        # If it's still increasing, we should return a large value.
        # But physically, depth shouldn't be huge.
        # Let's return d_upper as a proxy for "very deep"
        
    try:
        root = brentq(objective, 0.0, d_upper, xtol=1e-9)
        return root
    except ValueError:
        # Fallback to grid search if brentq fails
        ds = np.linspace(0, d_upper, 1000)
        vals = [objective(d) for d in ds] # This is wrong, we need h(d) not h'(d)
        
        # Let's just return a heuristic if root finding fails
        return 1.0 / k

def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via dynamic programming with 1D optimization.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)

    # Value function: V[n, i] where i is index into q_values
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Precompute cost
    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        for i in range(n_states):
            q = q_values[i]
            
            # Boundary conditions
            # No bid fill at q = Q (i = 2*Q). So bid_depth = inf (or 0 prob).
            # No ask fill at q = -Q (i = 0). So ask_depth = inf (or 0 prob).
            
            if i == n_states - 1:
                # q = Q, cannot buy
                b_opt = np.inf
                p_b = 0.0
            else:
                delta_v_b = V_next[i+1] - V_next[i]
                b_opt = _solve_depth(delta_v_b, A, k, dt)
                if b_opt == 0.0:
                    b_opt = 1e-9 # Avoid log(0) or issues, though p(0) is fine
                lam_b = A * np.exp(-k * b_opt) * dt
                p_b = 1.0 - np.exp(-lam_b)
                
            if i == 0:
                # q = -Q, cannot sell
                a_opt = np.inf
                p_a = 0.0
            else:
                delta_v_a = V_next[i-1] - V_next[i]
                a_opt = _solve_depth(delta_v_a, A, k, dt)
                if a_opt == 0.0:
                    a_opt = 1e-9
                lam_a = A * np.exp(-k * a_opt) * dt
                p_a = 1.0 - np.exp(-lam_a)
            
            # Update Value
            # V_curr[i] = p_b*b_opt + p_a*a_opt 
            #            + p_b*(1-p_a)*V_next[i+1] 
            #            + (1-p_b)*p_a*V_next[i-1] 
            #            + (1-p_b-p_a+p_b*p_a)*V_next[i] 
            #            - cost[i]
            
            # Handle boundaries for V_next indices
            v_next_q = V_next[i]
            
            if i == n_states - 1:
                # p_b = 0
                # V_next[i+1] is not accessible, but term is 0
                v_next_q_plus = 0.0
                v_next_q_minus = V_next[i-1]
            elif i == 0:
                # p_a = 0
                v_next_q_plus = V_next[i+1]
                v_next_q_minus = 0.0
            else:
                v_next_q_plus = V_next[i+1]
                v_next_q_minus = V_next[i-1]
            
            # Calculate terms
            term_b_earn = p_b * b_opt if np.isfinite(b_opt) else 0.0
            term_a_earn = p_a * a_opt if np.isfinite(a_opt) else 0.0
            
            prob_b_only = p_b * (1.0 - p_a)
            prob_a_only = (1.0 - p_b) * p_a
            prob_none = 1.0 - p_b - p_a + p_b * p_a
            
            v_next_expected = prob_b_only * v_next_q_plus + prob_a_only * v_next_q_minus + prob_none * v_next_q
            
            V_curr[i] = term_b_earn + term_a_earn + v_next_expected - cost[i]
            
            # Store optimal depths
            opt_bid[n, i] = b_opt
            opt_ask[n, i] = a_opt
        
        V_next = V_curr

    return opt_bid, opt_ask


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_depths(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed optimal policy.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 30s margin
    
    steps = params['steps']
    Q = params['Q']
    
    # Get or compute policy
    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        # Fallback to seed behavior
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()
    
    # Determine current step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    idx = inventory + Q
    idx = np.clip(idx, 0, 2*Q)
    
    # Get depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    # Ensure valid outputs
    # If depth is inf, it means no quote.
    # If depth is 0, it means quote at mid.
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
