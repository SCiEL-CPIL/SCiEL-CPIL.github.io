"""Optimal market making with inventory risk using dynamic programming."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0  # Safety margin before 300s limit
_POLICY_CACHE = {}


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal policy using backward induction.
    
    The state is inventory q in [-Q, Q].
    At each step, we choose bid_depth b >= 0 and ask_depth a >= 0.
    
    The objective is to maximize:
      sum over steps of (depths earned) - sum over steps of (gamma/2) * sigma^2 * q^2 * dt
    
    where q is the inventory before that step's fills.
    
    Fill probabilities:
      P_bid = 1 - exp(-A * exp(-k * b))
      P_ask = 1 - exp(-A * exp(-k * a))
    
    If bid fills: earn b, inventory increases by 1 (q -> q+1)
    If ask fills: earn a, inventory decreases by 1 (q -> q-1)
    
    Constraints: no bid fill at q = Q, no ask fill at q = -Q.
    
    We use the fact that the optimal depth for a given continuation value difference
    can be found analytically or via scalar optimization.
    
    For bid: maximize P_bid * (b + V(q+1) - V(q)) - (gamma/2)*sigma^2*q^2*dt + V(q)
    Actually, let's be more careful.
    
    Let V_n(q) be the value function at step n with inventory q.
    
    V_{n+1}(q) = max_{b,a >= 0} [ 
        P_bid * (b + V_n(q+1)) + P_ask * (a + V_n(q-1)) + (1-P_bid)*(1-P_ask)*V_n(q)
        - (gamma/2)*sigma^2*q^2*dt
    ]
    
    where P_bid = 1 - exp(-A*exp(-k*b)), P_ask = 1 - exp(-A*exp(-k*a))
    
    With boundary conditions:
    - At q = Q: P_bid = 0 (no bid fill)
    - At q = -Q: P_ask = 0 (no ask fill)
    
    V_T(q) = 0 for all q (no terminal charge)
    
    The key insight: for fixed V_n, the optimal b and a can be found independently
    because the objective is separable in b and a (cross terms only through the
    (1-P_bid)(1-P_ask) term, but we can handle this).
    
    Actually, the cross term makes it coupled. Let me reconsider.
    
    Let p = P_bid, q_a = P_ask (using different variable name)
    
    Objective = p*(b + V(q+1)) + q_a*(a + V(q-1)) + (1-p)*(1-q_a)*V(q) - cost
    
    = p*b + p*V(q+1) + q_a*a + q_a*V(q-1) + V(q) - p*V(q) - q_a*V(q) + p*q_a*V(q) - cost
    
    = p*b + q_a*a + V(q) + p*(V(q+1)-V(q)) + q_a*(V(q-1)-V(q)) + p*q_a*V(q) - cost
    
    This is coupled through p*q_a*V(q). However, for small dt, p and q_a are small,
    so the coupling is weak. We can approximate by solving for b and a independently.
    
    For bid (assuming q_a is small/known):
    Maximize over b: p(b) * (b + C_bid) where C_bid = V(q+1) - V(q) + q_a*V(q)
    
    Wait, let me redo this more carefully.
    
    Let me define:
    - For bid: the contribution is p*b + p*V(q+1) + (1-p)*(1-q_a)*V(q) - p*(1-q_a)*V(q) ... 
    
    Let me just do it numerically with a grid search or scalar optimization for each state.
    
    Given the small number of states (2Q+1 = 61) and steps (1000), and the need to
    find optimal b and a for each state, we need an efficient approach.
    
    Key observation: the optimal depth for a problem of the form
    max_x (1 - exp(-alpha * exp(-k*x))) * (x + C)
    can be found efficiently.
    
    Let f(x) = (1 - exp(-alpha*exp(-k*x))) * (x + C)
    
    For C > 0: we want to balance getting a high probability with a decent depth.
    For C < 0: we might want to avoid fills (set x large so probability ~ 0) or
    accept a small depth with high probability if the net benefit is positive.
    
    Let's use scalar optimization with a good initial guess.
    """
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    # V[q_index] where q_index = q + Q
    V_next = np.zeros(n_states)
    
    # Precompute the optimal depth function
    # For given C (continuation value difference), alpha, k, find optimal x
    def _optimal_depth(C, alpha, k_val):
        """Find x >= 0 maximizing (1 - exp(-alpha*exp(-k*x))) * (x + C)"""
        if alpha <= 0:
            return 0.0
        
        # The function g(x) = (1 - exp(-alpha*exp(-k*x))) * (x + C)
        # For large x: exp(-k*x) -> 0, so g(x) -> 0 * (x+C) = 0 (if C > 0, it's 0 from below? No)
        # Actually as x -> inf: exp(-k*x) -> 0, 1 - exp(0) = 0, so g(x) -> 0
        # As x -> 0: 1 - exp(-alpha) times C
        # There's an interior maximum.
        
        # Use a simple approach: try a few candidate points and refine
        # The maximum is typically around x ~ 1/k to a few/k
        
        candidates = []
        # Add some heuristic candidates
        if C >= 0:
            # Want high probability, moderate depth
            candidates.append(1.0 / k_val)
            candidates.append(0.5 / k_val)
            candidates.append(2.0 / k_val)
        else:
            # C < 0, might want to avoid fills or accept small depth
            candidates.append(0.0)
            candidates.append(1.0 / k_val)
            candidates.append(3.0 / k_val)
            candidates.append(5.0 / k_val)
        
        # Evaluate and find best
        best_x = 0.0
        best_val = -np.inf
        for x in candidates:
            if x < 0:
                continue
            p = 1.0 - np.exp(-alpha * np.exp(-k_val * x))
            val = p * (x + C)
            if val > best_val:
                best_val = val
                best_x = x
        
        # Refine around best_x using a few local steps
        if best_val > -np.inf:
            x0 = best_x
            # Try nearby points
            for dx in [-0.5/k_val, -0.2/k_val, 0.0, 0.2/k_val, 0.5/k_val]:
                x = x0 + dx
                if x < 0:
                    continue
                p = 1.0 - np.exp(-alpha * np.exp(-k_val * x))
                val = p * (x + C)
                if val > best_val:
                    best_val = val
                    best_x = x
        
        return best_x
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        for qi in range(n_states):
            q = q_values[qi]
            cost = (gamma / 2.0) * sigma ** 2 * q ** 2 * dt
            
            if q >= Q:
                # No bid fill possible
                # Only ask side matters
                # Maximize over a: P_ask * (a + V_next[qi-1]) + (1-P_ask)*V_next[qi] - cost
                # where P_ask = 1 - exp(-A*exp(-k*a))
                C_ask = V_next[qi - 1] - V_next[qi] if qi > 0 else 0.0
                # If qi == 0, q = -Q, so q-1 is out of bounds, but q=-Q means no ask fill
                # Actually at q = -Q, no ask fill. So both sides are zero.
                if q == -Q:
                    V_curr[qi] = -cost
                    continue
                # At q = Q, only ask is possible
                a_opt = _optimal_depth(C_ask, A, k)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * a_opt))
                V_curr[qi] = p_ask * (a_opt + V_next[qi - 1]) + (1 - p_ask) * V_next[qi] - cost
            elif q <= -Q:
                # No ask fill possible
                # Only bid side matters
                C_bid = V_next[qi + 1] - V_next[qi] if qi < n_states - 1 else 0.0
                if q == -Q:
                    V_curr[qi] = -cost
                    continue
                b_opt = _optimal_depth(C_bid, A, k)
                p_bid = 1.0 - np.exp(-A * np.exp(-k * b_opt))
                V_curr[qi] = p_bid * (b_opt + V_next[qi + 1]) + (1 - p_bid) * V_next[qi] - cost
            else:
                # Both sides possible
                # Coupled problem, but we approximate by solving independently
                # First pass: assume no cross term
                C_bid = V_next[qi + 1] - V_next[qi]
                C_ask = V_next[qi - 1] - V_next[qi]
                
                b_opt = _optimal_depth(C_bid, A, k)
                a_opt = _optimal_depth(C_ask, A, k)
                
                p_bid = 1.0 - np.exp(-A * np.exp(-k * b_opt))
                p_ask = 1.0 - np.exp(-A * np.exp(-k * a_opt))
                
                # Compute the full objective with coupling
                val = (p_bid * (b_opt + V_next[qi + 1]) + 
                       p_ask * (a_opt + V_next[qi - 1]) + 
                       (1 - p_bid) * (1 - p_ask) * V_next[qi] - cost)
                
                V_curr[qi] = val
        
        V_next = V_curr
    
    # Now V_next is V_0, the value at step 0
    # We need to extract the optimal actions at each step
    # But we only need the policy for the first step (or all steps for caching)
    # Actually, we need to be able to quote at any step given inventory.
    # Let's recompute and store the optimal depths at each step.
    
    # Re-run backward induction storing optimal depths
    V_next = np.zeros(n_states)
    policy_bid = np.zeros((steps, n_states))
    policy_ask = np.zeros((steps, n_states))
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        for qi in range(n_states):
            q = q_values[qi]
            cost = (gamma / 2.0) * sigma ** 2 * q ** 2 * dt
            
            if q == Q:
                # No bid fill
                policy_bid[n, qi] = np.inf  # No bid quote
                if q == -Q:
                    policy_ask[n, qi] = np.inf
                    V_curr[qi] = -cost
                    continue
                C_ask = V_next[qi - 1] - V_next[qi]
                a_opt = _optimal_depth(C_ask, A, k)
                policy_ask[n, qi] = a_opt
                p_ask = 1.0 - np.exp(-A * np.exp(-k * a_opt))
                V_curr[qi] = p_ask * (a_opt + V_next[qi - 1]) + (1 - p_ask) * V_next[qi] - cost
            elif q == -Q:
                # No ask fill
                policy_ask[n, qi] = np.inf
                C_bid = V_next[qi + 1] - V_next[qi]
                b_opt = _optimal_depth(C_bid, A, k)
                policy_bid[n, qi] = b_opt
                p_bid = 1.0 - np.exp(-A * np.exp(-k * b_opt))
                V_curr[qi] = p_bid * (b_opt + V_next[qi + 1]) + (1 - p_bid) * V_next[qi] - cost
            else:
                C_bid = V_next[qi + 1] - V_next[qi]
                C_ask = V_next[qi - 1] - V_next[qi]
                
                b_opt = _optimal_depth(C_bid, A, k)
                a_opt = _optimal_depth(C_ask, A, k)
                
                policy_bid[n, qi] = b_opt
                policy_ask[n, qi] = a_opt
                
                p_bid = 1.0 - np.exp(-A * np.exp(-k * b_opt))
                p_ask = 1.0 - np.exp(-A * np.exp(-k * a_opt))
                
                V_curr[qi] = (p_bid * (b_opt + V_next[qi + 1]) + 
                             p_ask * (a_opt + V_next[qi - 1]) + 
                             (1 - p_bid) * (1 - p_ask) * V_next[qi] - cost)
        
        V_next = V_curr
    
    return policy_bid, policy_ask, V_next


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels
        mid: current mid price (not used in the model, but part of interface)
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two float arrays of shape (4000,)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return safe depths
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    # Get or compute policy
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _POLICY_CACHE:
        policy_bid, policy_ask, _ = _compute_policy(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[cache_key] = (policy_bid, policy_ask)
    else:
        policy_bid, policy_ask = _POLICY_CACHE[cache_key]
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to policy indices
    # inventory is in [-Q, Q], map to [0, 2Q]
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = policy_bid[n, q_idx].astype(np.float64)
    ask_depths = policy_ask[n, q_idx].astype(np.float64)
    
    # Ensure no negative depths and handle boundary cases
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # At q = Q, bid should be inf (no quote)
    bid_depths[inventory == Q] = np.inf
    # At q = -Q, ask should be inf (no quote)
    ask_depths[inventory == -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
