"""
Strategy: Continuous Depth Optimization via Analytical Derivatives.
Instead of searching a discrete grid for the optimal depth, we compute the 
expected value function's derivative with respect to depth analytically and 
solve for the root (optimal depth) using a scalar root-finder. This eliminates 
grid discretization error entirely and allows for exact optimal depths within 
the continuous domain, provided the value function is sufficiently smooth 
(which it is, given the exponential fill probabilities).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

# Cache for pre-computed optimal policies
_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 325.0  # Safety margin (evaluator kills at 330s)

def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all inventory states q using 
    Value Iteration with continuous depth optimization.
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
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Precompute cost for each state
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt
    
    # We iterate to find the stationary policy.
    # V is the value function for each state.
    V = np.zeros(n_states)
    
    # Optimal depths for each state
    opt_bid_q = np.zeros(n_states)
    opt_ask_q = np.zeros(n_states)
    
    # Define the derivative of the expected value w.r.t depth d
    # E[d] = p_bid(d) * (V[q-1] + d) + p_ask(d) * (V[q+1] + d) 
    #       + (1-p_bid(d))(1-p_ask(d)) * V[q]
    #       - cost
    # Note: p_bid(d) = 1 - exp(-A exp(-k d) dt)
    #       p_ask(d) = 1 - exp(-A exp(-k d) dt) (symmetric)
    # Let P(d) = p_bid(d) = p_ask(d) for simplicity in this symmetric model.
    # E[d] = P(d)(V_{q-1} + d) + P(d)(V_{q+1} + d) + (1-P(d))^2 V_q - cost
    # 
    # We need dE/dd = 0.
    # P'(d) = A dt exp(-k d) exp(-A exp(-k d) dt)
    
    def derivative_bid(d, v_q, v_qm, v_qp):
        """
        Derivative of expected value w.r.t bid depth d.
        d is bid depth.
        v_q: V[q]
        v_qm: V[q+1] (ask side, inventory increases) -> Wait, let's check inventory logic.
        
        Inventory q.
        Bid hit: q -> q-1. Earn d.
        Ask hit: q -> q+1. Earn d (ask depth).
        
        In the Bellman update for bid depth d_b:
        p_b = 1 - exp(-A exp(-k d_b) dt)
        p_a = 1 - exp(-A exp(-k d_a) dt) (d_a is fixed from previous iteration or same step)
        
        Term: p_b * (V[q-1] + d_b) + p_a * (V[q+1] + d_a) + (1-p_b)(1-p_a) V[q] - cost
        d/dd_b:
        p_b' * (V[q-1] + d_b) + p_b
        - p_b' * (1-p_a) V[q]
        
        So: p_b' * (V[q-1] + d_b - (1-p_a)V[q]) + p_b
        """
        exp_term = np.exp(-k * d)
        p = 1.0 - np.exp(-A * exp_term * dt)
        p_prime = A * dt * exp_term * np.exp(-A * exp_term * dt)
        
        # p_a and v_qm, v_qp are constants for this derivative call
        # Note: v_qp is V[q-1], v_qm is V[q+1]
        # Wait, in my previous code:
        # v_qp = V[i+1] (index i+1 is q+1? No. q_values[i] = -Q + i. 
        # If q is at index i, q-1 is at i-1, q+1 is at i+1.
        # Let's stick to indices.
        # State i corresponds to q = -Q + i.
        # Bid hit: q -> q-1 (index i-1). Value V[i-1].
        # Ask hit: q -> q+1 (index i+1). Value V[i+1].
        
        # p_b' * (V[i-1] + d - (1-p_a)V[i]) + p_b
        
        # We need p_a and V[i], V[i-1], V[i+1].
        # These are passed in.
        
        p_a_val = p_a  # constant
        v_i = v_q
        v_im1 = v_qp  # V[q-1]
        
        return p_prime * (v_im1 + d - (1.0 - p_a_val) * v_i) + p

    def derivative_ask(d, v_q, v_qm, v_qp):
        """
        Derivative of expected value w.r.t ask depth d.
        Symmetric to bid, but inventory moves to q+1.
        Term: p_a * (V[q+1] + d) + ...
        d/dd_a: p_a' * (V[q+1] + d - (1-p_b)V[q]) + p_a
        """
        exp_term = np.exp(-k * d)
        p = 1.0 - np.exp(-A * exp_term * dt)
        p_prime = A * dt * exp_term * np.exp(-A * exp_term * dt)
        
        p_b_val = p_b  # constant
        v_i = v_q
        v_ip1 = v_qm   # V[q+1]
        
        return p_prime * (v_ip1 + d - (1.0 - p_b_val) * v_i) + p

    max_iter = 100
    tol = 1e-5
    
    # Initial guess for depths: small positive
    cur_bid = np.full(n_states, 0.1)
    cur_ask = np.full(n_states, 0.1)
    
    # Boundaries for depth search
    d_min = 1e-6
    d_max = 5.0 # 3/k is ~2 for k=1.5, 4 for k=0.75. 5 is safe.
    
    for it in range(max_iter):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        new_V = np.zeros(n_states)
        new_bid = np.zeros(n_states)
        new_ask = np.zeros(n_states)
        
        for i in range(n_states):
            v_q = V[i]
            
            # Handle boundaries
            if i == 0: # q = -Q, no ask fill possible (inventory would go to -Q-1)
                # Actually, contract says "no ask fill at q = -Q". 
                # So p_ask is effectively 0? Or we just don't quote ask?
                # "no ask fill at q = -Q" implies we should not quote ask, or if we do, it doesn't fill.
                # Optimal is to set ask depth to inf (no quote) or just handle probability 0.
                # If we set depth to inf, p=0.
                # Let's just optimize bid, and set ask to inf (or a very large number, but inf is allowed).
                # However, the problem says "np.inf means no quote".
                # So for q=-Q, ask_depth = inf.
                # For q=Q, bid_depth = inf.
                
                # Optimize Bid
                v_im1 = V[i-1] if i > 0 else 0.0 # If i=0, q=-Q, bid hit goes to -Q-1 which is invalid.
                # Wait, "no bid fill at q = Q". 
                # At q=-Q, bid hit is allowed? 
                # "no bid fill at q = Q, no ask fill at q = -Q".
                # So at q=-Q (i=0), bid hit is allowed (goes to -Q-1? No, inventory is in [-Q, Q]).
                # Usually this means the state space is bounded, and transitions outside are not allowed or value is 0?
                # The contract says "inventory (4000,) ints in [-Q, Q]".
                # If a fill moves it out, what happens?
                # "A fill ... moves inventory one unit".
                # If q=-Q and bid hit, q becomes -Q-1. But inventory must be in [-Q, Q].
                # This implies that at boundaries, the corresponding fill probability is 0.
                # "no bid fill at q = Q" -> if q=Q, bid hit prob = 0.
                # "no ask fill at q = -Q" -> if q=-Q, ask hit prob = 0.
                
                # So:
                # If i == 0 (q = -Q):
                #   Ask hit prob = 0.
                #   Bid hit prob > 0.
                #   Transition: q -> q-1? No, if ask is blocked, what about bid?
                #   If bid hits, q becomes -Q-1. Is that allowed?
                #   The state space is [-Q, Q]. 
                #   Usually in these problems, the boundary condition implies the transition is disallowed 
                #   or the value of the out-of-bounds state is treated as 0 (absorbing).
                #   Given "no ask fill at q=-Q", it likely means p_ask = 0.
                #   What about bid at q=-Q? It's not explicitly forbidden. 
                #   But if it moves to -Q-1, and -Q-1 is not a state, we assume V[-Q-1] = 0.
                #   Let's assume V_out = 0.
                
                v_im1 = 0.0 # V[q-1] is out of bounds
                v_ip1 = V[i+1] # V[q+1] is valid
                p_a = 0.0 # Ask is blocked
                
                # Optimize bid
                def f_bid(d):
                    return derivative_bid(d, v_q, v_ip1, v_im1, p_a)
                # We need to pass p_a to derivative_bid. 
                # My derivative_bid function above uses p_a from closure. 
                # I need to make it accept p_a or use a lambda.
                
                # Let's refactor derivative functions to take all constants.
                
                try:
                    # Check sign change
                    f_min = f_bid(d_min)
                    f_max = f_bid(d_max)
                    if f_min * f_max < 0:
                        d_opt = brentq(f_bid, d_min, d_max, xtol=1e-6)
                    else:
                        # If no sign change, check which end is better or if derivative is always positive/negative
                        # If f(d) > 0 everywhere, derivative is positive -> value increasing -> max at d_max?
                        # Wait, if dE/dd > 0, we should increase d.
                        # If dE/dd < 0, we should decrease d.
                        # If f_min > 0 and f_max > 0, max is at d_max.
                        # If f_min < 0 and f_max < 0, max is at d_min.
                        if f_min > 0:
                            d_opt = d_max
                        else:
                            d_opt = d_min
                except Exception:
                    d_opt = 0.1
                
                new_bid[i] = d_opt
                new_ask[i] = np.inf
                
                # Compute V
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_opt) * dt)
                # E = p_b (V_im1 + d_opt) + 0 + (1-p_b)(1-0) V_q - cost
                # Note: (1-p_b)(1-p_a) = (1-p_b)(1) = 1-p_b
                # Wait, if p_a=0, then (1-p_a)=1.
                # Term: p_b(V_im1 + d) + (1-p_b)V_q
                new_V[i] = p_b * (v_im1 + d_opt) + (1.0 - p_b) * v_q - cost_per_state[i]

            elif i == n_states - 1: # q = Q
                # Bid is blocked.
                v_im1 = V[i-1]
                v_ip1 = 0.0 # Out of bounds
                p_b = 0.0 # Bid is blocked
                
                def f_ask(d):
                    return derivative_ask(d, v_q, v_ip1, v_im1, p_b)
                
                try:
                    f_min = f_ask(d_min)
                    f_max = f_ask(d_max)
                    if f_min * f_max < 0:
                        d_opt = brentq(f_ask, d_min, d_max, xtol=1e-6)
                    else:
                        if f_min > 0:
                            d_opt = d_max
                        else:
                            d_opt = d_min
                except Exception:
                    d_opt = 0.1
                
                new_bid[i] = np.inf
                new_ask[i] = d_opt
                
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_opt) * dt)
                new_V[i] = p_a * (v_ip1 + d_opt) + (1.0 - p_a) * v_q - cost_per_state[i]

            else:
                # Interior state
                v_im1 = V[i-1]
                v_ip1 = V[i+1]
                
                # First, optimize bid assuming current ask depth (or previous iteration's)
                # We can do a Gauss-Seidel style update or simultaneous.
                # Let's use the current iteration's ask depth if available, else previous.
                # Since we are in a loop over i, we can use new_ask[i] if already computed?
                # No, we are computing i sequentially. new_ask[i] is not computed yet.
                # We can use cur_ask[i] from the previous outer iteration.
                
                p_a_curr = 1.0 - np.exp(-A * np.exp(-k * cur_ask[i]) * dt)
                
                def f_bid(d):
                    return derivative_bid(d, v_q, v_ip1, v_im1, p_a_curr)
                
                try:
                    f_min = f_bid(d_min)
                    f_max = f_bid(d_max)
                    if f_min * f_max < 0:
                        d_b_opt = brentq(f_bid, d_min, d_max, xtol=1e-6)
                    else:
                        if f_min > 0:
                            d_b_opt = d_max
                        else:
                            d_b_opt = d_min
                except Exception:
                    d_b_opt = 0.1
                
                # Now optimize ask assuming new bid depth
                p_b_new = 1.0 - np.exp(-A * np.exp(-k * d_b_opt) * dt)
                
                def f_ask(d):
                    return derivative_ask(d, v_q, v_ip1, v_im1, p_b_new)
                
                try:
                    f_min = f_ask(d_min)
                    f_max = f_ask(d_max)
                    if f_min * f_max < 0:
                        d_a_opt = brentq(f_ask, d_min, d_max, xtol=1e-6)
                    else:
                        if f_min > 0:
                            d_a_opt = d_max
                        else:
                            d_a_opt = d_min
                except Exception:
                    d_a_opt = 0.1
                
                new_bid[i] = d_b_opt
                new_ask[i] = d_a_opt
                
                # Compute V
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b_opt) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a_opt) * dt)
                
                # E = p_b(V_im1 + d_b) + p_a(V_ip1 + d_a) + (1-p_b)(1-p_a)V_q - cost
                # Note: (1-p_b)(1-p_a) = 1 - p_b - p_a + p_b p_a
                new_V[i] = p_b * (v_im1 + d_b_opt) + p_a * (v_ip1 + d_a_opt) + (1.0 - p_b) * (1.0 - p_a) * v_q - cost_per_state[i]

        # Update V and policies
        V = new_V
        cur_bid = new_bid
        cur_ask = new_ask
        
        # Convergence check
        diff = np.max(np.abs(new_V - V))
        if diff < tol:
            break

    # Construct the full policy arrays of shape (steps, n_states)
    # Since it's stationary, we just tile the q-based policy
    # Replace inf with a large number? No, quote() returns inf for no quote.
    # But we need to return arrays. 
    # The contract says "np.inf means no quote".
    # So we keep inf.
    
    opt_bid = np.tile(cur_bid[None, :], (steps, 1))
    opt_ask = np.tile(cur_ask[None, :], (steps, 1))
    
    return opt_bid, opt_ask

# Helper functions for derivative (defined inside _compute_optimal_policy to capture dt, A, k)
# Wait, I defined them inside the loop in the draft above? No, I defined them outside in the draft.
# I need to make sure they have access to dt, A, k.
# I will define them as nested functions inside _compute_optimal_policy.

def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all inventory states q using 
    Value Iteration with continuous depth optimization.
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
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Precompute cost for each state
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt
    
    # We iterate to find the stationary policy.
    # V is the value function for each state.
    V = np.zeros(n_states)
    
    # Optimal depths for each state
    cur_bid = np.full(n_states, 0.1)
    cur_ask = np.full(n_states, 0.1)
    
    # Boundaries for depth search
    d_min = 1e-6
    d_max = 5.0
    
    # Nested derivative functions
    def calc_p(d):
        return 1.0 - np.exp(-A * np.exp(-k * d) * dt)
        
    def calc_p_prime(d):
        exp_term = np.exp(-k * d)
        return A * dt * exp_term * np.exp(-A * exp_term * dt)

    def derivative_bid(d, v_q, v_qm, v_qp, p_a):
        """
        d/dd_b [ p_b (V_qp + d) + p_a (V_qm + d_a) + (1-p_b)(1-p_a) V_q - cost ]
        = p_b' (V_qp + d) + p_b - p_b' (1-p_a) V_q
        = p_b' (V_qp + d - (1-p_a)V_q) + p_b
        """
        p = calc_p(d)
        pp = calc_p_prime(d)
        return pp * (v_qp + d - (1.0 - p_a) * v_q) + p

    def derivative_ask(d, v_q, v_qm, v_qp, p_b):
        """
        d/dd_a [ p_a (V_qm + d) + p_b (V_qp + d_b) + (1-p_b)(1-p_a) V_q - cost ]
        = p_a' (V_qm + d) + p_a - p_a' (1-p_b) V_q
        = p_a' (V_qm + d - (1-p_b)V_q) + p_a
        """
        p = calc_p(d)
        pp = calc_p_prime(d)
        return pp * (v_qm + d - (1.0 - p_b) * v_q) + p

    max_iter = 100
    tol = 1e-5
    
    for it in range(max_iter):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        new_V = np.zeros(n_states)
        new_bid = np.zeros(n_states)
        new_ask = np.zeros(n_states)
        
        for i in range(n_states):
            v_q = V[i]
            
            if i == 0: # q = -Q
                v_qp = 0.0 # V[q-1] out of bounds
                v_qm = V[i+1]
                p_a = 0.0 # Ask blocked
                
                def f_bid(d):
                    return derivative_bid(d, v_q, v_qm, v_qp, p_a)
                
                try:
                    f_min = f_bid(d_min)
                    f_max = f_bid(d_max)
                    if f_min * f_max < 0:
                        d_opt = brentq(f_bid, d_min, d_max, xtol=1e-6)
                    else:
                        if f_min > 0:
                            d_opt = d_max
                        else:
                            d_opt = d_min
                except Exception:
                    d_opt = 0.1
                
                new_bid[i] = d_opt
                new_ask[i] = np.inf
                
                p_b = calc_p(d_opt)
                new_V[i] = p_b * (v_qp + d_opt) + (1.0 - p_b) * v_q - cost_per_state[i]

            elif i == n_states - 1: # q = Q
                v_qp = V[i-1]
                v_qm = 0.0 # V[q+1] out of bounds
                p_b = 0.0 # Bid blocked
                
                def f_ask(d):
                    return derivative_ask(d, v_q, v_qm, v_qp, p_b)
                
                try:
                    f_min = f_ask(d_min)
                    f_max = f_ask(d_max)
                    if f_min * f_max < 0:
                        d_opt = brentq(f_ask, d_min, d_max, xtol=1e-6)
                    else:
                        if f_min > 0:
                            d_opt = d_max
                        else:
                            d_opt = d_min
                except Exception:
                    d_opt = 0.1
                
                new_bid[i] = np.inf
                new_ask[i] = d_opt
                
                p_a = calc_p(d_opt)
                new_V[i] = p_a * (v_qm + d_opt) + (1.0 - p_a) * v_q - cost_per_state[i]

            else:
                v_qp = V[i-1]
                v_qm = V[i+1]
                
                p_a_curr = calc_p(cur_ask[i])
                
                def f_bid(d):
                    return derivative_bid(d, v_q, v_qm, v_qp, p_a_curr)
                
                try:
                    f_min = f_bid(d_min)
                    f_max = f_bid(d_max)
                    if f_min * f_max < 0:
                        d_b_opt = brentq(f_bid, d_min, d_max, xtol=1e-6)
                    else:
                        if f_min > 0:
                            d_b_opt = d_max
                        else:
                            d_b_opt = d_min
                except Exception:
                    d_b_opt = 0.1
                
                p_b_new = calc_p(d_b_opt)
                
                def f_ask(d):
                    return derivative_ask(d, v_q, v_qm, v_qp, p_b_new)
                
                try:
                    f_min = f_ask(d_min)
                    f_max = f_ask(d_max)
                    if f_min * f_max < 0:
                        d_a_opt = brentq(f_ask, d_min, d_max, xtol=1e-6)
                    else:
                        if f_min > 0:
                            d_a_opt = d_max
                        else:
                            d_a_opt = d_min
                except Exception:
                    d_a_opt = 0.1
                
                new_bid[i] = d_b_opt
                new_ask[i] = d_a_opt
                
                p_b = calc_p(d_b_opt)
                p_a = calc_p(d_a_opt)
                
                new_V[i] = p_b * (v_qp + d_b_opt) + p_a * (v_qm + d_a_opt) + (1.0 - p_b) * (1.0 - p_a) * v_q - cost_per_state[i]

        V = new_V
        cur_bid = new_bid
        cur_ask = new_ask
        
        diff = np.max(np.abs(new_V - V))
        if diff < tol:
            break

    opt_bid = np.tile(cur_bid[None, :], (steps, 1))
    opt_ask = np.tile(cur_ask[None, :], (steps, 1))
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    # Ensure validity: values below -3/k are invalid.
    # Our depths are positive or inf.
    # Check for NaN/inf issues? Inf is allowed.
    
    return bids, asks

# EVOLVE-BLOCK-END
