"""
Rewritten program: Discrete DP over inventory with continuous (Newton-Raphson) optimization over depth.
This eliminates the depth grid discretization error, allowing for near-exact optimal spreads.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _solve_optimal_depths(delta_v, A, k, dt, Q):
    """
    Solves for the optimal bid/ask depth for all inventory levels q in [-Q, Q].
    
    Parameters:
    delta_v: array of shape (2*Q+1,) representing V_{n+1}(q+1) - V_{n+1}(q) for bids 
             and V_{n+1}(q-1) - V_{n+1}(q) for asks? 
             Actually, for bid at q, the state change is q -> q+1. So we need V(q+1) - V(q).
             For ask at q, the state change is q -> q-1. So we need V(q-1) - V(q).
    
    We return two arrays: opt_bid_depths, opt_ask_depths.
    """
    n_states = 2 * Q + 1
    c = A * dt
    
    # We need to solve f(u) = (c/k) ln(u) - c * delta - 1/(k*u) + 1 = 0 for u = exp(-k*d)
    # d = -ln(u)/k
    
    # Initialize u with a reasonable guess. 
    # If delta is positive (buying is good), u is likely closer to 1 (small d).
    # If delta is negative, u is likely smaller (large d).
    # A safe initial guess is u = 0.5, or based on a heuristic.
    # Let's use u0 = exp(-k * 1.0/k) = exp(-1) ~ 0.37 as a neutral guess.
    u_bid = np.full(n_states, 0.5)
    u_ask = np.full(n_states, 0.5)
    
    # Deltas for bid: V(q+1) - V(q). For q=Q, this is invalid (no bid).
    # Deltas for ask: V(q-1) - V(q). For q=-Q, this is invalid (no ask).
    
    # We will compute deltas based on the value function passed in via a wrapper 
    # or we pass the V array directly. 
    # Let's refactor: this function should take V_next and return opt_depths.
    
    pass # Placeholder, will be implemented in the main DP loop for efficiency.


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal policy using DP over inventory and Newton-Raphson for depth.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    c = A * dt
    n_states = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Pre-compute penalty
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    V_next = np.zeros(n_states)
    opt_bid = np.full((steps, n_states), np.inf)
    opt_ask = np.full((steps, n_states), np.inf)
    
    # Newton-Raphson parameters
    # f(u) = (c/k)*ln(u) - c*delta - 1/(k*u) + 1
    # f'(u) = c/(k*u) + 1/(k*u^2)
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5.0:
            break

        V_curr = np.zeros(n_states)
        opt_bid_n = np.full(n_states, np.inf)
        opt_ask_n = np.full(n_states, np.inf)
        
        # Interior: q in [-Q+1, Q-1] -> indices 1 to 2Q-1
        if Q > 0:
            idx_int = np.arange(1, 2 * Q)
            
            # Compute deltas for bid and ask
            # Bid at q: next state q+1. Delta = V_next[q+1] - V_next[q]
            # Ask at q: next state q-1. Delta = V_next[q-1] - V_next[q]
            delta_bid = V_next[idx_int + 1] - V_next[idx_int]
            delta_ask = V_next[idx_int - 1] - V_next[idx_int]
            
            # Solve for optimal u for bid and ask using Newton-Raphson
            # We do this for all q in parallel.
            # Initial guess: u = 0.5
            u_b = np.full(len(idx_int), 0.5)
            u_a = np.full(len(idx_int), 0.5)
            
            # Newton iterations
            for _ in range(10):
                # Clamp u to avoid log(0) or division by zero
                u_b = np.clip(u_b, 1e-9, 1.0)
                u_a = np.clip(u_a, 1e-9, 1.0)
                
                # f(u) for bid
                f_b = (c / k) * np.log(u_b) - c * delta_bid - 1.0 / (k * u_b) + 1.0
                df_b = c / (k * u_b) + 1.0 / (k * u_b**2)
                u_b_new = u_b - f_b / df_b
                
                # f(u) for ask
                f_a = (c / k) * np.log(u_a) - c * delta_ask - 1.0 / (k * u_a) + 1.0
                df_a = c / (k * u_a) + 1.0 / (k * u_a**2)
                u_a_new = u_a - f_a / df_a
                
                u_b = u_b_new
                u_a = u_a_new
                
            # Convert u back to depth d = -ln(u)/k
            d_b = -np.log(np.clip(u_b, 1e-12, 1.0)) / k
            d_a = -np.log(np.clip(u_a, 1e-12, 1.0)) / k
            
            # Calculate value for the chosen depths
            # p_b = 1 - exp(-A * u_b * dt) = 1 - exp(-c * u_b)
            p_b = 1.0 - np.exp(-c * u_b)
            p_a = 1.0 - np.exp(-c * u_a)
            
            # V(q) = V_next(q) + p_b*d_b + p_a*d_a + p_b*delta_bid + p_a*delta_ask - penalty(q)
            # Note: delta_bid = V(q+1) - V(q). The term is p_b * (d_b + V(q+1)-V(q)).
            # Wait, the objective is: sum (depths earned) - penalty + E[V_next]
            # E[V_next] = (1-p_b)(1-p_a)V(q) + p_b(1-p_a)V(q+1) + (1-p_b)p_a V(q-1) + p_b p_a V(q+1-1?) 
            # Actually, bid hit moves q->q+1. Ask hit moves q->q-1.
            # If both hit, it's ambiguous or modeled as independent? 
            # The problem says "independently". 
            # If both hit, do we end up at q? Or is it one fill? 
            # "A fill earns its depth and moves inventory one unit". 
            # Usually in these models, if both hit, it's a net 0 change in inventory but you earn both spreads? 
            # Or does one cancel the other? 
            # "bid hit with prob... ask likewise, independently".
            # If both happen, q -> q+1 (bid) and q -> q-1 (ask). Net change 0. 
            # Reward: d_b + d_a.
            # So:
            # V(q) = (1-p_b)(1-p_a) V(q) + p_b(1-p_a) [V(q+1) + d_b] + (1-p_b)p_a [V(q-1) + d_a] + p_b p_a [V(q) + d_b + d_a]
            # This simplifies to:
            # V(q) = V(q) + p_b d_b + p_a d_a + p_b (V(q+1) - V(q)) + p_a (V(q-1) - V(q))
            # V(q) = p_b d_b + p_a d_a + p_b delta_bid + p_a delta_ask + V(q) - penalty? No, penalty is separate.
            # V_curr(q) = p_b d_b + p_a d_a + p_b delta_bid + p_a delta_ask - penalty(q) + V_next(q)
            
            val_b = p_b * (d_b + delta_bid)
            val_a = p_a * (d_a + delta_ask)
            
            V_curr[idx_int] = val_b + val_a - penalties[idx_int] + V_next[idx_int]
            
            opt_bid_n[idx_int] = d_b
            opt_ask_n[idx_int] = d_a
            
            # q = Q (index 2Q): only ask possible (bid blocked)
            idx_q = 2 * Q
            delta_a_Q = V_next[idx_q - 1] - V_next[idx_q]
            u_a_Q = 0.5
            for _ in range(10):
                u_a_Q = np.clip(u_a_Q, 1e-9, 1.0)
                f_a = (c / k) * np.log(u_a_Q) - c * delta_a_Q - 1.0 / (k * u_a_Q) + 1.0
                df_a = c / (k * u_a_Q) + 1.0 / (k * u_a_Q**2)
                u_a_Q = u_a_Q - f_a / df_a
            d_a_Q = -np.log(np.clip(u_a_Q, 1e-12, 1.0)) / k
            p_a_Q = 1.0 - np.exp(-c * u_a_Q)
            V_curr[idx_q] = p_a_Q * (d_a_Q + delta_a_Q) - penalties[idx_q] + V_next[idx_q]
            opt_ask_n[idx_q] = d_a_Q
            # opt_bid_n[idx_q] remains inf
            
            # q = -Q (index 0): only bid possible (ask blocked)
            idx_q = 0
            delta_b_mQ = V_next[idx_q + 1] - V_next[idx_q]
            u_b_mQ = 0.5
            for _ in range(10):
                u_b_mQ = np.clip(u_b_mQ, 1e-9, 1.0)
                f_b = (c / k) * np.log(u_b_mQ) - c * delta_b_mQ - 1.0 / (k * u_b_mQ) + 1.0
                df_b = c / (k * u_b_mQ) + 1.0 / (k * u_b_mQ**2)
                u_b_mQ = u_b_mQ - f_b / df_b
            d_b_mQ = -np.log(np.clip(u_b_mQ, 1e-12, 1.0)) / k
            p_b_mQ = 1.0 - np.exp(-c * u_b_mQ)
            V_curr[idx_q] = p_b_mQ * (d_b_mQ + delta_b_mQ) - penalties[idx_q] + V_next[idx_q]
            opt_bid_n[idx_q] = d_b_mQ
            # opt_ask_n[idx_q] remains inf
            
        elif Q == 0:
            # Only q=0, no fills allowed effectively as q is at both bounds? 
            # If Q=0, q=0. Bid hits -> q=1? No, "no bid fill at q=Q". So no bid.
            # "no ask fill at q=-Q". So no ask.
            V_curr[0] = -penalties[0] + V_next[0]
            
        opt_bid[n] = opt_bid_n
        opt_ask[n] = opt_ask_n
        V_next = V_curr

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """Avellaneda-Stoikov style analytical fallback."""
    sigma = params["sigma"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    dt = T / steps
    remaining = T - t
    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    c = skew_coeff / (2.0 * k)
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    bid_depths = np.clip(bid_depths, 1e-4, 10.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 10.0 / k)
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 12.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass

    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        # Ensure valid depths: replace values <= 0 or NaN with inf
        bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 1e-4), bid_depths, np.inf)
        ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 1e-4), ask_depths, np.inf)
        return (np.asarray(bid_depths, dtype=np.float64),
                np.asarray(ask_depths, dtype=np.float64))
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
