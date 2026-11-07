"""Market-making with inventory risk: DP over inventory grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache DP value tables per parameter regime.
_V_TABLES = {}

# Global deadline guard (300 s hard limit).
_T0 = time.time()
_DEADLINE = 290.0  # seconds of wall clock budget for the whole run.


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _clipped_depths(q, shift, k):
    """Return (bid_depth, ask_depth) arrays clipped to [0, 3/k]."""
    inv_k = 1.0 / k
    base = inv_k + q * shift
    bid = np.clip(base, 0.0, 3.0 * inv_k)
    ask = np.clip(inv_k - q * shift, 0.0, 3.0 * inv_k)
    return bid, ask


def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """Solve the discrete DP for V(q, n_remaining) and return a dict of tables."""
    if _remaining_budget() < 60.0:
        return None  # not enough time

    # Precompute probabilities for each possible depth
    # The depth is determined by the shift, which depends on q and n_remaining
    # But in the optimal policy, the depth is a function of (q, n_remaining)
    # Let's use a simpler approach: assume the optimal depth is of the form
    #   d_bid = max(0, min(3/k, c + alpha * q))
    #   d_ask = max(0, min(3/k, c - alpha * q))
    # where c and alpha are to be determined.
    
    # Actually, let's do a proper DP.
    # V(q, n) = max over (d_bid, d_ask) of:
    #   - (gamma/2) * sigma^2 * q^2 * dt
    #   + p_bid * d_bid + p_ask * d_ask
    #   + E[V(q', n-1)]
    # where q' = q - 1 if bid fill, q + 1 if ask fill, else q
    # p_bid = 1 - exp(-A * exp(-k * d_bid) * dt)
    # p_ask = 1 - exp(-A * exp(-k * d_ask) * dt)
    # q' is random: with prob p_bid*(1-p_ask): q-1, with prob (1-p_bid)*p_ask: q+1,
    # with prob p_bid*p_ask: q (both fills cancel? No, both fills happen: q-1+1=q),
    # with prob (1-p_bid)*(1-p_ask): q
    
    # Wait, the problem says "A fill earns its depth and moves inventory one unit"
    # and "bid hit with probability ..., ask likewise, independently"
    # So both can fill in the same step. If bid fills, inventory decreases by 1.
    # If ask fills, inventory increases by 1.
    # So q' = q - I_bid + I_ask, where I_bid ~ Bernoulli(p_bid), I_ask ~ Bernoulli(p_ask)
    
    # The reward is d_bid * I_bid + d_ask * I_ask - (gamma/2) sigma^2 q^2 dt
    # Expected reward: p_bid * d_bid + p_ask * d_ask - (gamma/2) sigma^2 q^2 dt
    
    # V(q, n) = max_{d_bid, d_ask} [ p_bid(d_bid)*d_bid + p_ask(d_ask)*d_ask - (gamma/2)*sigma^2*q^2*dt
    #   + (1-p_bid)*(1-p_ask)*V(q, n-1) + p_bid*(1-p_ask)*V(q-1, n-1)
    #   + (1-p_bid)*p_ask*V(q+1, n-1) + p_bid*p_ask*V(q, n-1) ]
    # = max [ p_bid*d_bid + p_ask*d_ask - (gamma/2)*sigma^2*q^2*dt
    #   + V(q, n-1) + p_bid*(1-p_ask)*V(q-1, n-1) - p_bid*(1-p_ask)*V(q, n-1)
    #   + (1-p_bid)*p_ask*V(q+1, n-1) - (1-p_bid)*p_ask*V(q, n-1) ]
    
    # This is complex. Let's simplify by noting that the optimal d_bid and d_ask
    # can be found by considering the marginal benefit.
    
    # For a given q and n, let's define:
    #   f_bid(d) = p_bid(d)*d + p_bid(d)*(1-p_ask)*V(q-1, n-1) - p_bid(d)*(1-p_ask)*V(q, n-1)
    #   = p_bid(d) * [ d + (1-p_ask)*(V(q-1, n-1) - V(q, n-1)) ]
    # Similarly for ask.
    
    # But p_ask depends on d_ask, so it's coupled.
    
    # Let's just do a grid search over possible depths.
    # Since k is given, the depth range is [0, 3/k].
    # Let's discretize depth into, say, 100 points.
    
    n_depths = 50
    depths = np.linspace(0.0, 3.0/k, n_depths)
    dt = T / steps
    
    # V[q_offset, n] where q ranges from -Q to Q, so offset by Q
    # V shape: (2*Q+1, steps+1)
    V = np.zeros((2*Q+1, steps+1))
    
    # V(q, 0) = 0 for all q (no terminal charge)
    
    for n in range(1, steps+1):
        for q_idx in range(2*Q+1):
            q = q_idx - Q
            best_val = -np.inf
            best_bid_idx = 0
            best_ask_idx = 0
            
            for bid_idx in range(n_depths):
                d_bid = depths[bid_idx]
                p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
                
                # No bid fill at q = Q
                if q == Q:
                    p_bid = 0.0
                
                for ask_idx in range(n_depths):
                    d_ask = depths[ask_idx]
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
                    
                    # No ask fill at q = -Q
                    if q == -Q:
                        p_ask = 0.0
                    
                    # Compute expected next value
                    # q' = q - I_bid + I_ask
                    # E[V(q', n-1)] = (1-p_bid)*(1-p_ask)*V(q, n-1) 
                    #   + p_bid*(1-p_ask)*V(q-1, n-1) 
                    #   + (1-p_bid)*p_ask*V(q+1, n-1) 
                    #   + p_bid*p_ask*V(q, n-1)
                    
                    Vq = V[q_idx, n-1]
                    
                    Vq_minus = V[q_idx-1, n-1] if q > -Q else 0.0
                    Vq_plus = V[q_idx+1, n-1] if q < Q else 0.0
                    
                    # If q == -Q, q-1 is out of bounds, but p_bid should be 0 in that case? No, 
                    # at q=-Q, bid fill is allowed (it would move to q=-Q+1). 
                    # At q=Q, ask fill is allowed (it would move to q=Q-1).
                    # The constraint is: no bid fill at q=Q (can't go to Q+1), no ask fill at q=-Q (can't go to -Q-1).
                    
                    # So if q == Q: p_bid = 0 (no bid fill)
                    # if q == -Q: p_ask = 0 (no ask fill)
                    
                    # V(q-1, n-1) is valid if q-1 >= -Q, i.e., q >= -Q+1
                    # V(q+1, n-1) is valid if q+1 <= Q, i.e., q <= Q-1
                    
                    if q > -Q:
                        Vq_minus = V[q_idx-1, n-1]
                    else:
                        Vq_minus = 0.0  # won't be used if p_bid*(1-p_ask) == 0
                    
                    if q < Q:
                        Vq_plus = V[q_idx+1, n-1]
                    else:
                        Vq_plus = 0.0  # won't be used if (1-p_bid)*p_ask == 0
                    
                    E_next = ((1-p_bid)*(1-p_ask) + p_bid*p_ask) * Vq \
                           + p_bid*(1-p_ask) * Vq_minus \
                           + (1-p_bid)*p_ask * Vq_plus
                    
                    reward = p_bid * d_bid + p_ask * d_ask - 0.5 * gamma * sigma**2 * q**2 * dt
                    
                    val = reward + E_next
                    
                    if val > best_val:
                        best_val = val
                        best_bid_idx = bid_idx
                        best_ask_idx = ask_idx
            
            V[q_idx, n] = best_val
    
    # Now extract the optimal depths
    bid_depths_table = np.zeros((2*Q+1, steps+1))
    ask_depths_table = np.zeros((2*Q+1, steps+1))
    
    for n in range(1, steps+1):
        for q_idx in range(2*Q+1):
            q = q_idx - Q
            best_val = -np.inf
            best_bid_idx = 0
            best_ask_idx = 0
            
            for bid_idx in range(n_depths):
                d_bid = depths[bid_idx]
                p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
                
                if q == Q:
                    p_bid = 0.0
                
                for ask_idx in range(n_depths):
                    d_ask = depths[ask_idx]
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
                    
                    if q == -Q:
                        p_ask = 0.0
                    
                    Vq = V[q_idx, n-1]
                    
                    if q > -Q:
                        Vq_minus = V[q_idx-1, n-1]
                    else:
                        Vq_minus = 0.0
                    
                    if q < Q:
                        Vq_plus = V[q_idx+1, n-1]
                    else:
                        Vq_plus = 0.0
                    
                    E_next = ((1-p_bid)*(1-p_ask) + p_bid*p_ask) * Vq \
                           + p_bid*(1-p_ask) * Vq_minus \
                           + (1-p_bid)*p_ask * Vq_plus
                    
                    reward = p_bid * d_bid + p_ask * d_ask - 0.5 * gamma * sigma**2 * q**2 * dt
                    
                    val = reward + E_next
                    
                    if val > best_val:
                        best_val = val
                        best_bid_idx = bid_idx
                        best_ask_idx = ask_idx
            
            bid_depths_table[q_idx, n] = depths[best_bid_idx]
            ask_depths_table[q_idx, n] = depths[best_ask_idx]
    
    return {
        'bid_depths': bid_depths_table,
        'ask_depths': ask_depths_table,
        'depths': depths,
    }


def _get_table(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _V_TABLES:
        if _remaining_budget() < 120.0:
            return None
        result = _solve_dp(sigma, A, k, gamma, T, steps, Q)
        _V_TABLES[key] = result
    return _V_TABLES[key]


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n = int(round(t / (T / steps)))
    n = min(n, steps)
    
    table = _get_table(sigma, A, k, gamma, T, steps, Q)
    
    if table is None:
        # Fallback: simple heuristic
        inv_k = 1.0 / k
        bid = np.full(len(inventory), inv_k)
        ask = np.full(len(inventory), inv_k)
        return bid, ask
    
    bid_table = table['bid_depths']
    ask_table = table['ask_depths']
    
    # inventory is in [-Q, Q], so index = inventory + Q
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    bid_depths = bid_table[q_idx, n]
    ask_depths = ask_table[q_idx, n]
    
    # Clip to valid range
    max_depth = 3.0 / k
    bid_depths = np.clip(bid_depths, 0.0, max_depth)
    ask_depths = np.clip(ask_depths, 0.0, max_depth)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
