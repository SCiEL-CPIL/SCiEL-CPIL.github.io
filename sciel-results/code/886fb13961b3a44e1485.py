"""Optimal market making with inventory risk: Discrete HJB Solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin

# Precomputation storage
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: dict with 'V' (value function) and 'policy' (optimal depths)
# Since T=1, steps=1000, Q=30 are fixed in the prompt, we can key by regime params.
# However, the prompt says "Regimes ... all T=1, steps=1000, Q=30".
# We will precompute for the 3 known regimes at import time if time permits.
# If time is tight, we fall back to a fast heuristic.

# Grid for inventory: -Q to Q
# Grid for time: 0 to T with 'steps' intervals
# State: (n, q) where n is step index, q is inventory
# Action: (bid_depth, ask_depth)
# The continuous depth space is large. We need to discretize actions.
# However, the optimal depth in these models is often smooth.
# A better approach for discrete HJB with continuous actions is to use the fact
# that the value function is convex in inventory, allowing for root-finding or
# using a small set of candidate depths around the expected optimum.
# 
# Alternative: Since the action space is continuous, we can't just iterate over all.
# But we can use the structure. 
# Let's use a "Grid of Candidate Depths" approach for the HJB.
# The optimal depth is likely bounded. 
# For A=140, k=1.5, dt=0.001.
# P(fill) ~ A * exp(-k*d) * dt.
# If d=0, P ~ 0.14. If d=1, P ~ 0.03. If d=2, P ~ 0.007.
# Depths > 3 or 4 have negligible fill prob but high cost if filled? No, cost is inventory risk.
# Actually, deeper quotes earn more when filled (distance from mid).
# Objective: sum(depth) - risk.
# 
# Let's discretize the action space into a small set of "depth levels".
# Based on AS, optimal depth is around 1/k ~ 0.66.
# Let's try depths in range [0.01, 5.0].
# We can use a logarithmic or linear grid.
# To keep precomputation fast, we need a small action grid.
# Let's try 20x20 grid of (bid_depth, ask_depth).
# 400 actions per state.
# States: 1000 steps * 61 inventory levels = 61,000 states.
# 61,000 * 400 = 24.4 million operations per regime.
# 3 regimes = 73 million operations.
# In Python/NumPy, this might be too slow if done naively in a loop.
# We must vectorize the HJB backward induction.

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the discrete HJB for the given regime.
    Returns V (value function) and policy (optimal depths).
    V shape: (steps+1, 2*Q+1)
    Policy shape: (steps, 2*Q+1, 2) -> [bid_depth, ask_depth]
    """
    dt = T / steps
    q_min = -Q
    q_max = Q
    q_vals = np.arange(q_min, q_max + 1)
    num_q = len(q_vals)
    
    # Action Grid
    # We need to choose a reasonable range for depths.
    # Min depth: small positive to avoid immediate fill? Or 0?
    # Max depth: where prob is negligible or cost of waiting outweighs.
    # Let's use a grid of depths.
    # To save time, we can restrict the grid.
    # Optimal depth is roughly 1/k.
    # Let's use 10 bid depths and 10 ask depths.
    # Range: [0.01, 3.0/k] ? 
    # If k=1.5, 3/k = 2.
    # If k=0.75, 3/k = 4.
    # Let's use a dynamic range based on k.
    max_depth = 3.0 / k
    min_depth = 1e-4
    
    # Log-spaced grid might be better for depth
    # But linear is fine if range is small.
    # Let's use 15 points for bid and 15 for ask.
    bid_grid = np.linspace(min_depth, max_depth, 15)
    ask_grid = np.linspace(min_depth, max_depth, 15)
    
    # V_next: (num_q,)
    V_next = np.zeros(num_q)
    
    # V_curr: (num_q,)
    V_curr = np.zeros(num_q)
    
    # Policy storage: (steps, num_q, 2)
    policy = np.zeros((steps, num_q, 2))
    # V storage: (steps+1, num_q)
    V_all = np.zeros((steps + 1, num_q))
    
    # Precompute fill probabilities for all grid points
    # P_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
    # P_ask(d) = 1 - exp(-A * exp(-k*d) * dt)
    # Note: independent fills.
    
    # Vectorize over grids
    # bid_grid shape (15,), ask_grid shape (15,)
    # We need P for each combination?
    # In HJB, we iterate over actions.
    # Action is (bid_d, ask_d).
    # We can precompute P_bid for all bid_grid and P_ask for all ask_grid.
    
    P_bid = 1.0 - np.exp(-A * np.exp(-k * bid_grid) * dt)
    P_ask = 1.0 - np.exp(-A * np.exp(-k * ask_grid) * dt)
    
    # V_all[steps] = V_next (terminal value is 0)
    V_all[steps] = V_next
    
    for n in range(steps - 1, -1, -1):
        # t = n * dt
        # Inventory q corresponds to index i in q_vals
        
        # We need to maximize over (bid_idx, ask_idx) of:
        # Reward = 
        #   P_bid * (bid_depth + V_next[q+1]) +
        #   P_ask * (ask_depth + V_next[q-1]) +
        #   (1 - P_bid) * (1 - P_ask) * V_next[q]
        #   - (gamma / 2) * sigma^2 * q^2 * dt
        
        # Note: Inventory changes.
        # If bid filled: q -> q+1
        # If ask filled: q -> q-1
        # If both filled: q -> q (net 0) ? 
        # "A fill earns its depth and moves inventory one unit"
        # "bid hit ... ask likewise, independently"
        # Usually in these models, if both hit, inventory doesn't change (buy 1 sell 1).
        # Let's assume:
        # p_bb = P_bid * P_ask -> q stays q
        # p_ba = P_bid * (1-P_ask) -> q -> q+1
        # p_ab = (1-P_bid) * P_ask -> q -> q-1
        # p_aa = (1-P_bid) * (1-P_ask) -> q stays q
        
        # Reward = 
        #   p_bb * (bid_d + ask_d) +
        #   p_ba * bid_d +
        #   p_ab * ask_d
        #   + p_bb * V_next[q] + p_ba * V_next[q+1] + p_ab * V_next[q-1] + p_aa * V_next[q]
        #   - risk_cost
        
        # Risk cost is independent of action, so we can add it at the end or inside.
        risk_cost = (gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt
        
        # We need to handle boundaries q=Q and q=-Q.
        # "no bid fill at q = Q" -> if q==Q, P_bid = 0
        # "no ask fill at q = -Q" -> if q==-Q, P_ask = 0
        
        # V_next is shape (num_q,)
        # V_next_q_plus = V_next[q+1]
        # V_next_q_minus = V_next[q-1]
        
        V_next_q = V_next
        # q+1 index
        V_next_q_plus = np.roll(V_next, -1)
        V_next_q_plus[-1] = 0.0 # q=Q, cannot go to Q+1, effectively no state or value 0? 
                                # Actually, if q=Q, bid is disabled. So this term is 0.
        # q-1 index
        V_next_q_minus = np.roll(V_next, 1)
        V_next_q_minus[0] = 0.0 # q=-Q, cannot go to -Q-1
        
        # We want to compute Max over actions for each q.
        # This is a 2D maximization for each q.
        # We can vectorize over q (num_q) and actions (15x15).
        
        # P_bid depends on action bid_d.
        # P_ask depends on action ask_d.
        # V_next_q, V_next_q_plus, V_next_q_minus depend on q.
        
        # Let's create arrays for the maximization.
        # Shape: (num_q, 15, 15)
        
        # P_bid for each q? No, P_bid depends only on bid_d, but is disabled if q==Q.
        # P_ask depends only on ask_d, but is disabled if q==-Q.
        
        # Mask for q==Q (last index)
        mask_q_max = np.zeros(num_q, dtype=bool)
        mask_q_max[-1] = True
        mask_q_min = np.zeros(num_q, dtype=bool)
        mask_q_min[0] = True
        
        # Initialize best_value and best_action
        best_value = np.full(num_q, -np.inf)
        best_bid_idx = np.zeros(num_q, dtype=int)
        best_ask_idx = np.zeros(num_q, dtype=int)
        
        # Iterate over bid and ask grid points?
        # 15*15 = 225 iterations.
        # Inside each, vectorized over num_q (61).
        # 225 * 61 = 13,725 operations. Very fast.
        
        for bi, bd in enumerate(bid_grid):
            p_b = P_bid[bi]
            # If q==Q, p_b becomes 0
            # We can handle this by masking V_next_q_plus or just p_b.
            # Actually, if q==Q, the term involving V_next_q_plus should be 0.
            # And the reward from bid fill should be 0.
            
            for ai, ad in enumerate(ask_grid):
                p_a = P_ask[ai]
                
                # Probabilities
                # p_bb = p_b * p_a
                # p_ba = p_b * (1 - p_a)
                # p_ab = (1 - p_b) * p_a
                # p_aa = (1 - p_b) * (1 - p_a)
                
                # For q==Q: p_b = 0
                # For q==-Q: p_a = 0
                
                # Effective p_b for each q
                # p_b_eff = p_b * (1 - mask_q_max)
                # p_a_eff = p_a * (1 - mask_q_min)
                
                p_b_eff = p_b * (1.0 - mask_q_max.astype(float))
                p_a_eff = p_a * (1.0 - mask_q_min.astype(float))
                
                p_bb = p_b_eff * p_a_eff
                p_ba = p_b_eff * (1.0 - p_a_eff)
                p_ab = (1.0 - p_b_eff) * p_a_eff
                p_aa = (1.0 - p_b_eff) * (1.0 - p_a_eff)
                
                # Rewards
                # Bid reward: bd (if filled)
                # Ask reward: ad (if filled)
                # Note: if both filled, we get bd + ad.
                # My p_bb term above: p_bb * (bd + ad)
                # p_ba term: p_ba * bd
                # p_ab term: p_ab * ad
                
                reward = p_bb * (bd + ad) + p_ba * bd + p_ab * ad
                
                # Value terms
                val = p_bb * V_next_q + p_ba * V_next_q_plus + p_ab * V_next_q_minus + p_aa * V_next_q
                
                total = reward + val - risk_cost
                
                # Update best
                mask = total > best_value
                if np.any(mask):
                    best_value[mask] = total[mask]
                    best_bid_idx[mask] = bi
                    best_ask_idx[mask] = ai
        
        V_curr = best_value
        V_all[n] = V_curr
        policy[n] = np.stack([bid_grid[best_bid_idx], ask_grid[best_ask_idx]], axis=-1)
        
        V_next = V_curr
        
    return V_all, policy

# Cache for precomputed regimes
_REGIME_CACHE = {}

def _get_policy_for_params(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _REGIME_CACHE:
        # Check time
        if time.time() > _DEADLINE - 10:
            return None # Fallback
        
        # Precompute
        # This might take a few seconds.
        V, P = _precompute_regime(
            params['sigma'], params['A'], params['k'], 
            params['gamma'], params['T'], params['steps'], params['Q']
        )
        _REGIME_CACHE[key] = (V, P)
    return _REGIME_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Returns optimal depths from precomputed HJB policy.
    """
    # Get policy
    policy_data = _get_policy_for_params(params)
    
    if policy_data is None:
        # Fallback: AS heuristic
        sigma = params["sigma"]
        A = params["A"]
        k = params["k"]
        gamma = params["gamma"]
        T = params["T"]
        dt_rem = T - t
        R = gamma * (sigma ** 2) * inventory * dt_rem
        delta_star = 1.0 / k
        bid_depths = np.maximum(R + delta_star, 1e-6)
        ask_depths = np.maximum(-R + delta_star, 1e-6)
        return bid_depths, ask_depths

    V, policy = policy_data
    
    # Determine step index n
    # t = n * dt
    # dt = T / steps
    steps = params['steps']
    T = params['T']
    dt = T / steps
    
    # n = round(t / dt)
    # Clip to [0, steps-1]
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    # Inventory to index
    Q = params['Q']
    # inventory is in [-Q, Q]
    # Index = inventory + Q
    # Clip to [0, 2*Q]
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    # Get depths
    # policy[n] is shape (num_q, 2)
    # We need to index by q_idx
    depths = policy[n, q_idx, :]
    
    bid_depths = depths[:, 0]
    ask_depths = depths[:, 1]
    
    # Ensure validity
    # The grid used min_depth=1e-4, so these should be positive.
    # But just in case, clip to 1e-6.
    bid_depths = np.maximum(bid_depths, 1e-6)
    ask_depths = np.maximum(ask_depths, 1e-6)
    
    return bid_depths, ask_depths

# Precompute known regimes at import time to avoid latency in first call
# The prompt lists 3 regimes.
# We can try to precompute them.
_KNOWN_REGIMES = [
    (2, 140, 1.5, 0.05),
    (2, 140, 1.5, 0.5),
    (1.5, 50, 0.75, 0.2)
]

for sig, A, k, gam in _KNOWN_REGIMES:
    if time.time() > _DEADLINE - 30:
        break
    try:
        p = {'sigma': sig, 'A': A, 'k': k, 'gamma': gam, 'T': 1.0, 'steps': 1000, 'Q': 30}
        _get_policy_for_params(p)
    except Exception:
        pass
# EVOLVE-BLOCK-END
