"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np

_PARAMS_CACHE = {}

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Value function: V_next[q_idx] = optimal expected objective from next step with inventory q
    # V at terminal = 0 (no terminal charge)
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Depth grid: use a logarithmic grid from small to large depths
    min_depth = 1e-4
    max_depth = 10.0 / k
    n_depths = 100 # Keep 100 for speed, vectorization helps
    depth_grid = np.geomspace(min_depth, max_depth, n_depths)
    
    # Pre-compute fill probabilities for each depth
    lambdas = A * np.exp(-k * depth_grid) * dt
    fill_probs = -np.expm1(-lambdas)
    
    # Inventory risk cost coefficient
    risk_coeff = 0.5 * gamma * sigma**2 * dt
    risk_costs = risk_coeff * q_vals**2
    
    # Pre-allocate shapes for vectorized operations
    # PB: (N_d, 1, 1)
    # PA: (1, N_d, 1)
    # B: (N_d, 1, 1)
    # Aa: (1, N_d, 1)
    PB = fill_probs[:, np.newaxis, np.newaxis]
    PA = fill_probs[np.newaxis, :, np.newaxis]
    B = depth_grid[:, np.newaxis, np.newaxis]
    Aa = depth_grid[np.newaxis, :, np.newaxis]
    
    # V_next is shape (N_inv,)
    # We need V_q, V_q1, V_qm1 for each q_idx
    # V_q: (1, 1, N_inv)
    # V_q1: (1, 1, N_inv) -> V_next[q+1], with boundary handling
    # V_qm1: (1, 1, N_inv) -> V_next[q-1], with boundary handling
    
    for n in range(steps - 1, -1, -1):
        V_q = V_next[np.newaxis, np.newaxis, :]
        
        # V_q1: inventory q+1. If q=Q, this is invalid (no bid).
        # We can set V_q1 to 0 where q=Q, but the term multiplying it should be 0 anyway if we mask.
        # However, the term is PB * (1-PA) * (B + V_q1). If q=Q, we should not allow bid.
        # Easier: Compute E for all, then mask out invalid depths.
        # Or: Set V_q1 such that the term vanishes? No, B is still there.
        # Best: Compute E, then set E to -inf where bid is not allowed (q=Q) or ask is not allowed (q=-Q).
        
        # V_next indices
        # q_idx goes from 0 to N_inv-1
        # q = q_idx - Q
        # q+1 corresponds to q_idx+1. If q_idx+1 == N_inv, it's out of bounds.
        # q-1 corresponds to q_idx-1. If q_idx-1 == -1, it's out of bounds.
        
        V_q1 = np.zeros(n_inv)
        V_qm1 = np.zeros(n_inv)
        
        # Valid q for bid: q < Q => q_idx < 2Q => q_idx+1 < 2Q+1 = N_inv
        # So for q_idx in [0, 2Q-1], V_q1 = V_next[q_idx+1]
        # For q_idx = 2Q (q=Q), V_q1 is not used if we mask.
        V_q1[:2*Q] = V_next[1:2*Q+1]
        
        # Valid q for ask: q > -Q => q_idx > 0 => q_idx-1 >= 0
        # So for q_idx in [1, 2Q], V_qm1 = V_next[q_idx-1]
        # For q_idx = 0 (q=-Q), V_qm1 is not used if we mask.
        V_qm1[1:] = V_next[:2*Q]
        
        V_q1_3d = V_q1[np.newaxis, np.newaxis, :]
        V_qm1_3d = V_qm1[np.newaxis, np.newaxis, :]
        
        # Compute E for all (b, a, q)
        # E = -risk_cost + PB*PA*(B+Aa+V_q) + PB*(1-PA)*(B+V_q1) + (1-PB)*PA*(Aa+V_qm1) + (1-PB)*(1-PA)*V_q
        
        # risk_costs shape (N_inv,) -> (1, 1, N_inv)
        Risk_3d = risk_costs[np.newaxis, np.newaxis, :]
        
        term1 = PB * PA * (B + Aa + V_q)
        term2 = PB * (1 - PA) * (B + V_q1_3d)
        term3 = (1 - PB) * PA * (Aa + V_qm1_3d)
        term4 = (1 - PB) * (1 - PA) * V_q
        
        E = -Risk_3d + term1 + term2 + term3 + term4
        
        # Mask invalid quotes
        # If q = Q (q_idx = 2Q), bid is invalid. Set E[:, :, 2Q] to -inf for all b?
        # Actually, if bid is invalid, we should only consider a=inf? No, we just don't quote bid.
        # The model says "no bid fill at q=Q". This implies we can still quote ask.
        # But our depth grid is for *finite* depths.
        # If we must not quote bid, the optimal strategy is to quote bid at inf (prob 0) and optimize ask.
        # In our 2D grid, we don't have inf.
        # So for q=Q, we should restrict the search to the case where bid prob is 0?
        # Or simply set the bid depth to the largest in the grid (approx inf) and optimize ask?
        # Better: For q=Q, we know bid is off. We can solve the 1D problem for ask.
        # For q=-Q, we know ask is off. We can solve the 1D problem for bid.
        
        # Let's handle boundaries separately to be precise and fast.
        
        # 1. Interior points: 1 <= q_idx <= 2Q-1
        # 2. Boundary q_idx = 0 (q=-Q): Ask only
        # 3. Boundary q_idx = 2Q (q=Q): Bid only
        
        V_curr = np.zeros(n_inv)
        best_bi = np.zeros(n_inv, dtype=int)
        best_ai = np.zeros(n_inv, dtype=int)
        
        # Interior
        if n_inv > 2:
            E_int = E[:, :, 1:2*Q]
            max_idx_int = np.argmax(E_int, axis=(0, 1))
            # max_idx_int is shape (N_inv-2,)
            best_bi[1:2*Q] = max_idx_int // n_depths
            best_ai[1:2*Q] = max_idx_int % n_depths
            V_curr[1:2*Q] = E_int.reshape(-1, n_depths * n_depths)[np.arange(2*Q-2), max_idx_int]
            # Wait, E_int is (N_d, N_d, N_inv-2). argmax over (0,1) gives index in flattened (0,1) dims?
            # No, np.argmax(arr, axis=(0,1)) returns indices in the flattened 0,1 dimensions?
            # Actually, it returns the index of the max element in the flattened array of those axes?
            # Documentation: "If axis is a tuple, the maximum is taken over the flattened subarray."
            # So max_idx_int is a flat index into (N_d * N_d).
            # So bi = idx // N_d, ai = idx % N_d.
            # And the value is E_int.ravel()[max_idx_int]?
            # Let's verify.
            # E_int shape: (100, 100, 59).
            # argmax(axis=(0,1)) returns array of shape (59,).
            # Each value is index in 0..9999.
            # Value = E_int.reshape(-1, 59)[max_idx_int, np.arange(59)]?
            # E_int.reshape(-1, 59) shape (10000, 59).
            # Yes.
            
            E_int_flat = E_int.reshape(-1, 2*Q-2)
            V_curr[1:2*Q] = E_int_flat[max_idx_int, np.arange(2*Q-2)]
            
        # Boundary q_idx = 0 (q = -Q): Ask only (bid disabled)
        # Optimize a.
        # E(a) = -risk_cost + (1-pa)*V_q + pa*(a + V_qm1)
        # V_q = V_next[0]
        # V_qm1 is invalid, but since ask moves to q-1, and q=-Q, ask is disabled?
        # "no ask fill at q = -Q".
        # So if q=-Q, we cannot quote ask?
        # "no bid fill at q = Q, no ask fill at q = -Q".
        # This means if q=-Q, we can quote bid (which moves to -Q+1), but NOT ask.
        # So for q=-Q, only Bid is active.
        
        # My previous logic was:
        # q=Q: no bid. (Ask active)
        # q=-Q: no ask. (Bid active)
        
        # Let's re-read carefully.
        # "no bid fill at q = Q" -> If you are at max inventory, you don't buy more. So Bid is OFF.
        # "no ask fill at q = -Q" -> If you are at min inventory (max short), you don't sell more. So Ask is OFF.
        
        # So:
        # q = Q: Bid OFF, Ask ON.
        # q = -Q: Ask OFF, Bid ON.
        
        # Correcting the boundary logic in the vectorized part:
        
        # q_idx = 0 (q = -Q): Ask OFF, Bid ON.
        # Optimize b.
        # E(b) = -risk_cost + (1-pb)*V_q + pb*(b + V_q1)
        # V_q = V_next[0]
        # V_q1 = V_next[1]
        
        V_q_0 = V_next[0]
        V_q1_0 = V_next[1]
        risk_0 = risk_costs[0]
        
        vals_bid_0 = -risk_0 + (1 - fill_probs) * V_q_0 + fill_probs * (depth_grid + V_q1_0)
        best_bi_0 = np.argmax(vals_bid_0)
        best_bi[0] = best_bi_0
        best_ai[0] = 0 # dummy
        V_curr[0] = vals_bid_0[best_bi_0]
        
        # q_idx = 2Q (q = Q): Bid OFF, Ask ON.
        # Optimize a.
        # E(a) = -risk_cost + (1-pa)*V_q + pa*(a + V_qm1)
        # V_q = V_next[2Q]
        # V_qm1 = V_next[2Q-1]
        
        V_q_last = V_next[2*Q]
        V_qm1_last = V_next[2*Q-1]
        risk_last = risk_costs[2*Q]
        
        vals_ask_last = -risk_last + (1 - fill_probs) * V_q_last + fill_probs * (depth_grid + V_qm1_last)
        best_ai_last = np.argmax(vals_ask_last)
        best_bi[2*Q] = 0 # dummy
        best_ai[2*Q] = best_ai_last
        V_curr[2*Q] = vals_ask_last[best_ai_last]
        
        # Update tables
        bid_table[n, :] = depth_grid[best_bi]
        ask_table[n, :] = depth_grid[best_ai]
        
        # Set inf for disabled sides
        bid_table[n, 2*Q] = np.inf # q=Q, bid off
        ask_table[n, 0] = np.inf   # q=-Q, ask off
        
        V_next = V_curr
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _PARAMS_CACHE:
        _PARAMS_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], 
            params['gamma'], params['T'], params['steps'], params['Q']
        )
    return _PARAMS_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote optimal bid/ask depths based on pre-computed policy table.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two arrays of shape (4000,)
    """
    bid_table, ask_table = _get_policy_table(params)
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step
    T = params['T']
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to table indices
    inv_idx = np.clip(inventory, -Q, Q) + Q
    inv_idx = inv_idx.astype(int)
    
    # Get depths from table
    bid_depths = bid_table[n, inv_idx].copy()
    ask_depths = ask_table[n, inv_idx].copy()
    
    # Ensure no bid at q=Q and no ask at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure depths are valid (positive, not too small, not NaN)
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, 1e-6), bid_depths)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, 1e-6), ask_depths)
    
    return bid_depths, ask_depths
