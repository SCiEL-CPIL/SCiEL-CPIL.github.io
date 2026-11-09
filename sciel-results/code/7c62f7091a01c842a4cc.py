"""Optimal market making with inventory risk - DP-based policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 250.0
_CACHE = {}


def _check_deadline():
    return time.time() < _DEADLINE


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal depth policy via dynamic programming on the inventory grid.
    
    Returns arrays:
      bid_depths[step, q_idx] : optimal bid depth for step, inventory index
      ask_depths[step, q_idx] : optimal ask depth for step, inventory index
    
    q_idx maps to inventory values -Q..Q.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Number of depth grid points
    n_d = 160
    # Depth grid: from 0 to some max
    # Maximum reasonable depth: when prob ~ 1, A*exp(-k*d)*dt ~ 1 => d ~ log(A*dt)/k
    # But we need a range. Let's use up to ~ 3/k to be safe, plus some margin
    max_depth = 4.0 / k
    d_grid = np.linspace(0.0, max_depth, n_d)
    
    # Precompute fill probabilities for each depth
    # prob_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = A * np.exp(-k * d_grid) * dt
    prob_fill = 1.0 - np.exp(-exp_term)  # shape (n_d,)
    
    # Value function: V[i, q] = value from step i onwards given inventory q
    # V[0, q] = 0 (terminal, no charge)
    V = np.zeros((N + 1, n_q))
    
    # Optimal depths
    opt_bid = np.full((N, n_q), np.inf, dtype=np.float64)
    opt_ask = np.full((N, n_q), np.inf, dtype=np.float64)
    
    # Work backwards
    for i in range(N - 1, -1, -1):
        if not _check_deadline():
            # Deadline exceeded - use heuristic depths
            heuristic = np.full((N, n_q), 1.0 / k)
            _CACHE[key] = (heuristic, heuristic.copy())
            return _CACHE[key]
        
        # Inventory cost at this step: (gamma/2) * sigma^2 * q^2 * dt
        inv_cost = (gamma / 2.0) * sigma * sigma * q_values * q_values * dt  # shape (n_q,)
        
        # For each inventory level, we need to choose bid_depth and ask_depth
        # to maximize: -inv_cost + expected_value_next
        # 
        # Next inventory:
        #   bid fill: q -> q - 1 (if q < Q), earn bid_depth
        #   ask fill: q -> q + 1 (if q > -Q), earn ask_depth
        #   both: q -> q (if both possible), earn bid_depth + ask_depth
        #   neither: q -> q, earn 0
        #
        # P(bid fill) = prob_fill(bid_depth), P(ask fill) = prob_fill(ask_depth)
        # Independent.
        #
        # Expected value:
        #   V_next[q-1] * p_b * (1-p_a) + V_next[q+1] * (1-p_b) * p_a 
        #   + V_next[q] * (p_b*p_a + (1-p_b)*(1-p_a))
        #   + bid_depth * p_b * (1-p_a) + ask_depth * (1-p_b) * p_a
        #   + (bid_depth + ask_depth) * p_b * p_a
        #
        # Wait, let me reconsider. The problem says "A fill earns its depth and moves inventory one unit"
        # and "bid hit with probability ... ask likewise, independently"
        #
        # So:
        #   bid fill: prob p_b, earn bid_depth, inventory q -> q-1
        #   ask fill: prob p_a, earn ask_depth, inventory q -> q+1
        #   Both independent.
        #
        # Cases:
        #   bid only (not ask): prob p_b*(1-p_a), earn bid_depth, q -> q-1
        #   ask only (not bid): prob (1-p_b)*p_a, earn ask_depth, q -> q+1
        #   both: prob p_b*p_a, earn bid_depth+ask_depth, q -> q-1+1 = q
        #   neither: prob (1-p_b)*(1-p_a), earn 0, q -> q
        #
        # But with boundary constraints: no bid fill at q=Q, no ask fill at q=-Q.
        
        # For efficiency, we'll use a grid search over depths.
        # For each q, we want to maximize over (d_b, d_a):
        #   -inv_cost[q] + 
        #   V[i+1, q-1] * p_b * (1-p_a)  [if q < Q]
        #   + V[i+1, q+1] * (1-p_b) * p_a  [if q > -Q]
        #   + V[i+1, q] * (p_b*p_a + (1-p_b)*(1-p_a))
        #   + d_b * p_b * (1-p_a) + d_a * (1-p_b) * p_a + (d_b + d_a) * p_b * p_a
        #
        # This is a 2D optimization. We can do it on a grid.
        
        V_next = V[i + 1]  # shape (n_q,)
        
        # We'll compute the optimal depths for each q
        # For speed, vectorize over q where possible.
        
        # Let's think about this differently. The objective for given q is:
        # f(d_b, d_a) = -inv_cost[q] 
        #   + V_next[q-1] * p_b * (1-p_a)   [if q < Q, else 0]
        #   + V_next[q+1] * (1-p_b) * p_a   [if q > -Q, else 0]
        #   + V_next[q] * (p_b*p_a + (1-p_b)*(1-p_a))
        #   + d_b * p_b * (1-p_a) + d_a * (1-p_b) * p_a + (d_b + d_a) * p_b * p_a
        
        # Let's expand:
        # = -inv_cost[q] + V_next[q] * (1 - (1-p_b)*(1-p_a) - p_b*p_a + p_b*p_a)
        #   Wait, let me just compute directly.
        
        # Actually, let me define:
        # p_b = prob_fill[d_b_idx], p_a = prob_fill[d_a_idx]
        
        # For each q, we need to find argmax over (d_b_idx, d_a_idx)
        
        # To speed this up, note that the problem is separable in a useful way.
        # Let's just do a grid search. With n_d=160, that's 160*160=25600 combos per q.
        # With 61 q values and 1000 steps, that's 1000*61*25600 = 1.56 billion - too slow.
        
        # We need a smarter approach. Let's use a smaller grid or optimize analytically.
        
        # Alternative: use a coarser grid for depth (e.g., 40 points) and optimize.
        # Or, use the fact that the optimal depth is likely near 1/k.
        
        # Let me use n_d = 60 for speed.
        n_d_eff = 60
        d_grid_eff = np.linspace(0.0, max_depth, n_d_eff)
        exp_term_eff = A * np.exp(-k * d_grid_eff) * dt
        prob_fill_eff = 1.0 - np.exp(-exp_term_eff)
        
        # For each q, compute optimal (d_b, d_a)
        # We'll do this with a nested loop over q, but vectorize over depth grid.
        
        for qi in range(n_q):
            q = q_values[qi]
            
            # Boundary conditions
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if not can_bid and not can_ask:
                # Can't trade at all
                opt_bid[i, qi] = np.inf
                opt_ask[i, qi] = np.inf
                V[i, qi] = -inv_cost[qi]
                continue
            
            # V_next values
            Vn_q = V_next[qi]
            Vn_qm1 = V_next[qi - 1] if can_bid else 0.0
            Vn_qp1 = V_next[qi + 1] if can_ask else 0.0
            
            # We need to maximize over (d_b_idx, d_a_idx):
            # -inv_cost[qi]
            #   + Vn_qm1 * p_b * (1-p_a) * can_bid
            #   + Vn_qp1 * (1-p_b) * p_a * can_ask
            #   + Vn_q * (p_b*p_a + (1-p_b)*(1-p_a))
            #   + d_b * p_b * (1-p_a) * can_bid + d_a * (1-p_b) * p_a * can_ask
            #   + (d_b + d_a) * p_b * p_a * (can_bid * can_ask)
            
            # Let's vectorize over d_b and d_a
            # p_b shape: (n_d_eff,), p_a shape: (n_d_eff,)
            # We want a 2D grid of (d_b, d_a)
            
            p_b = prob_fill_eff  # (n_d_eff,)
            d_b = d_grid_eff     # (n_d_eff,)
            
            # For each d_b, compute the optimal d_a
            # The objective as a function of d_a (with d_b fixed) is:
            # -inv_cost[qi]
            #   + Vn_qm1 * p_b * (1-p_a) * can_bid
            #   + Vn_qp1 * (1-p_b) * p_a * can_ask
            #   + Vn_q * (p_b*p_a + (1-p_b)*(1-p_a))
            #   + d_b * p_b * (1-p_a) * can_bid + d_a * (1-p_b) * p_a * can_ask
            #   + (d_b + d_a) * p_b * p_a * (can_bid * can_ask)
            
            # Let me expand in terms of p_a:
            # = -inv_cost[qi]
            #   + p_a * [Vn_qp1 * (1-p_b) * can_ask + Vn_q * p_b + d_a * (1-p_b) * p_a * can_ask + d_a * p_b * (can_bid * can_ask)]
            #   + (1-p_a) * [Vn_qm1 * p_b * can_bid + Vn_q * (1-p_b) + d_b * p_b * can_bid]
            #   + d_b * p_b * p_a * (can_bid * can_ask)  # wait, this is getting complicated
            
            # Let me just compute the full 2D grid for each q.
            # With n_d_eff=60, that's 60*60=3600 evaluations per q.
            # 61 q * 1000 steps * 3600 = 219.6M operations. Might be okay in ~10-20s.
            
            # Actually, let's use a smarter approach: for each d_b, find optimal d_a analytically or via 1D search.
            
            # The objective as a function of (p_a, d_a) with d_b fixed:
            # Let c1 = Vn_qm1 * p_b * can_bid + d_b * p_b * can_bid
            # Let c2 = Vn_qp1 * (1-p_b) * can_ask
            # Let c3 = Vn_q
            # Let c4 = d_b * p_b * (can_bid * can_ask)
            
            # f(p_a, d_a) = -inv_cost[qi]
            #   + c1 * (1-p_a)
            #   + c2 * p_a
            #   + c3 * (p_b*p_a + (1-p_b)*(1-p_a))
            #   + d_a * (1-p_b) * p_a * can_ask
            #   + d_a * p_b * p_a * (can_bid * can_ask)
            
            # = -inv_cost[qi] + c1 - c1*p_a + c2*p_a + c3*p_b*p_a + c3*(1-p_b)*(1-p_a) + d_a*p_a*((1-p_b)*can_ask + p_b*(can_bid*can_ask))
            
            # = -inv_cost[qi] + c1 + c3*(1-p_b)
            #   + p_a * [-c1 + c2 + c3*p_b - c3*(1-p_b) + d_a*((1-p_b)*can_ask + p_b*(can_bid*can_ask))]
            
            # = const + p_a * [ -c1 + c2 + c3*p_b - c3 + c3*p_b + d_a*((1-p_b)*can_ask + p_b*(can_bid*can_ask))]
            
            # = const + p_a * [ -c1 + c2 + 2*c3*p_b - c3 + d_a*((1-p_b)*can_ask + p_b*(can_bid*can_ask))]
            
            # So for fixed d_b (hence fixed p_b), the optimal p_a is:
            #   p_a = 0 if coefficient < 0
            #   p_a = 1 if coefficient > 0
            #   (if coefficient = 0, any p_a works)
            
            # But p_a is tied to d_a through p_a = prob_fill(d_a).
            # So we need to find d_a that maximizes:
            #   const + prob_fill(d_a) * [ -c1 + c2 + 2*c3*p_b - c3 + d_a*((1-p_b)*can_ask + p_b*(can_bid*can_ask))]
            
            # Let coef = -c1 + c2 + 2*c3*p_b - c3
            # Let slope = (1-p_b)*can_ask + p_b*(can_bid*can_ask)
            
            # We want to maximize: prob_fill(d_a) * (coef + slope * d_a)
            
            # This is a 1D optimization over d_a in [0, max_depth].
            # prob_fill(d_a) = 1 - exp(-A*exp(-k*d_a)*dt)
            
            # For each d_b, we can find optimal d_a by evaluating on the grid.
            
            c1 = Vn_qm1 * p_b * can_bid + d_b * p_b * can_bid
            c2 = Vn_qp1 * (1-p_b) * can_ask
            c3 = Vn_q
            c4 = d_b * p_b * (can_bid * can_ask)
            
            # const = -inv_cost[qi] + c1 + c3*(1-p_b)
            const = -inv_cost[qi] + c1 + c3 * (1 - p_b)
            
            # coef = -c1 + c2 + 2*c3*p_b - c3
            coef = -c1 + c2 + 2 * c3 * p_b - c3
            
            # slope = (1-p_b)*can_ask + p_b*(can_bid*can_ask)
            slope = (1 - p_b) * can_ask + p_b * (can_bid * can_ask)
            
            # Objective for d_a: prob_fill(d_a) * (coef + slope * d_a)
            # We want to maximize this over d_a in d_grid_eff
            
            obj_a = prob_fill_eff * (coef + slope * d_grid_eff)
            
            # But wait, this is only valid if the expression is correct.
            # Let me just compute the full objective for each (d_b, d_a) pair to be safe.
            # Given time constraints, let's do a simpler approach.
            
            # Actually, let me just do the full 2D grid search but with a smaller grid.
            n_d_search = 40
            d_search = np.linspace(0.0, max_depth, n_d_search)
            p_search = 1.0 - np.exp(-A * np.exp(-k * d_search) * dt)
            
            # 2D grid
            D_b, D_a = np.meshgrid(d_search, d_search, indexing='ij')
            P_b, P_a = np.meshgrid(p_search, p_search, indexing='ij')
            
            # Objective matrix
            obj = np.zeros_like(D_b)
            
            # Term 1: -inv_cost
            obj += -inv_cost[qi]
            
            # Term 2: Vn_qm1 * p_b * (1-p_a) * can_bid
            if can_bid:
                obj += Vn_qm1 * P_b * (1 - P_a)
            
            # Term 3: Vn_qp1 * (1-p_b) * p_a * can_ask
            if can_ask:
                obj += Vn_qp1 * (1 - P_b) * P_a
            
            # Term 4: Vn_q * (p_b*p_a + (1-p_b)*(1-p_a))
            obj += Vn_q * (P_b * P_a + (1 - P_b) * (1 - P_a))
            
            # Term 5: d_b * p_b * (1-p_a) * can_bid
            if can_bid:
                obj += D_b * P_b * (1 - P_a)
            
            # Term 6: d_a * (1-p_b) * p_a * can_ask
            if can_ask:
                obj += D_a * (1 - P_b) * P_a
            
            # Term 7: (d_b + d_a) * p_b * p_a * (can_bid * can_ask)
            if can_bid and can_ask:
                obj += (D_b + D_a) * P_b * P_a
            
            # Find max
            max_idx = np.argmax(obj)
            db_idx = max_idx // n_d_search
            da_idx = max_idx % n_d_search
            
            if can_bid:
                opt_bid[i, qi] = d_search[db_idx]
            else:
                opt_bid[i, qi] = np.inf
            
            if can_ask:
                opt_ask[i, qi] = d_search[da_idx]
            else:
                opt_ask[i, qi] = np.inf
            
            V[i, qi] = obj[max_idx]
    
    _CACHE[key] = (opt_bid, opt_ask)
    return _CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths for each path.
    
    Returns (bid_depths, ask_depths) each of shape (4000,).
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _CACHE:
        if not _check_deadline():
            # No time to solve, use heuristic
            depth = np.full(len(inventory), 1.0 / k)
            return depth, depth.copy()
        _solve_regime(sigma, A, k, gamma, T, steps, Q)
    
    opt_bid, opt_ask = _CACHE[key]
    
    # Determine current step
    dt = T / steps
    step = int(round(t / dt))
    step = min(step, steps - 1)
    
    # Inventory to index: q_idx = q + Q
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[step, q_idx]
    ask_depths = opt_ask[step, q_idx]
    
    # Ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, np.inf)
    
    # Clip to valid range: must be >= 0 and <= 3/k (actually the constraint is "below -3/k" is invalid,
    # so depth should be in [0, 3/k]? Let me re-read: "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
    # So depth must be >= -3/k. But depth should be positive. Let's clip to [0, 3/k] to be safe.
    max_d = 3.0 / k
    bid_depths = np.clip(bid_depths, 0.0, max_d)
    ask_depths = np.clip(ask_depths, 0.0, max_d)
    
    # Replace inf with a large value (but <= 3/k)
    bid_depths = np.where(np.isinf(bid_depths), max_d, bid_depths)
    ask_depths = np.where(np.isinf(ask_depths), max_d, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
