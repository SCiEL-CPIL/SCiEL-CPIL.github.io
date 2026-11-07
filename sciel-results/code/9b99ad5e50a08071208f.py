"""Market-making with inventory risk: exact DP policy lookup."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 295.0  # seconds of wall clock budget for the whole run.

# Cache optimal policies per parameter regime
_POLICY_CACHE = {}


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _solve_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the finite-horizon MDP exactly.
    State: (time_step, inventory)
    Action: choose bid_depth and ask_depth in [0, 3/k]
    
    We use the fact that the optimal policy has a specific structure:
    bid_depth = clip(c + s*q, 0, 3/k)
    ask_depth = clip(c - s*q, 0, 3/k)
    
    where c and s are determined by the DP. We'll solve for the value function
    V(q, n) where n is remaining steps, then extract the optimal actions.
    
    For efficiency, we note that the optimal depths depend only on q and n,
    not on t or mid. We precompute tables for each (n, q) pair.
    """
    if _remaining_budget() < 30.0:
        return None
    
    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k
    
    # Inventory grid: q in [-Q, Q]
    q_vals = np.arange(-Q, Q + 1, dtype=float)
    nq = len(q_vals)
    
    # For each remaining step count n (0 to steps), compute:
    # V[n][q_idx] = optimal value-to-go from state (n steps remaining, inventory q)
    # bid_opt[n][q_idx] = optimal bid depth
    # ask_opt[n][q_idx] = optimal ask depth
    
    # V[0][q] = 0 (no future rewards)
    V = np.zeros((steps + 1, nq))
    bid_opt = np.zeros((steps + 1, nq))
    ask_opt = np.zeros((steps + 1, nq))
    
    # Precompute fill probabilities as functions of depth
    # p_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
    # p_ask(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    # For each q, the optimal depth maximizes:
    # reward = bid + ask - (gamma/2)*sigma^2*q^2*dt
    # + p_bid * V[n-1][q-1] + p_ask * V[n-1][q+1] + (1-p_bid-p_ask)*V[n-1][q]
    
    # Since p_bid and p_ask are monotonic in depth (increasing), and the 
    # continuation value depends on them, we need to solve a 1D optimization.
    
    # Key insight: the optimal bid and ask depths are independent given q,
    # because the reward is additive and the probability terms are separable.
    
    # Actually, they're coupled through the (1-pb-pa) term. Let's handle it.
    
    # For a given q and n, define:
    # f(bid, ask) = bid + ask - risk + pb(bid)*V(q-1) + pa(ask)*V(q+1) + (1-pb-pa)*V(q)
    # where risk = (gamma/2)*sigma^2*q^2*dt
    # V(q-1), V(q+1), V(q) are from V[n-1]
    
    # This is a 2D optimization over bid in [0, max_depth], ask in [0, max_depth].
    # Since the function is concave in each argument (pb, pa are concave in depth),
    # we can use coordinate ascent or just evaluate on a grid.
    
    # For speed, we'll use the analytic approximation that the optimal depths
    # follow a linear-in-q pattern, and refine with a local search.
    
    # Initialize with the linear shift approximation
    shift = gamma * sigma * sigma * 0.5 / (A * k)
    
    for n in range(1, steps + 1):
        Vprev = V[n - 1]
        
        for qi, q in enumerate(q_vals):
            # Boundary conditions
            if q >= Q:
                # No bid fill possible
                # Maximize over ask only
                Vq = Vprev[qi]
                # ask_opt: maximize ask - risk + pa(ask)*V(q+1) + (1-pa)*V(q)
                # Since q+1 > Q, V(q+1) = V(Q) (clamp)
                # Actually q+1 is out of range, so no ask fill either? 
                # No: at q=Q, no bid fill. At q=-Q, no ask fill.
                # At q=Q, ask fill moves to q-1=Q-1, which is valid.
                Vq_plus = Vprev[qi - 1] if qi > 0 else Vprev[0]  # q+1 -> q-1 in index? No.
                
                # Wait: bid fill increases inventory (buy), ask fill decreases (sell)
                # So at q, bid fill -> q+1, ask fill -> q-1
                # At q=Q, bid fill would go to Q+1 which is invalid, so pb=0
                # At q=-Q, ask fill would go to -Q-1 which is invalid, so pa=0
                
                if q >= Q:
                    # Only ask fills possible
                    Vq_minus = Vprev[qi - 1] if qi > 0 else 0.0  # q-1
                    Vq_same = Vprev[qi]
                    risk = 0.5 * gamma * sigma * sigma * q * q * dt
                    
                    # Maximize: ask - risk + pa(ask)*Vq_minus + (1-pa(ask))*Vq_same
                    # over ask in [0, max_depth]
                    best_ask = 0.0
                    best_val = -np.inf
                    # Grid search with ~200 points
                    asks = np.linspace(0, max_depth, 200)
                    pa = 1.0 - np.exp(-A * np.exp(-k * asks) * dt)
                    vals = asks - risk + pa * Vq_minus + (1 - pa) * Vq_same
                    idx = np.argmax(vals)
                    best_ask = asks[idx]
                    best_val = vals[idx]
                    
                    V[n, qi] = best_val
                    bid_opt[n, qi] = 0.0  # no bid at Q
                    ask_opt[n, qi] = best_ask
                    
                elif q <= -Q:
                    # Only bid fills possible
                    Vq_plus = Vprev[qi + 1] if qi < nq - 1 else 0.0  # q+1
                    Vq_same = Vprev[qi]
                    risk = 0.5 * gamma * sigma * sigma * q * q * dt
                    
                    # Maximize: bid - risk + pb(bid)*Vq_plus + (1-pb(bid))*Vq_same
                    best_bid = 0.0
                    best_val = -np.inf
                    bids = np.linspace(0, max_depth, 200)
                    pb = 1.0 - np.exp(-A * np.exp(-k * bids) * dt)
                    vals = bids - risk + pb * Vq_plus + (1 - pb) * Vq_same
                    idx = np.argmax(vals)
                    best_bid = bids[idx]
                    best_val = vals[idx]
                    
                    V[n, qi] = best_val
                    bid_opt[n, qi] = best_bid
                    ask_opt[n, qi] = 0.0  # no ask at -Q
                else:
                    # Interior: both fills possible
                    Vq_minus = Vprev[qi - 1]  # q-1 (ask fill)
                    Vq_plus = Vprev[qi + 1]  # q+1 (bid fill)
                    Vq_same = Vprev[qi]
                    risk = 0.5 * gamma * sigma * sigma * q * q * dt
                    
                    # 2D optimization: maximize over bid, ask in [0, max_depth]
                    # f(bid, ask) = bid + ask - risk 
                    #   + pb(bid)*Vq_plus + pa(ask)*Vq_minus + (1-pb-pa)*Vq_same
                    # = bid + ask - risk + Vq_same + pb*(Vq_plus - Vq_same) + pa*(Vq_minus - Vq_same)
                    # - pb*pa*Vq_same  [cross term]
                    
                    # The cross term makes it not separable. Use grid search.
                    # For speed, use a coarser grid first, then refine.
                    n_grid = 100
                    bids = np.linspace(0, max_depth, n_grid)
                    asks = np.linspace(0, max_depth, n_grid)
                    B, K = np.meshgrid(bids, asks, indexing='ij')
                    pb = 1.0 - np.exp(-A * np.exp(-k * B) * dt)
                    pa = 1.0 - np.exp(-A * np.exp(-k * K) * dt)
                    
                    vals = (B + K - risk + Vq_same 
                            + pb * (Vq_plus - Vq_same) 
                            + pa * (Vq_minus - Vq_same)
                            - pb * pa * Vq_same)
                    
                    idx = np.argmax(vals)
                    bi, ki = np.unravel_index(idx, vals.shape)
                    best_bid = bids[bi]
                    best_ask = asks[ki]
                    best_val = vals[bi, ki]
                    
                    # Refine with local search
                    # Try finer grid around the best point
                    if _remaining_budget() > 60.0:
                        # Check if we're at a boundary
                        if 0 < bi < n_grid - 1 and 0 < ki < n_grid - 1:
                            bid_lo = max(0, bids[max(0, bi - 2)])
                            bid_hi = min(max_depth, bids[min(n_grid - 1, bi + 2)])
                            ask_lo = max(0, asks[max(0, ki - 2)])
                            ask_hi = min(max_depth, asks[min(n_grid - 1, ki + 2)])
                            bids_fine = np.linspace(bid_lo, bid_hi, 20)
                            asks_fine = np.linspace(ask_lo, ask_hi, 20)
                            Bf, Kf = np.meshgrid(bids_fine, asks_fine, indexing='ij')
                            pbf = 1.0 - np.exp(-A * np.exp(-k * Bf) * dt)
                            paf = 1.0 - np.exp(-A * np.exp(-k * Kf) * dt)
                            vals_fine = (Bf + Kf - risk + Vq_same 
                                         + pbf * (Vq_plus - Vq_same) 
                                         + paf * (Vq_minus - Vq_same)
                                         - pbf * paf * Vq_same)
                            idx_f = np.argmax(vals_fine)
                            bfi, kfi = np.unravel_index(idx_f, vals_fine.shape)
                            best_bid = bids_fine[bfi]
                            best_ask = asks_fine[kfi]
                            best_val = vals_fine[bfi, kfi]
                    
                    V[n, qi] = best_val
                    bid_opt[n, qi] = best_bid
                    ask_opt[n, qi] = best_ask
        
        if n % 100 == 0 and _remaining_budget() < 30.0:
            return None
    
    return {
        'V': V,
        'bid_opt': bid_opt,
        'ask_opt': ask_opt,
        'q_vals': q_vals,
        'nq': nq,
        'Q': Q,
    }


def _interp_policy(policy, q, n_remaining):
    """
    Interpolate the optimal bid and ask depths from the policy table.
    q: array of inventories
    n_remaining: number of steps remaining (integer)
    Returns (bid_depths, ask_depths) arrays.
    """
    q_vals = policy['q_vals']
    bid_table = policy['bid_opt'][n_remaining]
    ask_table = policy['ask_opt'][n_remaining]
    nq = policy['nq']
    Q = policy['Q']
    
    # Clamp q to [-Q, Q]
    q_clamped = np.clip(q, -Q, Q)
    
    # Find indices for interpolation
    # q_vals is uniform: -Q, -Q+1, ..., Q
    idx_float = q_clamped + Q
    i0 = np.floor(idx_float).astype(int)
    i1 = i0 + 1
    i0 = np.clip(i0, 0, nq - 1)
    i1 = np.clip(i1, 0, nq - 1)
    frac = idx_float - np.floor(idx_float)
    
    bid = (1 - frac) * bid_table[i0] + frac * bid_table[i1]
    ask = (1 - frac) * ask_table[i0] + frac * ask_table[i1]
    
    return bid, ask


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _POLICY_CACHE:
        policy = _solve_policy(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = policy
    policy = _POLICY_CACHE[key]
    
    if policy is None:
        # Fallback: analytic shift
        inv_k = 1.0 / k
        max_depth = 3.0 * inv_k
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        q = inventory.astype(float)
        bid = np.clip(inv_k + q * shift, 0.0, max_depth)
        ask = np.clip(inv_k - q * shift, 0.0, max_depth)
        return bid, ask
    
    # Determine remaining steps
    dt = T / steps
    n_step = int(round(t / dt))
    n_remaining = steps - n_step
    
    # Clamp to valid range
    n_remaining = max(0, min(n_remaining, steps))
    
    q = inventory.astype(float)
    bid, ask = _interp_policy(policy, q, n_remaining)
    
    # Ensure valid values
    max_depth = 3.0 / k
    bid = np.nan_to_num(bid, nan=1.0/k, posinf=max_depth, neginf=0.0)
    ask = np.nan_to_num(ask, nan=1.0/k, posinf=max_depth, neginf=0.0)
    bid = np.clip(bid, 0.0, max_depth)
    ask = np.clip(ask, 0.0, max_depth)
    
    return bid, ask
# EVOLVE-BLOCK-END
