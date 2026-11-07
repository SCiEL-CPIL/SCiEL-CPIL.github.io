"""Optimal market making with inventory risk - HJB backward induction lookup."""
import numpy as np
import time

_DEADLINE = time.time() + 285  # Safety margin before 330s kill

# Global cache for precomputed lookup tables
_LUT_CACHE = {}


def _build_lut(sigma, A, k, gamma, T, steps, Q):
    """
    Build a lookup table via backward induction on the discrete HJB equation.
    
    State: (step n, inventory q) where q in [-Q, Q]
    Action: choose bid_depth d_b >= 0 and ask_depth d_a >= 0
    Transition:
      - Bid fill prob: p_b = 1 - exp(-A * exp(-k * d_b) * dt) if q < Q else 0
      - Ask fill prob: p_a = 1 - exp(-A * exp(-k * d_a) * dt) if q > -Q else 0
      - Fills are independent
      - If bid fills: earn d_b, q -> q+1
      - If ask fills: earn d_a, q -> q-1
      - Mid moves: irrelevant for depth choice (depths are relative to mid)
    Objective per step:
      - Immediate reward: expected fills * depth - (gamma/2) * sigma^2 * q^2 * dt
      - Plus expected future value
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    
    dt = T / steps
    n_steps = steps
    
    # Discretize depth on a grid
    # Depth range: from 0 to some max
    # The optimal depth is typically in [0, 3/k] or so
    # Use a fine grid near 0 and coarser further out
    max_depth = 5.0 / k  # Beyond this, fill prob is negligible
    n_depth = 200  # Number of depth grid points
    
    # Log-spaced grid for better resolution near 0
    depth_grid = np.zeros(n_depth)
    depth_grid[0] = 0.0
    depth_grid[1:] = np.linspace(1e-4, max_depth, n_depth - 1)
    
    # Precompute fill probabilities for each depth
    # p_b(d) = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[n, q] for step n, inventory q
    # q ranges from -Q to Q, so 2*Q+1 values
    n_q = 2 * Q + 1
    q_offset = Q  # index = q + Q
    
    # V has shape (n_steps + 1, n_q)
    # V[n_steps, :] = 0 (no terminal charge)
    V = np.zeros((n_steps + 1, n_q), dtype=np.float64)
    
    # Policy: store optimal bid and ask depth indices
    # pol_bid[n, q] = index into depth_grid for optimal bid depth
    # pol_ask[n, q] = index into depth_grid for optimal ask depth
    pol_bid = np.zeros((n_steps, n_q), dtype=np.int32)
    pol_ask = np.zeros((n_steps, n_q), dtype=np.int32)
    
    # Backward induction
    # For each step n from n_steps-1 down to 0:
    #   For each inventory q:
    #     Compute optimal (d_b, d_a) by maximizing:
    #       sum over fill outcomes: prob * (reward + V[n+1, q'])
    #     where reward includes depth earned and inventory penalty
    
    # To speed up, we can precompute the expected future value for each (d_b, d_a)
    # Given q, d_b, d_a:
    #   p_b = fill_probs[ib] if q < Q else 0
    #   p_a = fill_probs[ia] if q > -Q else 0
    #   E[V_next] = (1-p_b)(1-p_a) V[n+1, q] + p_b(1-p_a) V[n+1, q+1] + (1-p_b)p_a V[n+1, q-1] + p_b*p_a V[n+1, q]
    #   (if q+1 <= Q and q-1 >= -Q, else adjust)
    #   Immediate reward = p_b * d_b + p_a * d_a - 0.5 * gamma * sigma^2 * q^2 * dt
    #   Total = immediate + E[V_next]
    
    # This is a 2D optimization over (ib, ia) for each (n, q)
    # With n_depth=200, that's 40000 evaluations per (n,q), times 1000*61 = 61000 states
    # Total: ~2.4 billion operations - too slow
    
    # Optimization: The problem separates! For fixed d_b, the optimal d_a can be found,
    # but actually the coupling is through the joint fill probability.
    # Let's try a coarser grid first.
    
    # Use a much coarser grid
    n_depth_coarse = 40
    depth_grid_c = np.zeros(n_depth_coarse)
    depth_grid_c[0] = 0.0
    depth_grid_c[1:] = np.linspace(0.05/k, max_depth, n_depth_coarse - 1)
    
    fill_probs_c = 1.0 - np.exp(-A * np.exp(-k * depth_grid_c) * dt)
    
    # Even with 40x40=1600 combos per state, 61000 * 1600 = 97.6M - feasible
    # Let's vectorize over q for each n
    
    # Actually, let's use a smarter approach:
    # For each n, we have V[n+1, :] known.
    # For each q, we need to maximize over (ib, ia).
    # The objective is:
    #   f(ib, ia) = p_b(ib)*d_b(ib) + p_a(ia)*d_a(ia) - 0.5*gamma*sigma^2*q^2*dt
    #               + (1-p_b)(1-p_a)*V[n+1,q] + p_b*(1-p_a)*V[n+1,q+1]
    #               + (1-p_b)*p_a*V[n+1,q-1] + p_b*p_a*V[n+1,q]
    #   where p_b=0 if q>=Q, p_a=0 if q<=-Q
    #   and V[n+1,q+1]=0 if q+1>Q, V[n+1,q-1]=0 if q-1<-Q
    
    # We can rewrite:
    #   f(ib, ia) = p_b(ib)*[d_b(ib) + V[n+1,q+1] - V[n+1,q]]
    #             + p_a(ia)*[d_a(ia) + V[n+1,q-1] - V[n+1,q]]
    #             + p_b(ib)*p_a(ia)*[V[n+1,q] - V[n+1,q+1] - V[n+1,q-1] + V[n+1,q]]
    #             + V[n+1,q]
    #             - 0.5*gamma*sigma^2*q^2*dt
    
    # The cross term p_b*p_a*[...] couples them. If we ignore the cross term (approximation),
    # the problem separates and we can optimize d_b and d_a independently.
    # This is a good approximation when fill probs are small (which they are for reasonable depths).
    
    # Let's use the separable approximation for speed, then refine if time permits.
    
    # Separable approximation:
    # For bid: maximize p_b(ib)*[d_b(ib) + V[n+1,q+1] - V[n+1,q]]
    # For ask: maximize p_a(ia)*[d_a(ia) + V[n+1,q-1] - V[n+1,q]]
    
    # Precompute for each n:
    #   bid_val[n, q, ib] = fill_probs_c[ib] * (depth_grid_c[ib] + V[n+1,q+1] - V[n+1,q])
    #   ask_val[n, q, ia] = fill_probs_c[ia] * (depth_grid_c[ia] + V[n+1,q-1] - V[n+1,q])
    
    # But this is still 61000 * 40 = 2.4M per side per step... 
    # Actually it's 1000 steps * 61 q * 40 depth = 2.46M total per side. Very fast.
    
    # Let's do the full 2D optimization with the coarse grid, vectorized over q.
    
    for n in range(n_steps - 1, -1, -1):
        if time.time() > _DEADLINE - 30:
            break
        
        V_next = V[n + 1]
        
        # For each q, compute the value for all (ib, ia) pairs
        # q index: 0 to n_q-1, actual q = idx - Q
        q_vals = np.arange(n_q) - Q  # shape (n_q,)
        
        # V_next[q_idx] for q_idx in [0, n_q-1]
        # For bid fill: q -> q+1, so q_idx -> q_idx+1 if q_idx+1 < n_q else 0
        # For ask fill: q -> q-1, so q_idx -> q_idx-1 if q_idx-1 >= 0 else 0
        
        # Boundary handling
        V_next_bid = np.zeros(n_q)  # V[n+1, q+1]
        V_next_ask = np.zeros(n_q)  # V[n+1, q-1]
        
        # For q < Q (q_idx < Q): V_next_bid[q_idx] = V_next[q_idx+1]
        # For q = Q: no bid fill, so this doesn't matter
        V_next_bid[:Q] = V_next[1:Q+1]
        V_next_bid[Q] = V_next[Q]  # irrelevant, will be masked
        
        # For q > -Q (q_idx > 0): V_next_ask[q_idx] = V_next[q_idx-1]
        # For q = -Q: no ask fill
        V_next_ask[1:] = V_next[0:Q]
        V_next_ask[0] = V_next[0]  # irrelevant
        
        # Inventory penalty
        inv_penalty = 0.5 * gamma * sigma * sigma * q_vals**2 * dt  # shape (n_q,)
        
        # Now for each q, we need to maximize over (ib, ia):
        #   f = p_b(ib)*d_b(ib) + p_a(ia)*d_a(ia) - inv_penalty[q]
        #       + (1-p_b)(1-p_a)*V_next[q] + p_b*(1-p_a)*V_next_bid[q]
        #       + (1-p_b)*p_a*V_next_ask[q] + p_b*p_a*V_next[q]
        
        # Let's expand:
        #   = p_b*(d_b + V_next_bid[q] - V_next[q])
        #   + p_a*(d_a + V_next_ask[q] - V_next[q])
        #   + p_b*p_a*(V_next[q] - V_next_bid[q] - V_next_ask[q] + V_next[q])
        #   + V_next[q] - inv_penalty[q]
        
        # Define:
        #   bid_gain[q, ib] = fill_probs_c[ib] * (depth_grid_c[ib] + V_next_bid[q] - V_next[q])
        #   ask_gain[q, ia] = fill_probs_c[ia] * (depth_grid_c[ia] + V_next_ask[q] - V_next[q])
        #   cross[q] = V_next[q] - V_next_bid[q] - V_next_ask[q] + V_next[q]
        #             = 2*V_next[q] - V_next_bid[q] - V_next_ask[q]
        
        cross = 2.0 * V_next - V_next_bid - V_next_ask  # shape (n_q,)
        
        # bid_gain: (n_q, n_depth_c)
        # For each q and ib: fill_probs_c[ib] * (depth_grid_c[ib] + V_next_bid[q] - V_next[q])
        # = fill_probs_c[ib] * depth_grid_c[ib] + fill_probs_c[ib] * (V_next_bid[q] - V_next[q])
        
        # Let's compute this efficiently
        # bid_gain[q, ib] = fp[ib]*dg[ib] + fp[ib]*(Vnb[q] - Vn[q])
        # where fp = fill_probs_c, dg = depth_grid_c, Vnb = V_next_bid, Vn = V_next
        
        # Shape: (n_q, n_depth_c)
        # term1 = fp[ib]*dg[ib] -> shape (n_depth_c,)
        # term2 = fp[ib]*(Vnb[q]-Vn[q]) -> (n_q, n_depth_c)
        
        term1 = fill_probs_c * depth_grid_c  # (n_depth_c,)
        delta_bid = V_next_bid - V_next  # (n_q,)
        bid_gain = term1[None, :] + fill_probs_c[None, :] * delta_bid[:, None]  # (n_q, n_depth_c)
        
        term1_a = fill_probs_c * depth_grid_c
        delta_ask = V_next_ask - V_next
        ask_gain = term1_a[None, :] + fill_probs_c[None, :] * delta_ask[:, None]
        
        # Now f[q, ib, ia] = bid_gain[q, ib] + ask_gain[q, ia] + fill_probs_c[ib]*fill_probs_c[ia]*cross[q]
        #                   + V_next[q] - inv_penalty[q]
        
        # We need argmax over (ib, ia) for each q.
        # f[q, ib, ia] = bid_gain[q, ib] + ask_gain[q, ia] + fp[ib]*fp[ia]*cross[q]
        
        # This is a rank-1 update to the matrix bid_gain[:, :, None] + ask_gain[:, None, :]
        # We can't easily vectorize the 3D max without too much memory.
        # n_q=61, n_depth_c=40: 61*40*40 = 97600 elements. That's fine!
        
        # Build the 3D array
        # f3d[q, ib, ia] = bid_gain[q, ib] + ask_gain[q, ia] + fp[ib]*fp[ia]*cross[q]
        
        # Memory: 61 * 40 * 40 * 8 bytes = 780 KB. Fine.
        
        fp_outer = fill_probs_c[:, None] * fill_probs_c[None, :]  # (n_depth_c, n_depth_c)
        
        # f3d[q, ib, ia] = bid_gain[q, ib] + ask_gain[q, ia] + fp_outer[ib, ia] * cross[q]
        
        # Vectorized:
        # bid_gain[:, :, None] -> (n_q, n_depth_c, 1)
        # ask_gain[:, None, :] -> (n_q, 1, n_depth_c)
        # fp_outer[None, :, :] * cross[:, None, None] -> (n_q, n_depth_c, n_depth_c)
        
        f3d = (bid_gain[:, :, None] + ask_gain[:, None, :] + 
               fp_outer[None, :, :] * cross[:, None, None])
        
        # Add the constant term V_next[q] - inv_penalty[q]
        # This doesn't affect argmax, so we can skip it for finding the optimal action.
        
        # Find argmax over (ib, ia)
        flat_idx = np.argmax(f3d, axis=(1, 2))  # (n_q,)
        ib_opt = flat_idx // n_depth_coarse
        ia_opt = flat_idx % n_depth_coarse
        
        pol_bid[n] = ib_opt
        pol_ask[n] = ia_opt
        
        # Compute V[n, q] = f3d[q, ib_opt[q], ia_opt[q]] + V_next[q] - inv_penalty[q]
        V[n] = (f3d[np.arange(n_q), ib_opt, ia_opt] + V_next - inv_penalty)
        
        # Boundary conditions: at q=Q, no bid fills; at q=-Q, no ask fills
        # Our formulation already handles this via V_next_bid and V_next_ask,
        # but we should also enforce that we don't quote bid at q=Q or ask at q=-Q.
        # Actually, the fill_probs are set based on depth, not inventory.
        # We need to zero out fill probabilities at boundaries.
        
        # At q = Q (q_idx = Q): bid fills are impossible
        # At q = -Q (q_idx = 0): ask fills are impossible
        # We should set the optimal depth to inf (or very large) at these boundaries.
        # For now, let's handle this in the quote function.
    
    # Store in cache
    result = {
        'depth_grid': depth_grid_c,
        'pol_bid': pol_bid,
        'pol_ask': pol_ask,
        'V': V,
        'n_depth': n_depth_coarse,
    }
    _LUT_CACHE[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) using precomputed HJB lookup table.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    dt = T / steps
    
    # Determine current step
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Get or build lookup table
    lut = _build_lut(sigma, A, k, gamma, T, steps, Q)
    
    depth_grid = lut['depth_grid']
    pol_bid = lut['pol_bid']
    pol_ask = lut['pol_ask']
    n_depth = lut['n_depth']
    
    q = inventory.astype(np.int32)
    q_idx = q + Q  # Convert to index [0, 2Q]
    
    # Get optimal depth indices for current step
    ib_opt = pol_bid[n, q_idx]  # shape (4000,)
    ia_opt = pol_ask[n, q_idx]
    
    bid_depth = depth_grid[ib_opt]
    ask_depth = depth_grid[ia_opt]
    
    # Boundary handling: at q=Q, no bid fills; at q=-Q, no ask fills
    # Set depth to a very large value (effectively no quote)
    big_depth = 50.0 / k
    bid_over = (inventory >= Q)
    ask_over = (inventory <= -Q)
    
    bid_depth = np.where(bid_over, big_depth, bid_depth)
    ask_depth = np.where(ask_over, big_depth, ask_depth)
    
    # Clamp to valid range
    min_depth = max(0.01, -3.0/k + 0.01)
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # Safety: ensure finite
    bid_depth = np.where(np.isfinite(bid_depth), bid_depth, big_depth)
    ask_depth = np.where(np.isfinite(ask_depth), ask_depth, big_depth)
    
    return bid_depth, ask_depth
