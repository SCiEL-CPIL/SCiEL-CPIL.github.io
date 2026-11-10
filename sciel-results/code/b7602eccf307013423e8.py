"""
Optimal market making with inventory risk using HJB approximation.

Strategy:
1. Precompute value function V(t, q) on a grid using semi-Lagrangian DP.
2. At each step, look up V(t, q) for the current inventory.
3. Compute optimal bid/ask depths from the first-order condition.

The key insight: for a given state (t, q), the optimal depth on each side
balances the expected fill gain against the inventory risk. We use the
discrete model's transition probabilities directly.
"""

import numpy as np
import time

# Global cache for precomputed value functions
_V_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete DP for the optimal policy.
    
    Returns arrays:
      bid_depths[n, q] : optimal bid depth at step n, inventory q
      ask_depths[n, q] : optimal ask depth at step n, inventory q
      V[n, q] : value function at step n, inventory q
    """
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # q ranges from -Q to Q
    
    # Index: q_idx = q + Q, so q = -Q -> 0, q = Q -> 2Q
    # q ranges from -Q to Q
    
    # Precompute fill probabilities for a range of depths
    # For efficiency, we'll compute these on the fly or precompute for common depths
    
    # Value function: V[n, q] = max over depths of expected reward + continuation
    # At step n, time t = n * dt
    # t_rem = (n_steps - n) * dt
    
    # We'll do backward induction
    # V[steps, q] = 0 (no terminal charge)
    V = np.zeros((n_steps + 1, n_inv), dtype=np.float64)
    
    # Optimal depths
    bid_depths = np.zeros((n_steps, n_inv), dtype=np.float64)
    ask_depths = np.zeros((n_steps, n_inv), dtype=np.float64)
    
    # For the first-order condition, we need to optimize over depth.
    # The reward from quoting at depth d on bid side:
    #   Fill prob: p_bid = 1 - exp(-A * exp(-k*d) * dt)
    #   If filled: gain d, inventory changes from q to q-1
    #   If not filled: gain 0, inventory stays q
    #   Plus: -gamma/2 * sigma^2 * q^2 * dt (inventory holding cost)
    #   Plus: continuation value E[V[n+1, q']]
    
    # Since the mid moves randomly, but the inventory holding cost depends on q,
    # and the fill probabilities depend only on depth (not on mid level),
    # the optimal policy is actually independent of the mid level!
    # This is a key simplification.
    
    # So we can solve this as a 1D DP over inventory.
    
    # For each step n (backward from n_steps-1 to 0) and each inventory q:
    #   Maximize over bid_depth >= 0, ask_depth >= 0:
    #     -gamma/2 * sigma^2 * q^2 * dt
    #     + p_bid * (bid_depth + V[n+1, q-1])  if q < Q
    #     + (1 - p_bid) * V[n+1, q]  if q < Q
    #     + p_ask * (ask_depth + V[n+1, q+1])  if q > -Q
    #     + (1 - p_ask) * V[n+1, q]  if q > -Q
    #   where p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt)
    #         p_ask = 1 - exp(-A * exp(-k * ask_depth) * dt)
    
    # Note: bid and ask decisions are independent given q, so we can separate them.
    
    # For bid side (only if q < Q):
    #   Maximize over d >= 0:
    #     p(d) * (d + V[n+1, q-1]) + (1 - p(d)) * V[n+1, q]
    #   where p(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    # Let's define:
    #   f(d) = p(d) * (d + V_next[q-1]) + (1 - p(d)) * V_next[q]
    #        = V_next[q] + p(d) * (d + V_next[q-1] - V_next[q])
    #   Let delta = V_next[q-1] - V_next[q]
    #   f(d) = V_next[q] + p(d) * (d + delta)
    
    # We need to maximize p(d) * (d + delta) over d >= 0.
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    # For small dt, p(d) ≈ A * exp(-k*d) * dt
    # Then f(d) ≈ V_next[q] + A * dt * exp(-k*d) * (d + delta)
    # Maximize g(d) = exp(-k*d) * (d + delta)
    # g'(d) = exp(-k*d) * (-k*(d+delta) + 1) = 0
    # => d* = 1/k - delta
    
    # If d* < 0, then d* = 0 (boundary)
    # If d* > 0, use d*
    
    # But this is the small-dt approximation. For larger dt, we should be more careful.
    # Let's use the exact formula but solve numerically or use a good approximation.
    
    # Actually, let's just use the closed-form approximation since dt = 1/1000 is small.
    
    for n in range(n_steps - 1, -1, -1):
        V_next = V[n + 1]
        for q_idx in range(n_inv):
            q = q_idx - Q
            holding_cost = -0.5 * gamma * sigma * sigma * q * q * dt
            
            # Bid side
            if q < Q:
                # delta = V_next[q-1+Q] - V_next[q+Q]
                delta_bid = V_next[q - 1 + Q] - V_next[q_idx]
                # Optimal depth: d* = 1/k - delta_bid, clipped to [0, inf)
                d_opt = 1.0 / k - delta_bid
                if d_opt < 0:
                    d_opt = 0.0
                # Cap at a reasonable maximum to avoid numerical issues
                # exp(-k*d) becomes very small for large d
                if d_opt > 20.0 / k:
                    d_opt = 20.0 / k
                bid_depths[n, q_idx] = d_opt
            else:
                bid_depths[n, q_idx] = np.inf
            
            # Ask side
            if q > -Q:
                # For ask: gain d, inventory changes from q to q+1
                # delta_ask = V_next[q+1+Q] - V_next[q+Q]
                delta_ask = V_next[q + 1 + Q] - V_next[q_idx]
                d_opt = 1.0 / k - delta_ask
                if d_opt < 0:
                    d_opt = 0.0
                if d_opt > 20.0 / k:
                    d_opt = 20.0 / k
                ask_depths[n, q_idx] = d_opt
            else:
                ask_depths[n, q_idx] = np.inf
            
            # Compute V[n, q]
            # Bid contribution
            if q < Q:
                d_bid = bid_depths[n, q_idx]
                if np.isinf(d_bid):
                    p_bid = 0.0
                else:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
                bid_val = p_bid * (d_bid + V_next[q - 1 + Q]) + (1 - p_bid) * V_next[q_idx]
            else:
                bid_val = V_next[q_idx]
            
            # Ask contribution
            if q > -Q:
                d_ask = ask_depths[n, q_idx]
                if np.isinf(d_ask):
                    p_ask = 0.0
                else:
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
                ask_val = p_ask * (d_ask + V_next[q + 1 + Q]) + (1 - p_ask) * V_next[q_idx]
            else:
                ask_val = V_next[q_idx]
            
            V[n, q_idx] = holding_cost + bid_val + ask_val
    
    return bid_depths, ask_depths, V


def _get_regime_data(sigma, A, k, gamma, T, steps, Q):
    """Get or compute the DP solution for a given regime."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _V_CACHE:
        bid_depths, ask_depths, V = _solve_regime(sigma, A, k, gamma, T, steps, Q)
        _V_CACHE[key] = (bid_depths, ask_depths, V)
    return _V_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote both sides with optimal depths based on inventory and time remaining.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int32)
    
    # Determine step index
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Get precomputed depths
    bid_depths_grid, ask_depths_grid, V = _get_regime_data(sigma, A, k, gamma, T, steps, Q)
    
    # Look up depths for each path
    # q_idx = q + Q
    q_indices = inv + Q
    # Clip to valid range (should already be in [-Q, Q])
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    bid_depths = bid_depths_grid[n, q_indices]
    ask_depths = ask_depths_grid[n, q_indices]
    
    # Ensure no negative depths or NaN
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, 0.0), np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, 0.0), np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
