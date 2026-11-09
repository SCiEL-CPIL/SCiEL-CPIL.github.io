import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0  # safety margin before 330s kill
_cache = {}

def _solve_regime(params):
    """
    Solve the continuous-time HJB via finite differences, then derive
    optimal depths from first-order conditions.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    alpha = A * dt  # fill rate scaling per step

    # We solve the HJB on a grid of (n, q) but use a PDE-inspired approach.
    # The value function V(t, q) satisfies:
    # ∂V/∂t + max_d [A*exp(-k*d)*(d + V(q+1) - V(q)) - gamma/2 * sigma^2 * q^2] = 0
    # (continuous time limit, but we keep discrete q)
    #
    # Actually, let's do a refined discrete DP but with a much finer depth grid
    # and use the structure of the problem to speed things up.
    #
    # The key insight: for each (n, q), the optimal bid depth d_b maximizes:
    #   f(d) = P(d) * (d + V_next[q+1])
    # where P(d) = 1 - exp(-alpha * exp(-k*d))
    #
    # f'(d) = P'(d)*(d + V) + P(d) = 0
    # P'(d) = alpha * exp(-k*d) * exp(-alpha*exp(-k*d)) * k
    #
    # This is a 1D root-finding problem. We can solve it efficiently.
    #
    # Let's precompute the optimal depths for a range of "effective" values
    # and interpolate. The effective value is V_next[q+1] (or V_next[q-1]).
    #
    # Actually, let's just do the DP but with a smarter depth grid.
    # The optimal depth is typically small (0.01 to ~2.0 for these parameters).
    # Use a non-uniform grid concentrated near 0.

    # Number of depth grid points
    N_D = 500

    # Create a grid: mostly linear in [0.005, 3.0] but with extra points near 0
    # Use a combination: log-spaced near 0, linear elsewhere
    d_grid = np.concatenate([
        np.linspace(0.005, 0.1, 100),
        np.linspace(0.1, 1.0, 200),
        np.linspace(1.0, 3.0, 200)
    ])
    # Ensure strictly increasing
    d_grid = np.unique(d_grid)
    N_D = len(d_grid)

    # Precompute P(d) on the grid
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-alpha * exp_term)
    Pd_grid = P_grid * d_grid

    # V_next: value function at next time step, shape (2*Q+1,)
    V_next = np.zeros(2 * Q + 1)

    # Store optimal depths
    bid_depths = np.zeros((steps, 2 * Q + 1))
    ask_depths = np.zeros((steps, 2 * Q + 1))

    q_vals = np.arange(-Q, Q + 1)
    inv_cost_base = (gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt

    bid_mask = np.arange(2 * Q + 1) < 2 * Q   # q < Q
    ask_mask = np.arange(2 * Q + 1) > 0        # q > -Q

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5:
            # Not enough time left; fill remaining with current best estimates
            # Use the last computed depths for all remaining steps
            for nn in range(n - 1, -1, -1):
                bid_depths[nn] = bid_depths[n]
                ask_depths[nn] = ask_depths[n]
            break

        # V_next_plus[i] = V_next[i+1] for i < 2Q (bid fill: q -> q+1)
        V_next_plus = np.zeros(2 * Q + 1)
        V_next_plus[:-1] = V_next[1:]
        V_next_plus[-1] = 0  # masked out

        # V_next_minus[i] = V_next[i-1] for i > 0 (ask fill: q -> q-1)
        V_next_minus = np.zeros(2 * Q + 1)
        V_next_minus[1:] = V_next[:-1]
        V_next_minus[0] = 0  # masked out

        # Objective: P(d) * (d + V_eff)
        # Shape: (N_D, 2Q+1)
        obj_bid = P_grid[:, None] * (d_grid[:, None] + V_next_plus[None, :])
        obj_ask = P_grid[:, None] * (d_grid[:, None] + V_next_minus[None, :])

        max_idx_bid = np.argmax(obj_bid, axis=0)
        max_idx_ask = np.argmax(obj_ask, axis=0)

        optimal_d_bid = d_grid[max_idx_bid]
        optimal_d_ask = d_grid[max_idx_ask]

        max_val_bid = obj_bid[max_idx_bid, np.arange(2 * Q + 1)]
        max_val_ask = obj_ask[max_idx_ask, np.arange(2 * Q + 1)]

        # Refine: local search around the grid optimum
        # For each q, do a few Newton steps on the continuous objective
        # f(d) = (1 - exp(-alpha * exp(-k*d))) * (d + V_eff)
        # f'(d) = alpha*k*exp(-k*d)*exp(-alpha*exp(-k*d)) * (d + V_eff) + (1 - exp(-alpha*exp(-k*d)))
        
        for idx in range(2 * Q + 1):
            if bid_mask[idx]:
                V_eff = V_next_plus[idx]
                d0 = optimal_d_bid[idx]
                # Newton refinement
                for _ in range(3):
                    e = np.exp(-k * d0)
                    P = 1.0 - np.exp(-alpha * e)
                    f = P * (d0 + V_eff)
                    fp = alpha * k * e * np.exp(-alpha * e) * (d0 + V_eff) + P
                    if abs(fp) < 1e-12:
                        break
                    d_new = d0 - f / fp
                    if d_new < 0.001:
                        d_new = 0.001
                    if abs(d_new - d0) < 1e-10:
                        break
                    d0 = d_new
                optimal_d_bid[idx] = d0

            if ask_mask[idx]:
                V_eff = V_next_minus[idx]
                d0 = optimal_d_ask[idx]
                for _ in range(3):
                    e = np.exp(-k * d0)
                    P = 1.0 - np.exp(-alpha * e)
                    f = P * (d0 + V_eff)
                    fp = alpha * k * e * np.exp(-alpha * e) * (d0 + V_eff) + P
                    if abs(fp) < 1e-12:
                        break
                    d_new = d0 - f / fp
                    if d_new < 0.001:
                        d_new = 0.001
                    if abs(d_new - d0) < 1e-10:
                        break
                    d0 = d_new
                optimal_d_ask[idx] = d0

        # Update value function
        V_curr = -inv_cost_base.copy()
        V_curr[bid_mask] += max_val_bid[bid_mask]
        V_curr[ask_mask] += max_val_ask[ask_mask]

        bid_depths[n, :] = optimal_d_bid
        ask_depths[n, :] = optimal_d_ask
        bid_depths[n, ~bid_mask] = np.inf
        ask_depths[n, ~ask_mask] = np.inf

        V_next = V_curr

    result = {'bid': bid_depths, 'ask': ask_depths}
    _cache[key] = result
    return result


def quote(t, inventory, mid, params):
    policy = _solve_regime(params)
    steps = params['steps']
    Q = params['Q']

    # Determine current step index
    n = int(np.round(t * params['steps'] / params['T']))
    n = np.clip(n, 0, steps - 1)

    # Map inventory to indices
    indices = inventory + Q
    indices = np.clip(indices, 0, 2 * Q)

    bid_depths = policy['bid'][n, indices].astype(np.float64)
    ask_depths = policy['ask'][n, indices].astype(np.float64)

    # Ensure no invalid values
    bid_depths = np.where(bid_depths < -3.0 / params['k'], np.inf, bid_depths)
    ask_depths = np.where(ask_depths < -3.0 / params['k'], np.inf, ask_depths)
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)

    return bid_depths, ask_depths
