"""
Optimal market making via pre-computed DP policy lookup.
The state space is (step, inventory) with 1000 steps and 61 inventory levels (-30..30).
We solve the Bellman equation backwards to find optimal bid/ask depths for each state.
Results are cached per regime so only the first call pays the DP cost.
Optimized: inventory-independent terms computed once per step, log-spaced depth grid.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _heuristic_depths(params, t, inventory):
    """Analytical fallback: time-varying base depth with inventory skew."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    d0 = 1.0 / k
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0/k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0/k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    Optimized: precompute inventory-independent terms per step.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1)

    # Log-spaced depth grid: focus resolution around 1/k, extend range for boundaries
    depth_min = 0.005
    depth_max = 8.0 / k
    n_depths = 80
    # Log-spaced for better resolution at small depths
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    # Precompute fill probabilities for all depths (same for every step and inventory)
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)

    # Track whether we need a polish pass
    _polish_needed = True

    # V[n, qi] = max expected future reward from step n to end at inventory qi
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Fallback: fill remaining with heuristic
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * (T - n2 * dt) / (A * k)
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            break

        inv_cost = (gamma / 2.0) * sigma**2 * dt  # factor out; q^2 varies per state

        # Precompute per-step terms that don't depend on inventory
        # p_fill is already computed globally
        # For the matrix M[i,j] = p_bid[i]*diff[j] + ask_term[j], we need:
        #   diff[j] = term1[j] - term2[j]  -- depends on V[n+1, q], V[n+1, q+1], V[n+1, q-1]
        #   ask_term[j] = term2[j] + depth_grid[j] * p_ask[j]  -- depends on V[n+1, q], V[n+1, q-1]
        # These depend on inventory, so we still loop over inventory.
        # But we can precompute depth_grid * p_fill once.
        depth_p = depth_grid * p_fill  # shape (n_depths,)

        Vn1 = V[n + 1]  # reference to next row

        for qi in range(n_q):
            q = q_range[qi]
            cost = inv_cost * q * q

            Vn1_q = Vn1[qi]

            # Boundary: effective fill probabilities
            if q == Q:
                p_bid_eff = np.zeros(n_depths)
            else:
                p_bid_eff = p_fill

            if q == -Q:
                p_ask_eff = np.zeros(n_depths)
            else:
                p_ask_eff = p_fill

            if q < Q:
                Vn1_q_plus = Vn1[qi + 1]
            else:
                Vn1_q_plus = Vn1_q

            if q > -Q:
                Vn1_q_minus = Vn1[qi - 1]
            else:
                Vn1_q_minus = Vn1_q

            # term1[j] = p_ask_eff[j]*Vn1_q + (1-p_ask_eff[j])*Vn1_q_plus
            # term2[j] = p_ask_eff[j]*Vn1_q_minus + (1-p_ask_eff[j])*Vn1_q
            if q == -Q:
                # p_ask_eff = 0, so term1 = Vn1_q_plus, term2 = Vn1_q
                term1 = np.full(n_depths, Vn1_q_plus)
                term2 = np.full(n_depths, Vn1_q)
            elif q == Q:
                # p_bid_eff = 0, but term1/term2 still depend on p_ask_eff
                term1 = p_ask_eff * Vn1_q + (1 - p_ask_eff) * Vn1_q_plus
                term2 = p_ask_eff * Vn1_q_minus + (1 - p_ask_eff) * Vn1_q
            else:
                term1 = p_ask_eff * Vn1_q + (1 - p_ask_eff) * Vn1_q_plus
                term2 = p_ask_eff * Vn1_q_minus + (1 - p_ask_eff) * Vn1_q

            diff = term1 - term2
            ask_term = term2 + depth_p  # depth_p = depth_grid * p_fill = depth_grid * p_ask_eff (when not at -Q)

            if q == -Q:
                # p_ask_eff = 0, so ask_term = term2 + 0 = Vn1_q
                # depth_p should be 0 here; fix:
                ask_term = term2  # since p_ask_eff=0, depth*p=0

            # M[i,j] = p_bid_eff[i] * diff[j] + ask_term[j]
            M = np.outer(p_bid_eff, diff) + ask_term[None, :]

            # For each i, best j:
            idx_max = np.argmax(M, axis=1)
            best_val_i = M[np.arange(n_depths), idx_max] + p_bid_eff * depth_grid - cost

            best_i = np.argmax(best_val_i)
            best_val = best_val_i[best_i]
            best_bid = depth_grid[best_i]
            best_ask = depth_grid[idx_max[best_i]]

            V[n, qi] = best_val
            bid_table[n, qi] = best_bid
            ask_table[n, qi] = best_ask

    # Polish pass: refine depths around initial optima for higher precision
    if _polish_needed and time.time() - _START_TIME < TIME_LIMIT - 10:
        bid_table, ask_table = _polish_policy(params, bid_table, ask_table,
                                                depth_grid, p_fill, V, q_range,
                                                n_q, dt, k, A, sigma, gamma, T, steps, Q)
    
    return bid_table, ask_table


def _polish_policy(params, bid_table, ask_table, depth_grid, p_fill, V, q_range,
                   n_q, dt, k, A, sigma, gamma, T, steps, Q):
    """
    High-precision polish: for each (step, inventory), create a local fine grid
    around the initially-found optimal bid and ask depths, recompute the Bellman
    update, and update the policy.
    """
    n_depths = len(depth_grid)
    
    # Number of local points around each initial optimum
    n_local = 15
    # Ratio for local grid spacing (tighter than global)
    local_ratio = 1.1
    
    # We need to redo the DP with refined grids per state.
    # This is expensive, so we do it step-by-step from the end.
    V_refined = V.copy()
    bid_refined = bid_table.copy()
    ask_refined = ask_table.copy()
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            break
            
        inv_cost = (gamma / 2.0) * sigma**2 * dt
        Vn1 = V_refined[n + 1]
        
        for qi in range(n_q):
            q = q_range[qi]
            cost = inv_cost * q * q
            Vn1_q = Vn1[qi]
            
            if q == Q:
                p_bid_eff = np.zeros(n_depths)
            else:
                p_bid_eff = p_fill
                
            if q == -Q:
                p_ask_eff = np.zeros(n_depths)
            else:
                p_ask_eff = p_fill
                
            if q < Q:
                Vn1_q_plus = Vn1[qi + 1]
            else:
                Vn1_q_plus = Vn1_q
                
            if q > -Q:
                Vn1_q_minus = Vn1[qi - 1]
            else:
                Vn1_q_minus = Vn1_q
                
            if q == -Q:
                term1 = np.full(n_depths, Vn1_q_plus)
                term2 = np.full(n_depths, Vn1_q)
            else:
                term1 = p_ask_eff * Vn1_q + (1 - p_ask_eff) * Vn1_q_plus
                term2 = p_ask_eff * Vn1_q_minus + (1 - p_ask_eff) * Vn1_q
            
            diff = term1 - term2
            depth_p = depth_grid * p_fill
            
            if q == -Q:
                ask_term = term2
            else:
                ask_term = term2 + depth_p
            
            M = np.outer(p_bid_eff, diff) + ask_term[None, :]
            
            idx_max = np.argmax(M, axis=1)
            best_val_i = M[np.arange(n_depths), idx_max] + p_bid_eff * depth_grid - cost
            best_i = np.argmax(best_val_i)
            best_bid = depth_grid[best_i]
            best_ask = depth_grid[idx_max[best_i]]
            
            # Create local fine grids around best_bid and best_ask
            # Bid local grid
            bid_span = max(best_bid * 0.3, 0.01)
            bid_local = np.linspace(best_bid - bid_span, best_bid + bid_span, n_local)
            bid_local = np.clip(bid_local, 0.001, 10.0 / k)
            
            # Ask local grid
            ask_span = max(best_ask * 0.3, 0.01)
            ask_local = np.linspace(best_ask - ask_span, best_ask + ask_span, n_local)
            ask_local = np.clip(ask_local, 0.001, 10.0 / k)
            
            # Compute fill probabilities for local grids
            p_bid_local = 1.0 - np.exp(-A * np.exp(-k * bid_local) * dt)
            p_ask_local = 1.0 - np.exp(-A * np.exp(-k * ask_local) * dt)
            
            if q == Q:
                p_bid_local = np.zeros(n_local)
            if q == -Q:
                p_ask_local = np.zeros(n_local)
            
            # Recompute term1, term2 using local ask grid
            if q == -Q:
                term1_local = np.full(n_local, Vn1_q_plus)
                term2_local = np.full(n_local, Vn1_q)
            else:
                term1_local = p_ask_local * Vn1_q + (1 - p_ask_local) * Vn1_q_plus
                term2_local = p_ask_local * Vn1_q_minus + (1 - p_ask_local) * Vn1_q
            
            diff_local = term1_local - term2_local
            depth_p_local = ask_local * p_ask_local
            
            if q == -Q:
                ask_term_local = term2_local
            else:
                ask_term_local = term2_local + depth_p_local
            
            # Build local matrix: M_local[i,j] = p_bid_local[i] * diff_local[j] + ask_term_local[j]
            M_local = np.outer(p_bid_local, diff_local) + ask_term_local[None, :]
            
            idx_max_local = np.argmax(M_local, axis=1)
            best_val_local = M_local[np.arange(n_local), idx_max_local] + p_bid_local * bid_local - cost
            best_i_local = np.argmax(best_val_local)
            best_bid_local = bid_local[best_i_local]
            best_ask_local = ask_local[idx_max_local[best_i_local]]
            
            V_refined[n, qi] = best_val_local[best_i_local]
            bid_refined[n, qi] = best_bid_local
            ask_refined[n, qi] = best_ask_local
    
    return bid_refined, ask_refined


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    if time.time() - _START_TIME > TIME_LIMIT:
        return _heuristic_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    # Ensure valid values
    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
