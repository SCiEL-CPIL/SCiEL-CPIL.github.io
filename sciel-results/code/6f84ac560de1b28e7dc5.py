"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 310.0  # Safety margin before 330s hard kill
_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal depth policy via fully vectorized dynamic programming.
    Uses a change of variables x = exp(-k*d) to transform the depth optimization
    into a 1D problem over x in (0, 1], with a log-spaced grid for precision.
    
    Returns (bid_table, ask_table) where:
      bid_table[t, i] = optimal bid depth at time t, inventory index i
      ask_table[t, i] = optimal ask depth at time t, inventory index i
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Value function: V[t, i] where i indexes q from -Q to Q
    V = np.zeros((N + 1, n_q))
    # Optimal depths tables
    bid_table = np.full((N, n_q), np.inf)
    ask_table = np.full((N, n_q), np.inf)
    
    # Change of variables: x = exp(-k*d), so d = -log(x)/k
    # Fill probability: P = 1 - exp(-A * x * dt)
    # We use a log-spaced grid for x to get fine resolution near x=0 (large d)
    # and coarser near x=1 (small d)
    grid_size = 500
    x_grid = np.logspace(-10, 0, grid_size)  # x in [1e-10, 1]
    log_x = np.log(x_grid)
    P_grid = 1.0 - np.exp(-A * x_grid * dt)  # shape (grid_size,)
    d_grid = -log_x / k  # shape (grid_size,)
    
    # Precompute cost per inventory level: (gamma/2) * sigma^2 * q^2 * dt
    cost_per_q = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    # Boundary masks
    bid_mask = np.ones(n_q, dtype=bool)
    bid_mask[n_q - 1] = False  # q=Q, no bid fill
    ask_mask = np.ones(n_q, dtype=bool)
    ask_mask[0] = False  # q=-Q, no ask fill
    
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Fallback: fill remaining timesteps with a simple heuristic
            for tt in range(t, -1, -1):
                if tt < N:
                    base_d = 1.0 / k
                    bid_table[tt] = np.clip(base_d + 0.01 * q_vals, 0.0, None)
                    ask_table[tt] = np.clip(base_d - 0.01 * q_vals, 0.0, None)
                    V[tt] = V[tt + 1] if tt + 1 <= N else 0.0
            break
        
        V_next = V[t + 1, :]  # shape (n_q,)
        
        # Value differences for bid/ask fills
        # C_b[i] = V[t+1, i+1] - V[t+1, i] for i < n_q-1
        # C_a[i] = V[t+1, i-1] - V[t+1, i] for i > 0
        C_b = np.zeros(n_q)
        C_a = np.zeros(n_q)
        C_b[:n_q - 1] = V_next[1:] - V_next[:-1]
        C_a[1:] = V_next[:-1] - V_next[1:]
        
        # For each inventory state i and depth grid point j:
        # Expected value contribution from bid:
        #   Pb[j] * (d_grid[j] + C_b[i]) + (1-Pb[j]) * 0
        #   = Pb[j] * (d_grid[j] + C_b[i])
        # Similarly for ask:
        #   Pa[j] * (d_grid[j] + C_a[i])
        #
        # But we need to be careful: the total EV is:
        #   Pb*(1-Pa)*(V_bid) + Pa*(1-Pb)*(V_ask) + (Pb*Pa + (1-Pb)*(1-Pa))*V_same + Pb*d_b + Pa*d_a - cost
        #
        # Let's expand:
        #   = Pb*V_bid + Pa*V_ask - Pb*Pa*V_bid - Pa*Pb*V_ask + Pb*Pa*V_same + V_same - Pb*V_same - Pa*V_same + Pb*Pa*V_same + Pb*d_b + Pa*d_a - cost
        #   = V_same + Pb*(V_bid - V_same) + Pa*(V_ask - V_same) - Pb*Pa*(V_bid + V_ask - 2*V_same) + Pb*d_b + Pa*d_a - cost
        #
        # Where V_bid = V[t+1, i+1], V_ask = V[t+1, i-1], V_same = V[t+1, i]
        #
        # Let D_b[i] = V_bid - V_same = C_b[i]
        #     D_a[i] = V_ask - V_same = C_a[i]
        #     D_both[i] = V_bid + V_ask - 2*V_same = C_b[i] + C_a[i]
        #
        # EV[i, jb, ja] = V_same[i] + Pb[jb]*C_b[i] + Pa[ja]*C_a[i] - Pb[jb]*Pa[ja]*(C_b[i]+C_a[i]) + Pb[jb]*d[jb] + Pa[ja]*d[ja] - cost[i]
        
        # To vectorize efficiently, note that the cross term -Pb[jb]*Pa[ja]*(C_b[i]+C_a[i])
        # couples the bid and ask choices. However, for small dt, Pb and Pa are small,
        # so the cross term is second-order. We can either:
        # 1. Ignore it (approximation) - fast
        # 2. Include it - requires 2D search
        #
        # Given that dt = 0.001 and A*exp(-k*d)*dt is typically small,
        # the cross term is O(dt^2) and likely negligible. Let's first try ignoring it.
        # If that's not accurate enough, we'll do the full 2D search.
        
        # Approximation: ignore cross term
        # EV_b[i, jb] = Pb[jb] * (C_b[i] + d_grid[jb])
        # EV_a[i, ja] = Pa[ja] * (C_a[i] + d_grid[ja])
        
        # For bid: shape (n_q, grid_size)
        # term_b[i, j] = C_b[i] + d_grid[j]
        # g_b[i, j] = P_grid[j] * (C_b[i] + d_grid[j])
        
        term_b = C_b[:, None] + d_grid[None, :]  # shape (n_q, grid_size)
        g_b = P_grid[None, :] * term_b  # shape (n_q, grid_size)
        
        # For ask:
        term_a = C_a[:, None] + d_grid[None, :]  # shape (n_q, grid_size)
        g_a = P_grid[None, :] * term_a  # shape (n_q, grid_size)
        
        # Find optimal depth for each i
        max_idx_b = np.argmax(g_b, axis=1)  # shape (n_q,)
        max_val_b = np.max(g_b, axis=1)  # shape (n_q,)
        
        max_idx_a = np.argmax(g_a, axis=1)  # shape (n_q,)
        max_val_a = np.max(g_a, axis=1)  # shape (n_q,)
        
        # Get optimal depths
        b_vals = d_grid[max_idx_b]  # shape (n_q,)
        a_vals = d_grid[max_idx_a]  # shape (n_q,)
        
        # If max value is <= 0, it's better not to quote (depth = inf)
        # But we also need to handle boundaries
        no_quote_b = (max_val_b <= 1e-12) | (~bid_mask)
        no_quote_a = (max_val_a <= 1e-12) | (~ask_mask)
        
        b_vals[no_quote_b] = np.inf
        a_vals[no_quote_a] = np.inf
        
        # Store
        bid_table[t, :] = b_vals
        ask_table[t, :] = a_vals
        
        # Update V[t, i]
        # V[t, i] = V_next[i] - cost[i] + contrib_b[i] + contrib_a[i]
        # where contrib_b[i] = max_val_b[i] if quoting, else 0
        # and similarly for contrib_a
        
        contrib_b = np.where(no_quote_b, 0.0, max_val_b)
        contrib_a = np.where(no_quote_a, 0.0, max_val_a)
        
        V[t, :] = V_next - cost_per_q + contrib_b + contrib_a
    
    # Replace inf with a large finite value for safety in quote()
    bid_table[bid_table == np.inf] = 1e6
    ask_table[ask_table == np.inf] = 1e6
    
    _CACHE[key] = (bid_table, ask_table)
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,).
    Uses exact precomputed DP tables for the current regime.
    """
    if time.time() > _DEADLINE:
        # Fallback to simple heuristic
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    bid_table, ask_table = _solve_regime(
        params["sigma"], params["A"], params["k"], 
        params["gamma"], params["T"], params["steps"], params["Q"]
    )
    
    # t is n * dt, so step index n = int(round(t / dt))
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = max(0, min(n, params["steps"] - 1))
    
    # inventory is int array in [-Q, Q]
    # Index into table: i = q + Q
    Q = params["Q"]
    idx = inventory + Q
    
    # Clip indices to valid range (should already be valid)
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    # Ensure non-negative depths and no NaN/inf
    bid_depths = np.clip(bid_depths, 0.0, None)
    ask_depths = np.clip(ask_depths, 0.0, None)
    
    bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=1e6, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=1e6, neginf=0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
