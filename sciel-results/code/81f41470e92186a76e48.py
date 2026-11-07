"""Optimal market making with inventory risk: precomputed HJB backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 300s limit

# Precomputed HJB tables keyed by (sigma, A, k, gamma, T, steps, Q)
_HJB_CACHE = {}


def _solve_hjb(sigma, A, k, gamma, T, steps, Q, deadline):
    """Solve the discrete HJB via backward induction. Returns (bid_table, ask_table).

    bid_table[n, q] = optimal bid depth at step n, inventory q (q in [-Q, Q]).
    ask_table[n, q] = optimal ask depth at step n, inventory q.
    """
    nQ = 2 * Q + 1
    # Value function V[n, q] for n in [0, steps], q in [-Q, Q]
    V = np.zeros((steps + 1, nQ), dtype=np.float64)
    bid_table = np.zeros((steps, nQ), dtype=np.float64)
    ask_table = np.zeros((steps, nQ), dtype=np.float64)

    dt = T / steps
    inv_k = 1.0 / k

    # Depth grid for optimization
    # The optimal depth is typically in [0, ~5/k]. Use a fine grid.
    d_max = max(10.0 * inv_k, 5.0)
    N_D = 2000
    depth_grid = np.linspace(0.0, d_max, N_D)
    # Precompute p(d) = 1 - exp(-A exp(-k d) dt) on the grid
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with 1/k
            bid_table[n:] = inv_k
            ask_table[n:] = inv_k
            break

        Vn = V[n + 1]
        # Vn_qm1[i] = V[n+1, q_i - 1] where q_i = i - Q
        # For i=0 (q=-Q), q-1 = -Q-1 which is out of bounds, but we don't use bid at q=-Q... wait
        # Actually, bid is only relevant when q < Q. If q < Q, a bid fill moves q to q-1.
        # If q = -Q, q-1 = -Q-1 is invalid. But the problem says "no bid fill at q = Q".
        # Does it say no bid fill at q = -Q? No. It says "no ask fill at q = -Q".
        # So at q = -Q, a bid fill is possible, moving q to -Q-1?
        # Wait, inventory is in [-Q, Q]. If q = -Q and a bid fill happens, q becomes -Q-1?
        # The problem states: "inventory (4000,) ints in [-Q, Q]".
        # "no bid fill at q = Q, no ask fill at q = -Q".
        # This implies that if q < Q, a bid fill is possible. If q = -Q, q < Q is true (since -Q < Q for Q>0).
        # So a bid fill at q=-Q would make q = -Q-1. But the state space is [-Q, Q].
        # Usually in such models, the boundaries are absorbing or the fill is prevented.
        # Let's re-read carefully: "no bid fill at q = Q, no ask fill at q = -Q".
        # It does NOT say "no bid fill at q = -Q".
        # However, if q becomes -Q-1, it's outside the defined range.
        # Standard interpretation: The state space is bounded. If a fill would push it out, it's either blocked or the state is clipped.
        # Given "no bid fill at q=Q" prevents going above Q.
        # Does "no ask fill at q=-Q" prevent going below -Q? Yes.
        # What about bid fill at q=-Q? It would go to -Q-1.
        # Ask fill at q=Q? It would go to Q+1.
        # The constraints only mention the "wrong" direction fills at the boundaries.
        # "no bid fill at q = Q" -> prevents q from increasing to Q+1.
        # "no ask fill at q = -Q" -> prevents q from decreasing to -Q-1.
        # This implies:
        # - At q=Q, bid is disabled. Ask is enabled (moves to Q+1? No, Q is max. Wait.)
        # Let's check the direction.
        # Bid fill: Customer buys from us? Or we buy?
        # "bid_depths ... distances below mid".
        # If we quote a bid, we are selling. So we give an asset. Inventory decreases?
        # Or is inventory the number of assets we hold?
        # "A fill earns its depth and moves inventory one unit".
        # Usually:
        # Bid quote (sell): Inventory decreases by 1.
        # Ask quote (buy): Inventory increases by 1.
        # Let's verify with "no bid fill at q=Q".
        # If q=Q (max inventory), and we sell (bid fill), q becomes Q-1. This is valid.
        # So why "no bid fill at q=Q"?
        # Maybe the convention is:
        # Bid fill -> Inventory increases? (We are short? No, market maker holds inventory).
        # Let's look at the penalty: gamma/2 sigma^2 q^2. Symmetric.
        # Let's assume standard MM:
        # Bid = Sell. Inventory q -> q-1.
        # Ask = Buy. Inventory q -> q+1.
        # If q=Q (max), we don't want to buy more? So "no ask fill at q=Q"?
        # But the text says "no bid fill at q=Q".
        # This implies Bid fill increases inventory?
        # If Bid fill increases inventory:
        # q=Q, bid fill -> Q+1 (invalid). So no bid fill at Q.
        # q=-Q, ask fill -> -Q-1 (invalid). So no ask fill at -Q.
        # This matches the text perfectly!
        # So:
        # Bid fill: q -> q + 1.
        # Ask fill: q -> q - 1.
        # Wait, let's re-read "bid_depths ... distances below mid".
        # If I quote a bid below mid, I am offering to sell at a lower price.
        # If I sell, I should have less inventory.
        # Why would the text say "no bid fill at q=Q" if selling reduces inventory?
        # Unless "inventory" means "short position" or something?
        # Or maybe I have the direction of q wrong.
        # Let's look at the objective: "depths earned".
        # If I sell at bid depth d, I earn d? "A fill earns its depth".
        # If I sell, I get mid - d. I had the asset.
        # Let's assume the text is literal:
        # "no bid fill at q = Q"
        # "no ask fill at q = -Q"
        # This strongly implies:
        # Bid fill increases q.
        # Ask fill decreases q.
        # Let's proceed with this assumption.
        # Bid fill: q -> q+1.
        # Ask fill: q -> q-1.

        # Vn_qm1 is not needed for bid if bid increases q.
        # Bid at q leads to q+1.
        # Ask at q leads to q-1.

        # Let's define:
        # V_next_plus[i] = V[n+1, q_i + 1]
        # V_next_minus[i] = V[n+1, q_i - 1]

        # For q < Q: Bid is allowed. Next state q+1.
        # For q > -Q: Ask is allowed. Next state q-1.
        # For q = Q: Bid not allowed.
        # For q = -Q: Ask not allowed.

        # Indices:
        # q_i = i - Q
        # q_i + 1 corresponds to index i+1.
        # q_i - 1 corresponds to index i-1.

        Vn_qp1 = np.zeros(nQ, dtype=np.float64)  # V[n+1, q+1]
        Vn_qm1 = np.zeros(nQ, dtype=np.float64)  # V[n+1, q-1]

        # Vn_qp1[i] = V[n+1, i+1] for i < nQ-1. For i=nQ-1 (q=Q), q+1 is out of bounds, but bid is disabled there.
        Vn_qp1[:-1] = Vn[1:]
        Vn_qp1[-1] = 0.0  # Dummy

        # Vn_qm1[i] = V[n+1, i-1] for i > 0. For i=0 (q=-Q), q-1 is out of bounds, but ask is disabled there.
        Vn_qm1[1:] = Vn[:-1]
        Vn_qm1[0] = 0.0  # Dummy

        penalty = 0.5 * gamma * sigma * sigma * q_vals * q_vals * dt

        bid_row = np.zeros(nQ, dtype=np.float64)
        ask_row = np.zeros(nQ, dtype=np.float64)

        # --- Optimize bid depth for each q ---
        # Bid is allowed for q < Q (indices 0 to nQ-2).
        # Objective: d*p(d) + p(d)*V[n+1, q+1] + (1-p(d))*V[n+1, q]
        # = V[n+1, q] + p(d) * (d + V[n+1, q+1] - V[n+1, q])
        # Let C_bid[q] = V[n+1, q+1] - V[n+1, q]
        # Maximize p(d) * (d + C_bid[q])

        C_bid = Vn_qp1[:nQ-1] - Vn[:nQ-1]  # Shape (nQ-1,)
        # Process in batches
        batch = 64
        for s in range(0, nQ - 1, batch):
            e = min(s + batch, nQ - 1)
            c = C_bid[s:e]
            # obj[d, b] = p_grid[d] * (depth_grid[d] + c[b])
            obj = p_grid[:, None] * (depth_grid[:, None] + c[None, :])
            idx = np.argmax(obj, axis=0)
            # Parabolic refinement
            ref = np.empty(e - s, dtype=np.float64)
            for j in range(e - s):
                i_best = idx[j]
                if i_best > 0 and i_best < N_D - 1:
                    f0, f1, f2 = obj[i_best-1, j], obj[i_best, j], obj[i_best+1, j]
                    denom = f0 - 2.0*f1 + f2
                    if abs(denom) > 1e-18:
                        delta = 0.5 * (f0 - f2) / denom
                        if -1.0 < delta < 1.0:
                            d_ref = depth_grid[i_best-1] + delta * (depth_grid[i_best] - depth_grid[i_best-1])
                            # Ensure d_ref is within [0, d_max]
                            if 0.0 <= d_ref <= d_max:
                                ref[j] = d_ref
                                continue
                ref[j] = depth_grid[i_best]
            bid_row[s:e] = ref

        # --- Optimize ask depth for each q ---
        # Ask is allowed for q > -Q (indices 1 to nQ-1).
        # Objective: d*p(d) + p(d)*V[n+1, q-1] + (1-p(d))*V[n+1, q]
        # = V[n+1, q] + p(d) * (d + V[n+1, q-1] - V[n+1, q])
        # Let C_ask[q] = V[n+1, q-1] - V[n+1, q]
        # Maximize p(d) * (d + C_ask[q])

        C_ask = Vn_qm1[1:] - Vn[1:]  # Shape (nQ-1,)
        for s in range(1, nQ, batch):
            e = min(s + batch, nQ)
            c = C_ask[s-1:e-1]
            obj = p_grid[:, None] * (depth_grid[:, None] + c[None, :])
            idx = np.argmax(obj, axis=0)
            ref = np.empty(e - s, dtype=np.float64)
            for j in range(e - s):
                i_best = idx[j]
                if i_best > 0 and i_best < N_D - 1:
                    f0, f1, f2 = obj[i_best-1, j], obj[i_best, j], obj[i_best+1, j]
                    denom = f0 - 2.0*f1 + f2
                    if abs(denom) > 1e-18:
                        delta = 0.5 * (f0 - f2) / denom
                        if -1.0 < delta < 1.0:
                            d_ref = depth_grid[i_best-1] + delta * (depth_grid[i_best] - depth_grid[i_best-1])
                            if 0.0 <= d_ref <= d_max:
                                ref[j] = d_ref
                                continue
                ref[j] = depth_grid[i_best]
            ask_row[s:e] = ref

        bid_table[n] = bid_row
        ask_table[n] = ask_row

        # --- Update V[n, q] ---
        bid_contrib = np.zeros(nQ, dtype=np.float64)
        ask_contrib = np.zeros(nQ, dtype=np.float64)

        # For q < Q (indices 0 to nQ-2):
        p_b = 1.0 - np.exp(-A * np.exp(-k * bid_row[:nQ-1]) * dt)
        bid_contrib[:nQ-1] = (p_b * (bid_row[:nQ-1] + Vn_qp1[:nQ-1]) + (1.0 - p_b) * Vn[:nQ-1])

        # For q > -Q (indices 1 to nQ-1):
        p_a = 1.0 - np.exp(-A * np.exp(-k * ask_row[1:]) * dt)
        ask_contrib[1:] = (p_a * (ask_row[1:] + Vn_qm1[1:]) + (1.0 - p_a) * Vn[1:])

        V[n] = penalty + bid_contrib + ask_contrib

    return bid_table, ask_table


def _get_tables(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key in _HJB_CACHE:
        return _HJB_CACHE[key]
    deadline = _DEADLINE
    tables = _solve_hjb(params["sigma"], params["A"], params["k"],
                        params["gamma"], params["T"], params["steps"],
                        params["Q"], deadline)
    _HJB_CACHE[key] = tables
    return tables


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (N,) from precomputed HJB."""
    n = int(round(t / (params["T"] / params["steps"])))
    steps = params["steps"]
    Q = params["Q"]
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0

    bid_table, ask_table = _get_tables(params)

    # Map inventory q in [-Q, Q] to index q + Q, clipped
    q = np.asarray(inventory, dtype=np.int64)
    idx = np.clip(q + Q, 0, 2 * Q)

    bid = bid_table[n, idx].astype(np.float64)
    ask = ask_table[n, idx].astype(np.float64)

    # Enforce boundary:
    # no bid fill at q = Q -> bid depth 0 (irrelevant but safe)
    # no ask fill at q = -Q -> ask depth 0
    bid = np.where(q >= Q, 0.0, bid)
    ask = np.where(q <= -Q, 0.0, ask)

    # Clamp to valid range: must be >= -3/k (task says below -3/k invalid), and finite
    min_allowed = -3.0 / params["k"]
    bid = np.maximum(bid, min_allowed)
    ask = np.maximum(ask, min_allowed)
    bid = np.where(np.isfinite(bid), bid, min_allowed)
    ask = np.where(np.isfinite(ask), ask, min_allowed)

    return bid, ask
# EVOLVE-BLOCK-END
