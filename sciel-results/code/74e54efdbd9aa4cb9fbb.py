"""Optimal market making via precomputed discrete-time Bellman DP (vectorized)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 300.0
_SAFE = 30.0  # seconds of safety margin before the hard kill

# Cache: key -> (bid_table, ask_table, dt)
# bid_table[step, q_idx] = bid depth; step 0 = last step (T - dt), step (steps-1) = first (t=0)
_CACHE = {}
_COMPUTED = set()


def _build_policy(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid/ask depth tables for all inventory states and times.

    Returns (bid_table, ask_table) each of shape (steps, 2*Q+1),
    indexed [step, q + Q] where step 0 corresponds to the last decision (t = T - dt).
    """
    dt = T / steps
    n_q = 2 * Q + 1
    q_idx = np.arange(n_q)
    q_vals = q_idx - Q  # inventory values -Q..Q
    q_sq = q_vals.astype(float) ** 2

    # Depth grid: log-spaced, high resolution for accuracy
    d_min = max(1e-4, 0.05 / k)
    d_max = 20.0 / k
    n_d = 500
    depth_grid = np.geomspace(d_min, d_max, n_d)
    exp_kd = np.exp(-k * depth_grid)          # (n_d,)
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    p_grid = 1.0 - np.exp(-A * exp_kd * dt)   # (n_d,)

    # Terminal value = 0
    V = np.zeros(n_q)

    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    # Iterate backwards in time: step index s = steps-1 (first decision) down to 0 (last)
    # We store table[s] = optimal depths when remaining time = (s+1)*dt
    for s in range(steps - 1, -1, -1):
        # V_next = V (value after this step's fills, i.e., next time step)
        V_next = V
        cost = (gamma / 2.0) * sigma ** 2 * q_sq * dt  # (n_q,)

        # --- Bid side ---
        # For state q: can fill bid only if q < Q
        # If bid depth d chosen:
        #   no fill (prob 1-p): value = V_next[q]
        #   fill (prob p): earn d, inventory q+1, value = d + V_next[q+1]
        #   expected = (1-p)*V_next[q] + p*(d + V_next[q+1]) = V_next[q] + p*(d + V_next[q+1] - V_next[q])
        #   total = -cost[q] + that
        # We want max over d in depth_grid.
        # Vectorize: for all q simultaneously.
        # V_next[q] is shape (n_q,); V_next[q+1] for q < Q is V_next[1:]
        # For q = Q: no bid fill allowed -> bid depth = inf (no quote), value = V_next[q]
        #   Actually if no bid fill at q=Q, the bid depth is irrelevant; set to inf.
        # For q < Q:
        #   benefit(d) = p(d) * (d + V_next[q+1] - V_next[q])
        #   We maximize benefit(d) over d.
        #   V_next[q+1] - V_next[q] is shape (n_q-1,) for q=0..Q-1
        delta_bid = V_next[1:] - V_next[:-1]  # (n_q-1,) for q = -Q..Q-1
        # benefit matrix: (n_q-1, n_d)
        benefit_bid = p_grid[None, :] * (depth_grid[None, :] + delta_bid[:, None])
        best_idx_bid = np.argmax(benefit_bid, axis=1)
        bid_depths = depth_grid[best_idx_bid]  # (n_q-1,) for q=-Q..Q-1

        # For q = Q: no bid quote
        bid_full = np.empty(n_q)
        bid_full[:-1] = bid_depths
        bid_full[-1] = np.inf  # q = Q: no bid

        # --- Ask side ---
        # For state q: can fill ask only if q > -Q
        # If ask depth d chosen:
        #   no fill: V_next[q]
        #   fill: earn d, inventory q-1, value = d + V_next[q-1]
        #   expected = V_next[q] + p*(d + V_next[q-1] - V_next[q])
        # For q = -Q: no ask fill -> ask depth = inf
        delta_ask = V_next[:-1] - V_next[1:]  # (n_q-1,) for q = -Q+1..Q
        benefit_ask = p_grid[None, :] * (depth_grid[None, :] + delta_ask[:, None])
        best_idx_ask = np.argmax(benefit_ask, axis=1)
        ask_depths = depth_grid[best_idx_ask]  # (n_q-1,) for q=-Q+1..Q

        ask_full = np.empty(n_q)
        ask_full[1:] = ask_depths
        ask_full[0] = np.inf  # q = -Q: no ask

        # Update value: V[q] = -cost[q] + max_bid_value + max_ask_value
        # max_bid_value for q < Q: V_next[q] + benefit_bid[q, best_idx]
        # For q = Q: V_next[q] (no bid)
        bid_val = np.empty(n_q)
        bid_val[:-1] = V_next[:-1] + benefit_bid[np.arange(n_q - 1), best_idx_bid]
        bid_val[-1] = V_next[-1]

        ask_val = np.empty(n_q)
        ask_val[1:] = V_next[1:] + benefit_ask[np.arange(n_q - 1), best_idx_ask]
        ask_val[0] = V_next[0]

        V = -cost + bid_val + ask_val

        bid_table[s] = bid_full
        ask_table[s] = ask_full

    return bid_table, ask_table


def _get_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key in _CACHE:
        return _CACHE[key]
    if key in _COMPUTED:
        return None
    _COMPUTED.add(key)
    try:
        bid_t, ask_t = _build_policy(
            params["sigma"], params["A"], params["k"], params["gamma"],
            params["T"], params["steps"], params["Q"]
        )
        _CACHE[key] = (bid_t, ask_t)
        return _CACHE[key]
    except Exception:
        return None


# Precompute known regimes at import time
_KNOWN_REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]
_COMMON = {"T": 1.0, "steps": 1000, "Q": 30}

for _sig, _A, _k, _gam in _KNOWN_REGIMES:
    if time.time() + _SAFE >= _DEADLINE:
        break
    _p = {"sigma": _sig, "A": _A, "k": _k, "gamma": _gam, **_COMMON}
    _get_policy(_p)


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (len(inventory),)."""
    n = len(inventory)
    steps = params["steps"]
    Q = params["Q"]

    pol = _get_policy(params)
    if pol is None:
        # Fallback: seed policy
        depth = 1.0 / params["k"]
        d = np.full(n, depth)
        return d, d.copy()

    bid_table, ask_table = pol

    # Determine step index: t = n*dt, dt = T/steps, so step_num = round(t/dt)
    # Remaining decisions = steps - step_num
    # Our table is indexed [s] where s = steps-1-step_num (0 = last decision)
    dt = params["T"] / steps
    step_num = int(round(t / dt))
    step_num = max(0, min(steps - 1, step_num))
    s = steps - 1 - step_num

    # inventory values: clip to [-Q, Q] for safety
    inv = np.clip(inventory, -Q, Q)
    q_idx = (inv + Q).astype(int)

    bid_depths = bid_table[s, q_idx]
    ask_depths = ask_table[s, q_idx]

    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
