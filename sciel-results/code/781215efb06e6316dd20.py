"""Market-making with inventory risk: parametric quadratic-in-inventory depth policy,
tuned per regime via fast Monte-Carlo policy search (much cheaper than full backward induction)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_WALL_LIMIT = 300.0
_SAFETY_MARGIN = 15.0  # seconds before hard wall to stop and return best

# Per-regime tuning cache: key -> (d0, alpha, beta)
_TUNED = {}

# Global RNG for policy search (independent of evaluator path RNG)
_search_rng = np.random.RandomState(12345)

# Valid depth bounds (must be > -3/k, finite, non-negative practically)
_MIN_DEPTH = 1e-3


def _regime_key(params):
    return (round(float(params["sigma"]), 6), round(float(params["A"]), 6),
            round(float(params["k"]), 6), round(float(params["gamma"]), 6))


def _simulate_policy(params, d0, alpha, beta, n_paths, seed):
    """Run the discrete model under the quadratic policy; return mean objective."""
    dt = params["T"] / params["steps"]
    T = params["T"]
    steps = params["steps"]
    Q = int(params["Q"])
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    inv_sigma = sigma * np.sqrt(dt)

    rng = np.random.RandomState(seed)
    n_paths = int(n_paths)
    inv = np.zeros(n_paths, dtype=np.int32)
    obj = np.zeros(n_paths, dtype=np.float64)

    for _ in range(steps):
        q = inv.astype(np.float64)
        # objective penalty (inventory before fills)
        obj -= (gamma / 2.0) * sigma * sigma * q * q * dt

        # policy depths
        depths = d0 + alpha * q + beta * q * q
        depths = np.clip(depths, _MIN_DEPTH, None)
        d_b = depths
        d_a = depths

        # fill probabilities
        p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)

        # no bid fill at q=Q, no ask fill at q=-Q
        p_b = np.where(q >= Q, 0.0, p_b)
        p_a = np.where(q <= -Q, 0.0, p_a)

        u_b = rng.random_sample(n_paths)
        u_a = rng.random_sample(n_paths)
        fill_b = (u_b < p_b) & (q < Q)
        fill_a = (u_a < p_a) & (q > -Q)

        # earned depths: bid fill earns d_b, ask fill earns d_a
        obj += np.where(fill_b, d_b, 0.0) + np.where(fill_a, d_a, 0.0)

        # inventory update
        inv = inv + fill_b.astype(np.int32) - fill_a.astype(np.int32)
        # clamp (safety)
        inv = np.clip(inv, -Q, Q)

    return float(np.mean(obj))


def _tune_for_regime(params, deadline):
    """Grid + local search over (d0, alpha, beta) maximizing mean objective."""
    key = _regime_key(params)
    if key in _TUNED:
        return _TUNED[key]

    k = params["k"]
    # base neutral depth
    d_neutral = 1.0 / k
    sigma = params["sigma"]
    gamma = params["gamma"]
    A = params["A"]
    Q = params["Q"]

    # Grid over the three parameters. Keep ranges modest for speed.
    d0_grid = np.array([0.5 * d_neutral, 0.75 * d_neutral, d_neutral,
                        1.25 * d_neutral, 1.5 * d_neutral, 2.0 * d_neutral])
    # alpha scales with risk penalty: larger gamma/sigma -> tighter when inventory high
    scale_a = (gamma * sigma * sigma) / (A * k)  # rough risk scale
    alpha_grid = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]) * scale_a
    beta_grid = np.array([0.0, 0.2, 0.5, 1.0, 2.0]) * scale_a * 0.1

    n_paths = 400  # small for speed during grid
    seed_base = 777

    best = None
    best_val = -np.inf
    # Limit grid size for wall-clock safety
    for d0 in d0_grid:
        for al in alpha_grid:
            for be in beta_grid:
                if time.time() > deadline - _SAFETY_MARGIN:
                    break
                val = _simulate_policy(params, d0, al, be, n_paths, seed_base)
                if val > best_val:
                    best_val = val
                    best = (float(d0), float(al), float(be))

    if best is None:
        best = (float(d_neutral), 0.0, 0.0)

    # Local refinement around best with more paths
    d0, al, be = best
    # small local grid
    d0_local = d0 + np.array([-0.25, -0.1, 0.0, 0.1, 0.25]) * d_neutral
    al_local = al + np.array([-0.25, -0.1, 0.0, 0.1, 0.25]) * max(abs(al), 1e-6)
    be_local = be + np.array([-0.25, -0.1, 0.0, 0.1, 0.25]) * max(abs(be), 1e-6)

    n_paths2 = 800
    for d0l in d0_local:
        for al_l in al_local:
            for be_l in be_local:
                if time.time() > deadline - _SAFETY_MARGIN:
                    break
                val = _simulate_policy(params, d0l, al_l, be_l, n_paths2, seed_base + 1)
                if val > best_val:
                    best_val = val
                    best = (float(d0l), float(al_l), float(be_l))

    _TUNED[key] = best
    return best


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (n_paths,) for this step."""
    deadline = _START_TIME + _WALL_LIMIT
    inventory = np.asarray(inventory, dtype=np.int64)
    n = inventory.shape[0]

    key = _regime_key(params)
    if key not in _TUNED:
        # If we still have time, tune; otherwise fall back to neutral.
        if time.time() < deadline - _SAFETY_MARGIN:
            _tune_for_regime(params, deadline)
        else:
            _TUNED[key] = (1.0 / params["k"], 0.0, 0.0)

    d0, alpha, beta = _TUNED[key]

    q = inventory.astype(np.float64)
    depths = d0 + alpha * q + beta * q * q
    depths = np.clip(depths, _MIN_DEPTH, None)
    depths = np.where(np.isfinite(depths), depths, _MIN_DEPTH)

    bid = depths.astype(np.float64)
    ask = depths.astype(np.float64)
    return bid, ask
# EVOLVE-BLOCK-END
