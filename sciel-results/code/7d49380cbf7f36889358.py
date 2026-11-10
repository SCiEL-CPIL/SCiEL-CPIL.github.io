"""Rewritten program: nonlinear (log-linear) inventory skewing for market making."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _safe_depths(d_bid, d_ask, q, Q, k):
    """Clip depths to valid range and apply boundary conditions."""
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    d_bid = np.clip(d_bid, min_depth, max_depth)
    d_ask = np.clip(d_ask, min_depth, max_depth)
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)
    return d_bid.astype(np.float64), d_ask.astype(np.float64)


def quote(t, inventory, mid, params):
    if time.time() - _start_time > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k, dtype=np.float64)
        return depth, depth.copy()

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    dt = T / steps
    n = int(round(t / dt))
    n_rem = max(0, steps - n)

    q = np.asarray(inventory, dtype=np.float64)

    # ---- Effective horizon for inventory-risk (gamma) term ----
    # The inventory penalty is (gamma/2)*sigma^2*q^2*dt per step.
    # Its effect on the *current* quote depends on how long the position
    # persists, which is limited by both the remaining horizon and the
    # natural inventory-reversion time scale. We use an effective horizon
    # that caps at a few reversion periods to avoid over-skewing late.
    # Reversion scale: how many steps until inventory risk matters ~ one
    # unit of depth. Roughly 1/(gamma*sigma^2*dt/k).
    inv_rate = gamma * sigma * sigma * dt / k
    if inv_rate > 1e-12:
        n_eff = min(n_rem, 4.0 / inv_rate)
    else:
        n_eff = n_rem

    # ---- Nonlinear (log-linear) skew ----
    # The optimal depth shift for inventory q is approximately
    #   (1/k) * log( (1 + q*gamma*sigma^2*dt*eff) / (1 - q*gamma*sigma^2*dt*eff) )
    # which is the log-odds form arising from the discrete Bellman
    # balance. For small x it reduces to ~ (2/k)*q*gamma*sigma^2*dt*eff
    # (matching the linear term), but saturates gracefully for large q.
    x = q * gamma * sigma * sigma * dt * n_eff
    # Clip x to avoid log(0) and keep depths in valid range.
    x = np.clip(x, -0.9, 0.9)
    log_odds = np.log1p(x) - np.log1p(-x)  # = log((1+x)/(1-x))

    d0 = 1.0 / k
    skew = log_odds / k

    # ---- Fill-probability / spread term ----
    # When A*dt is small, the optimal spread widens because the value of
    # waiting (mid may move) outweighs the small fill chance. This adds a
    # symmetric widening to both sides.
    # Spread widening ~ (sigma^2 * dt * n_eff) / (2 * A * dt) * k * d0^2
    # Simplified: a modest symmetric widening proportional to remaining
    # horizon and volatility, damped by fill rate A.
    spread_widen = (sigma * sigma * dt * n_eff) / (2.0 * A * dt + 1e-12) * 0.25
    spread_widen = min(spread_widen, 2.0 / k)

    d_bid = d0 + skew + spread_widen
    d_ask = d0 - skew + spread_widen

    d_bid, d_ask = _safe_depths(d_bid, d_ask, q, Q, k)

    return d_bid, d_ask

# EVOLVE-BLOCK-END
