"""
Market making with inventory risk: continuous-depth analytical policy.

Strategy: derive a closed-form quoting depth from the first-order condition
of the single-step Bellman equation, balancing expected fill profit against
inventory risk. This avoids any discrete depth grid and runs in O(N) per call.
"""
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # evaluator kills at 330s; keep margin
_TIME_MARGIN = 10.0  # seconds of safety margin


def _check_deadline():
    """Return True if we are past the safety deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - __TIME_MARGIN)


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (N,) with continuous
    analytical depths based on inventory risk.

    Derivation:
    At each step, the market maker chooses bid depth b and ask depth a.
    Fill probabilities:
        p_bid = 1 - exp(-A * exp(-k*b) * dt)
        p_ask = 1 - exp(-A * exp(-k*a) * dt)

    Expected immediate reward from fills:
        E[reward] = p_bid * b + p_ask * a  (each fill earns its depth)
    Inventory change: bid fill => q+1, ask fill => q-1
    Risk penalty: (gamma/2) * sigma^2 * q^2 * dt

    The Bellman first-order condition for optimal depth balances marginal
    fill probability gain against marginal inventory risk. For small dt,
    p ≈ A * exp(-k*d) * dt, so:
        E[immediate] ≈ A*dt*exp(-k*b)*b + A*dt*exp(-k*a)*a

    The risk cost of holding inventory q is (gamma/2)*sigma^2*q^2*dt.
    The marginal value of reducing inventory (via a fill) is approximately
    gamma * sigma^2 * q * dt (derivative of risk w.r.t. q).

    For the bid side: a bid fill increases inventory, so it's beneficial
    when the immediate profit b exceeds the marginal risk cost:
        A*dt*exp(-k*b)*b ≈ gamma*sigma^2*|q|*dt
    =>  A*exp(-k*b)*b ≈ gamma*sigma^2*|q|

    Similarly for ask: a fill decreases inventory, beneficial when profit
    exceeds marginal risk of the current position:
        A*exp(-k*a)*a ≈ gamma*sigma^2*|q|

    Solving A*exp(-k*d)*d = C where C = gamma*sigma^2*|q|:
        Let u = k*d, then A/k * u * exp(-u) = C
        u * exp(-u) = C*k/A
        Using Lambert W: u = -W(-C*k/A)
        d = -W(-C*k/A) / k

    For C=0 (q=0), the optimal is to quote as wide as possible (d->inf)
    to minimize risk, but we want to capture some profit. At q=0, there's
    no risk asymmetry, so we use a baseline depth.

    Actually, at q=0, the risk derivative is 0, so any depth is locally
    optimal for risk. We should maximize profit: d that maximizes
    A*dt*exp(-k*d)*d. The max of d*exp(-k*d) is at d=1/k.

    So:
        If |q| is small, use depth near 1/k (profit-maximizing).
        If |q| is large, use wider depth (risk-averse).

    Using the Lambert W solution:
        C = gamma * sigma^2 * |q|
        x = C * k / A
        If x < 1/e: two solutions; take the smaller d (closer to 1/k)
        If x >= 1/e: no real solution for the profit-maximizing branch,
                     meaning risk dominates; set depth to inf (no quote)
                     or a very large depth.

    Actually, let's reconsider. The objective is:
        sum of (depths earned) - sum of (gamma/2)*sigma^2*q^2*dt

    The depth earned is the distance from mid. A bid at depth b earns b
    when filled. The risk term penalizes large |q|.

    The correct first-order condition: the marginal benefit of quoting
    closer (smaller depth) is the increase in fill probability times the
    depth, minus the marginal risk cost.

    For simplicity and robustness, I'll use the Lambert W approach:
        d_opt = -W_0(-C*k/A) / k   where C = gamma*sigma^2*|q|

    When C*k/A > 1/e, no real solution exists on the principal branch,
    meaning the risk is too high to quote at any finite depth on that side.
    In that case, set depth to inf (no quote).

    For q=0: C=0, x=0, W(0)=0, d=0. That's wrong - we should quote at
    1/k. Let me handle q=0 separately: use d=1/k.

    Actually, let me re-derive more carefully. The single-step problem:
    Choose b, a >= 0.
    Reward = p_b * b + p_a * a - (gamma/2)*sigma^2*(q + delta_q)^2*dt
    where delta_q = +1 if bid fills, -1 if ask fills, 0 otherwise.
    p_b = 1 - exp(-A*exp(-k*b)*dt) ≈ A*exp(-k*b)*dt for small dt.
    p_a similarly.

    E[reward] ≈ A*dt*exp(-k*b)*b + A*dt*exp(-k*a)*a
                - (gamma/2)*sigma^2*dt * E[(q+delta_q)^2]

    E[(q+delta_q)^2] = p_b*(q+1)^2 + p_a*(q-1)^2 + (1-p_b-p_a)*q^2
    ≈ q^2 + 2q*(p_b - p_a) + (p_b + p_a)  [for small p]

    So E[reward] ≈ A*dt*exp(-k*b)*b + A*dt*exp(-k*a)*a
                  - (gamma/2)*sigma^2*dt*(q^2 + 2q*(p_b-p_a) + p_b+p_a)

    dE/db = A*dt*(-k*exp(-k*b)*b + exp(-k*b))
           - (gamma/2)*sigma^2*dt*2q*(-dp_b/db)  [since p_b appears in risk]
           - (gamma/2)*sigma^2*dt*dp_b/db

    dp_b/db = A*dt*(-k)*exp(-k*b) = -k*A*dt*exp(-k*b)

    dE/db = A*dt*exp(-k*b)*(1 - k*b)
           - (gamma/2)*sigma^2*dt*2q*k*A*dt*exp(-k*b)
           - (gamma/2)*sigma^2*dt*(-k*A*dt*exp(-k*b))

    Setting to 0:
    A*dt*exp(-k*b)*(1 - k*b) = (gamma/2)*sigma^2*dt*k*A*dt*exp(-k*b)*(2q - 1)

    Cancel A*dt*exp(-k*b) (assuming > 0):
    1 - k*b = (gamma/2)*sigma^2*dt*k*(2q - 1)

    k*b = 1 - (gamma/2)*sigma^2*dt*k*(2q - 1)
    b = 1/k - (gamma/2)*sigma^2*dt*(2q - 1)

    Similarly for ask:
    a = 1/k - (gamma/2)*sigma^2*dt*(1 - 2q)
       = 1/k + (gamma/2)*sigma^2*dt*(2q - 1)

    Wait, let me redo the ask side. For ask, delta_q = -1.
    E[(q+delta_q)^2] term: 2q*(p_b - p_a) + p_b + p_a
    dE/da: the p_a terms give:
    A*dt*exp(-k*a)*(1 - k*a)
    - (gamma/2)*sigma^2*dt*2q*(-dp_a/da)  [from 2q*(p_b-p_a), d/da of -p_a is -dp_a/da, times 2q]
    Wait, let me be more careful.

    Risk = (gamma/2)*sigma^2*dt*(q^2 + 2q*(p_b - p_a) + p_b + p_a)
    d(risk)/da = (gamma/2)*sigma^2*dt*(2q*(-dp_a/da) + dp_a/da)
                = (gamma/2)*sigma^2*dt*dp_a/da*(1 - 2q)

    dp_a/da = -k*A*dt*exp(-k*a)

    dE/da = A*dt*exp(-k*a)*(1 - k*a) - (gamma/2)*sigma^2*dt*(-k*A*dt*exp(-k*a))*(1-2q)
    = A*dt*exp(-k*a)*(1 - k*a) + (gamma/2)*sigma^2*dt*k*A*dt*exp(-k*a)*(1-2q)

    Setting to 0:
    1 - k*a = -(gamma/2)*sigma^2*dt*k*(1-2q)
    k*a = 1 + (gamma/2)*sigma^2*dt*k*(1-2q)
    a = 1/k + (gamma/2)*sigma^2*dt*(1-2q)

    So:
    b = 1/k - (gamma/2)*sigma^2*dt*(2q - 1)
    a = 1/k + (gamma/2)*sigma^2*dt*(1 - 2q)

    Note: 2q-1 and 1-2q are negatives of each other.
    Let delta = (gamma/2)*sigma^2*dt*(2q - 1)
    b = 1/k - delta
    a = 1/k + delta

    This makes sense: if q > 0 (long inventory), 2q-1 > 0 for q>=1, so
    delta > 0, b < 1/k (wider bid, less likely to fill, avoiding more long),
    a > 1/k (tighter ask, more likely to fill, reducing long inventory).

    If q < 0, delta < 0, b > 1/k (tighter bid), a < 1/k (wider ask).

    We need b, a > 0. If b <= 0, set to inf (no bid quote).
    If a <= 0, set to inf (no ask quote).

    Also, the constraint: no bid fill at q=Q, no ask fill at q=-Q.
    At q=Q, we should not quote bid (or it doesn't matter since no fill).
    Set b=inf at q=Q. At q=-Q, set a=inf.

    Let me also ensure depths are not too small (below -3/k is invalid per contract,
    but depths should be positive; the contract says values below -3/k make result
    invalid, but since we're quoting distances, they should be >= 0).

    Actually re-reading: "Values below -3/k, -inf, NaN or a wrong shape make the
    whole result invalid." So depths must be >= -3/k. Since depths are distances,
    they should be non-negative. I'll clip to a small positive value or inf.
    """
    N = len(inventory)
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    q = np.asarray(inventory, dtype=np.float64)

    # Base depth: 1/k
    base = 1.0 / k

    # Adjustment term
    delta = (gamma / 2.0) * (sigma ** 2) * dt * (2.0 * q - 1.0)

    bid_depth = base - delta
    ask_depth = base + delta

    # Handle boundary: no bid fill at q = Q, no ask fill at q = -Q
    bid_depth[q >= Q] = np.inf
    ask_depth[q <= -Q] = np.inf

    # Ensure depths are valid: must be >= -3/k, not -inf, not NaN
    # Depths should be positive; if <= 0, set to inf (no quote)
    bid_depth[bid_depth <= 0] = np.inf
    ask_depth[ask_depth <= 0] = np.inf

    # Replace any -inf or NaN with inf
    bid_depth[~np.isfinite(bid_depth) & (bid_depth < 0)] = np.inf
    ask_depth[~np.isfinite(ask_depth) & (ask_depth < 0)] = np.inf
    bid_depth[np.isnan(bid_depth)] = np.inf
    ask_depth[np.isnan(ask_depth)] = np.inf

    # Final safety: ensure shape and type
    bid_depth = np.asarray(bid_depth, dtype=np.float64)
    ask_depth = np.asarray(ask_depth, dtype=np.float64)

    return bid_depth, ask_depth
