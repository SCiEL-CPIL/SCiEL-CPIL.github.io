"""Optimal market making with inventory risk: analytic constant-depth policy."""
# EVOLVE-BLOCK-START
import numpy as np
import math

_POLICY_CACHE = {}


def _get_policy(params):
    """Compute (or fetch) the constant optimal bid/ask depth for a regime.

    In the small-dt limit the optimal depth is d* = (1/k)*ln(A*lam/nu),
    where lam = 1 + (gamma/2)*sigma^2*dt (marginal value of a bid fill)
    and nu  = 1 - (gamma/2)*sigma^2*dt (marginal value of an ask fill).
    It is independent of time and inventory; only the boundary q=+-Q
    disables one side.
    """
    key = (params["sigma"], params["A"], params["k"],
           params["gamma"], params["T"], params["steps"], params["Q"])
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    dt = params["T"] / params["steps"]

    risk = 0.5 * gamma * sigma * sigma * dt
    lam = 1.0 + risk
    nu = 1.0 - risk
    if nu <= 0.0:
        nu = 1e-12

    d = (1.0 / k) * math.log(A * lam / nu)
    if not (math.isfinite(d) and d > 0.0):
        d = 1.0 / k
    _POLICY_CACHE[key] = d
    return d


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (N,) for this step.

    Both sides quoted at the constant optimal depth d*, except the boundary
    inventory states where one side is disabled (np.inf).
    """
    n = len(inventory)
    d = _get_policy(params)
    Q = params["Q"]

    inv = np.asarray(inventory)
    bid = np.full(n, d, dtype=np.float64)
    ask = np.full(n, d, dtype=np.float64)

    # No bid fill allowed at q = Q (inventory at upper bound)
    bid[inv == Q] = np.inf
    # No ask fill allowed at q = -Q (inventory at lower bound)
    ask[inv == -Q] = np.inf

    return bid, ask
# EVOLVE-BLOCK-END
