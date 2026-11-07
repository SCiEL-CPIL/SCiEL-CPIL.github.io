"""Optimal market making with inventory risk via analytic HJB approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

# Cache for pre-computed parameters
# Key: (sigma, A, k, gamma, Q, steps)
# Value: dict with 'inv_scale' and 'base_depth'
_PARAMS_CACHE = {}


def _get_params(params):
    """
    Pre-compute scaling factors for the analytic solution.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _PARAMS_CACHE:
        return _PARAMS_CACHE[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    dt = T / steps

    # Base depth for risk-neutral case: max of d * exp(-k*d) is at d = 1/k
    base_depth = 1.0 / k

    # Inventory sensitivity
    # The marginal risk cost is ~ gamma * sigma^2 * q * dt
    # We want to adjust depth such that the change in fill probability offsets this.
    # Fill prob P ~ A * exp(-k*d) * dt
    # dP/dd = -k * P
    # We want dP/dd * dd = -Risk_Cost
    # -k * P * dd = -gamma * sigma^2 * q * dt
    # dd = (gamma * sigma^2 * q * dt) / (k * P)
    # This is nonlinear. A linear approximation around q=0, d=1/k:
    # P(1/k) = A * exp(-1) * dt
    # dd ~ (gamma * sigma^2 * q * dt) / (k * A * exp(-1) * dt) = (gamma * sigma^2 * q * e) / (k * A)
    # Let's use a simpler scaling that has been effective in similar tasks:
    # inv_scale = gamma * sigma^2 * dt / (A * k)
    
    inv_scale = (gamma * sigma * sigma * dt) / (A * k)
    
    _PARAMS_CACHE[key] = {'base_depth': base_depth, 'inv_scale': inv_scale, 'k': k}
    return _PARAMS_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) for each path.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a safe, simple quote
        k = params["k"]
        d = 1.0 / k
        return np.full_like(inventory, d, dtype=np.float64), np.full_like(inventory, d, dtype=np.float64)

    p = _get_params(params)
    base_depth = p['base_depth']
    inv_scale = p['inv_scale']
    k = p['k']
    Q = params["Q"]
    
    # inventory is an array of ints
    # Convert to float for calculation
    q = inventory.astype(np.float64)
    
    # Calculate depths
    # d_bid = base_depth + inv_scale * q
    # d_ask = base_depth - inv_scale * q
    
    bid_depths = base_depth + inv_scale * q
    ask_depths = base_depth - inv_scale * q
    
    # Ensure depths are positive and valid
    # Minimum depth to avoid -inf or NaN issues and respect -3/k bound
    min_depth = 1e-6
    # The task says values below -3/k are invalid. Since depths are distances, they should be positive.
    # However, if the formula yields a negative value, it means we should quote very tightly (near 0) or not at all?
    # If d < 0, it implies the risk cost is so high we shouldn't quote on that side.
    # But we must return a float. np.inf is allowed for "no quote".
    # However, the prompt says "distances below and above mid... np.inf means no quote".
    # If the optimal depth is negative, it's better to set it to np.inf (no quote) or a very small positive value?
    # Usually, if the marginal cost > marginal benefit, you don't quote.
    # Let's set negative depths to np.inf.
    
    bid_depths = np.where(bid_depths < min_depth, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < min_depth, np.inf, ask_depths)
    
    # Handle inventory boundaries
    # No bid fill at q = Q -> set bid to inf
    # No ask fill at q = -Q -> set ask to inf
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    # Final validation: replace NaN and -inf
    # The task says "Values below -3/k, -inf, NaN ... make the whole result invalid"
    # So we must ensure no NaN, no -inf, and no values < -3/k.
    # Since we use np.inf for no quote, that is allowed.
    # We just need to clean up any potential numerical artifacts.
    
    min_valid = -3.0 / k
    # If any value is < min_valid, it's invalid. Our depths are positive or inf, so this shouldn't happen.
    # But let's be safe.
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_valid) | (bid_depths == -np.inf), min_valid + 1e-6, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_valid) | (ask_depths == -np.inf), min_valid + 1e-6, ask_depths)
    
    # Ensure shape is (4000,)
    # inventory is (4000,), so bid_depths and ask_depths are (4000,)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
