"""Market-making with inventory risk: Analytical Closed-Form Policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 295.0

def quote(t, inventory, mid, params):
    """
    Quote policy using analytical closed-form approximation.
    
    The policy is derived from the HJB equation approximation for quadratic inventory risk:
    d_bid  = 1/k + alpha * q + alpha/2
    d_ask  = 1/k - alpha * q + alpha/2
    
    Where alpha captures the risk aversion and time-to-maturity.
    alpha approx = gamma * sigma^2 * (T - t) / k
    """
    # Deadline check
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    # t is the step index n, so time elapsed is t * dt.
    # Time remaining is (steps - t) * dt.
    time_remaining = (steps - t) * dt
    
    # Calculate the skew coefficient alpha
    # The "value" of inventory risk is proportional to gamma * sigma^2 * time_remaining.
    # The depth scale is 1/k.
    # We use alpha = (gamma * sigma^2 * time_remaining) / k
    # This ensures that as t -> T, alpha -> 0, and depths converge to 1/k.
    alpha = (gamma * sigma**2 * time_remaining) / k
    
    # Base depth is 1/k
    base_depth = 1.0 / k
    
    # Inventory array
    q = inventory.astype(np.float64)
    
    # Calculate depths
    # d_bid = base + alpha * q + alpha / 2
    # d_ask = base - alpha * q + alpha / 2
    # Note: The alpha/2 term comes from the discrete difference V(q-1)-V(q) approximation.
    
    bid_depths = base_depth + alpha * q + 0.5 * alpha
    ask_depths = base_depth - alpha * q + 0.5 * alpha
    
    # Enforce constraints
    # Depths must be >= 0.
    # Also, if q == Q, no bid fill is allowed. The problem says "no bid fill at q = Q".
    # This implies we should not quote a bid that would result in a fill?
    # Or simply that the probability is 0.
    # If we quote a bid depth, but q=Q, the evaluator handles the probability as 0.
    # However, quoting a bid depth > 0 when q=Q is harmless (just doesn't fill).
    # But to be safe and optimal, we might want to set depth to inf or large value?
    # The problem says "np.inf means no quote".
    # If q=Q, we cannot increase inventory. So we should NOT quote a bid.
    # Set bid_depths to np.inf where q == Q.
    # If q=-Q, we cannot decrease inventory. So we should NOT quote an ask.
    # Set ask_depths to np.inf where q == -Q.
    
    bid_depths[q == Q] = np.inf
    ask_depths[q == -Q] = np.inf
    
    # Clip negative depths to 0.
    # A depth of 0 means quoting at the mid.
    # Is 0 valid? "Values below -3/k ... invalid". 0 is valid.
    # However, quoting at mid might be too aggressive if alpha is large?
    # The formula ensures positivity for reasonable alpha and q.
    # But if alpha * q is very negative, d_ask could be negative.
    # Clip to 0.
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Ensure no NaNs or infs other than the intentional ones
    # np.maximum with 0.0 will turn -inf to 0, but we only set inf for boundaries.
    # If q is such that alpha*q is huge, depth is huge. That's fine.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
