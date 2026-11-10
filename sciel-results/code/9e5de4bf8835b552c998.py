"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 310.0  # Safety margin before 330s limit

def quote(t, inventory, mid, params):
    """
    Optimal market making quote with inventory skewing.
    
    Uses HWHM-inspired depth skewing based on inventory risk.
    Combines linear skew with a constant offset derived from the
    (2q+1) term in the derivative of q^2 cost function.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n_steps_remaining = max(1, steps - int(round(t / dt)))
    
    q = np.asarray(inventory, dtype=np.float64)
    
    # Base depth: optimal for no-inventory case
    d_0 = 1.0 / k
    
    # Time to horizon in continuous units
    tau = dt * n_steps_remaining
    
    # Linear skew coefficient: from first-order optimality
    # d = 1/k - DeltaV, DeltaV approx -gamma*sigma^2*tau*q
    alpha = gamma * sigma * sigma * tau
    
    # Quadratic term: captures convexity of value function
    # Second derivative of V w.r.t. q contributes a q^2 term
    # beta scales the quadratic correction
    beta = 0.5 * gamma * sigma * sigma * dt
    
    # Constant offset from the (2q+1) term in derivative of q^2
    offset = 0.5 * alpha
    
    # Bid depth: increase when long (q>0), decrease when short (q<0)
    # Add quadratic term that increases depth for large |q|
    d_bid = d_0 + alpha * q + beta * q * q + offset
    
    # Ask depth: decrease when long (q>0), increase when short (q<0)
    d_ask = d_0 - alpha * q + beta * q * q + offset
    
    # Clip to valid range
    min_depth = -3.0 / k + 1e-6
    max_depth = 50.0 / k
    
    d_bid = np.clip(d_bid, min_depth, max_depth)
    d_ask = np.clip(d_ask, min_depth, max_depth)
    
    # Boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)
    
    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    
    return (d_bid, d_ask)

# EVOLVE-BLOCK-END
