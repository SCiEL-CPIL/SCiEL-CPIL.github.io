"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

import sys
sys.setrecursionlimit(100000)

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 320s out of 330s limit, 10s safety margin

def quote(t, inventory, mid, params):
    """
    Optimal market making quote with inventory skewing.
    
    Uses HWHM-inspired depth skewing based on inventory risk.
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
    
    # Skew coefficient: proportional to inventory risk
    # The marginal cost of inventory q is gamma*sigma^2*q*dt per step
    # Over remaining steps, total cost impact is gamma*sigma^2*q*dt*n_remaining
    # Divide by k to get depth scale
    alpha = gamma * sigma * sigma * dt * n_steps_remaining / k
    
    # Bid depth: increase when long (q>0), decrease when short (q<0)
    d_bid = d_0 + alpha * q
    
    # Ask depth: decrease when long (q>0), increase when short (q<0)
    d_ask = d_0 - alpha * q
    
    # Clip to valid range
    min_depth = -3.0 / k + 1e-6
    max_depth = 50.0 / k
    
    d_bid = np.clip(d_bid, min_depth, max_depth)
    d_ask = np.clip(d_ask, min_depth, max_depth)
    
    # Boundary conditions
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)
    
    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    
    return (d_bid, d_ask)

# EVOLVE-BLOCK-END
