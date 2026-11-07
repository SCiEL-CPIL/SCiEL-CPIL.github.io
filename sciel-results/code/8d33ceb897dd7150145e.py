"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq
import time

_cache = {}
_start_time = None

def _solve_depth(delta_v, A, k, dt, q_idx, Q):
    """
    Solve for optimal depth d maximizing p(d)*(d + delta_v).
    p(d) = 1 - exp(-A * exp(-k*d) * dt)
    """
    if q_idx >= Q or q_idx <= -Q:
        return np.inf
    
    # If delta_v is very negative, it might be optimal not to quote (d -> inf or 0?)
    # Actually, if delta_v is negative, we still quote to earn depth d, but the 
    # probability penalty might outweigh it.
    # The function f(d) = (1 - exp(-lam(d))) * (d + delta_v)
    # f(0) = 0 (since p(0) is small? No, p(0) = 1 - exp(-A*dt). If A*dt is large, p(0)~1.
    # Wait, if d=0, we quote at mid. p(0) = 1 - exp(-A*dt).
    # If we quote at d=0, we earn 0 depth. Reward = 0.
    # So f(0) = 0.
    # As d -> inf, p(d) -> 1, reward -> d + delta_v.
    # If delta_v > 0, large d is good? No, p(d) drops.
    
    # Let's define the objective in terms of lambda = A * exp(-k*d) * dt
    # d = -1/k * ln(lambda * dt / A)
    # Range of lambda:
    # d=0 => lambda = A*dt
    # d=inf => lambda = 0
    # So we search lambda in (0, A*dt].
    
    # Derivative w.r.t lambda:
    # f(lam) = (1 - exp(-lam)) * ( -1/k * ln(lam*dt/A) + delta_v )
    # f'(lam) = exp(-lam) * ( -1/k * ln(lam*dt/A) + delta_v ) + (1 - exp(-lam)) * ( -1/(k*lam) )
    
    def d_obj(lam):
        if lam <= 0:
            return -np.inf # We want to find root, avoid log(0)
        term1 = np.exp(-lam) * (-1.0/k * np.log(lam * dt / A) + delta_v)
        term2 = (1.0 - np.exp(-lam)) * (-1.0 / (k * lam))
        return term1 + term2

    # We need to find lambda in (0, A*dt] such that d_obj(lam) = 0.
    # Let's check signs at boundaries.
    lam_max = A * dt
    # As lam -> 0, term1 -> -inf (log term dominates), term2 -> -inf. So d_obj -> -inf.
    # At lam = lam_max:
    # d_obj(lam_max) might be positive or negative.
    # If d_obj(lam_max) > 0, the max is at lam_max (d=0).
    # If d_obj(lam_max) < 0, there is a root in (0, lam_max).
    
    # However, if delta_v is very large positive, we might want large d (small lambda).
    # But d is limited by the fact that p(d) must be significant.
    # Actually, if delta_v is large positive, we want to maximize d, so minimize lambda.
    # But f(lam) as lam->0: (1) * ( -1/k * ln(lam) + delta_v ). This goes to infinity!
    # Wait. If delta_v > 0, and we can make d arbitrarily large, the reward is unbounded?
    # No, p(d) goes to 1. Reward is d + delta_v.
    # But the probability of fill is 1.
    # So if delta_v > 0, we should quote at infinite depth?
    # The problem statement says: "bid_depths ... distances below and above mid ... np.inf means no quote".
    # If we quote at inf, we never get filled?
    # "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt)"
    # If bid_depth -> inf, exp(-k*inf) -> 0, prob -> 0.
    # So if we quote at inf, we get 0 reward.
    # So f(inf) = 0.
    
    # Let's re-evaluate f(d) = (1 - exp(-A e^{-kd} dt)) * (d + delta_v)
    # As d -> 0, f(0) = (1 - e^{-A dt}) * delta_v.
    # As d -> inf, f(inf) = 1 * (inf + delta_v) ??? No, the probability term goes to 0.
    # Let lam = A e^{-kd} dt. d = -1/k ln(lam dt / A).
    # As d -> inf, lam -> 0.
    # f(lam) = (1 - e^{-lam}) * ( -1/k ln(lam dt / A) + delta_v )
    # As lam -> 0, (1-e^{-lam}) ~ lam.
    # f(lam) ~ lam * ( -1/k ln(lam) + C )
    # lim lam->0 lam * ln(lam) = 0.
    # So f(inf) = 0.
    
    # So f(0) = (1 - e^{-A dt}) delta_v.
    # f(inf) = 0.
    # If delta_v > 0, f(0) > 0. There must be a local max.
    # If delta_v < 0, f(0) < 0. f(inf) = 0. The max is 0 (at inf or 0?).
    # If we quote at d=0, we get negative reward. We should not quote?
    # "np.inf means no quote".
    # So if max reward <= 0, we should return np.inf.
    
    # Check if max reward > 0.
    # If delta_v <= 0, is it ever beneficial to quote?
    # f(d) = p(d)(d + delta_v).
    # If delta_v is very negative, d + delta_v is negative for small d.
    # But for large d, d + delta_v becomes positive.
    # However, p(d) becomes very small.
    # Let's just find the critical point.
    
    # We search for root of d_obj(lam) in (eps, lam_max).
    # If no root, check endpoints.
    
    eps = 1e-12
    if lam_max <= eps:
        return np.inf # A*dt too small, prob ~ 0, reward ~ 0
        
    val_max = d_obj(lam_max)
    val_min = d_obj(eps)
    
    # If both are negative or both positive, no root in between?
    # d_obj(eps) is usually -inf (due to -1/(k*lam)).
    # So d_obj(eps) < 0.
    # If val_max < 0, then d_obj is negative everywhere?
    # If d_obj is always negative, f is decreasing?
    # If f is decreasing from d=0, then max is at d=0.
    # But f(0) = (1-e^{-A dt}) delta_v.
    # If delta_v < 0, f(0) < 0. f(inf) = 0.
    # So max is 0 at d=inf (no quote).
    
    # If val_max > 0, there is a root.
    # If val_max <= 0, the max is at boundary d=0 or d=inf.
    # Compare f(0) and f(inf)=0.
    # f(0) = (1 - exp(-A*dt)) * delta_v
    # If f(0) > 0, optimal is d=0.
    # If f(0) <= 0, optimal is d=inf (no quote).
    
    f0 = (1.0 - np.exp(-A * dt)) * delta_v
    
    if val_max <= 0:
        if f0 > 0:
            return 0.0
        else:
            return np.inf
            
    # There is a root in (eps, lam_max)
    try:
        lam_opt = brentq(d_obj, eps, lam_max, xtol=1e-12)
        d_opt = -1.0 / k * np.log(lam_opt * dt / A)
        # Ensure d_opt is valid
        if not np.isfinite(d_opt) or d_opt < 0:
            return 0.0
        return d_opt
    except Exception:
        # Fallback to 0 or inf based on f0
        if f0 > 0:
            return 0.0
        else:
            return np.inf

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Value table
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for q_idx in range(-Q, Q + 1):
            V_next_q = V[n + 1, q_idx + Q]
            
            # Bid side
            if q_idx < Q:
                delta_b = V[n + 1, q_idx + 1 + Q] - V_next_q
                bid_depth = _solve_depth(delta_b, A, k, dt, q_idx, Q)
            else:
                bid_depth = np.inf
            
            # Ask side
            if q_idx > -Q:
                delta_a = V[n + 1, q_idx - 1 + Q] - V_next_q
                ask_depth = _solve_depth(delta_a, A, k, dt, q_idx, Q)
            else:
                ask_depth = np.inf
            
            # Calculate rewards for V update
            # p(d) = 1 - exp(-A exp(-k d) dt)
            if bid_depth == 0.0:
                p_bid = 1.0 - np.exp(-A * dt)
                reward_bid = p_bid * (0.0 + delta_b)
            elif np.isinf(bid_depth):
                reward_bid = 0.0
            else:
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_depth) * dt)
                reward_bid = p_bid * (bid_depth + delta_b)
            
            if ask_depth == 0.0:
                p_ask = 1.0 - np.exp(-A * dt)
                reward_ask = p_ask * (0.0 + delta_a)
            elif np.isinf(ask_depth):
                reward_ask = 0.0
            else:
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_depth) * dt)
                reward_ask = p_ask * (ask_depth + delta_a)
            
            V[n, q_idx + Q] = V_next_q - risk[q_idx + Q] + reward_bid + reward_ask
            opt_bid[n, q_idx + Q] = bid_depth
            opt_ask[n, q_idx + Q] = ask_depth
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit, leave 10s margin
    if time.time() - _start_time > 290:
        # Return a safe default: quote at 1/k
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(t / dt)
    n = min(n, steps - 1)  # clamp to valid range
    
    # Map inventory to indices
    inv_idx = inventory + Q
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    # The problem states: "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
    # Our depths are >= 0 or inf.
    # We must ensure no NaN or -inf.
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    # Also check for values below -3/k (shouldn't happen, but just in case)
    min_allowed = -3.0 / params['k']
    bid_depths = np.where(bid_depths < min_allowed, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < min_allowed, np.inf, ask_depths)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
