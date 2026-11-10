"""Optimal market making with inventory risk - Coordinate Ascent approach.

Uses the first-order conditions to derive a 1D optimization problem for bid
and ask depths, given the other depth. This allows for precise solutions
without a coarse 2D grid.
"""
import numpy as np
import time
from scipy.optimize import brentq

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _fill_prob(depth, A, k, dt):
    """Fill probability for a given depth, vectorized over depth array."""
    if depth.size == 0:
        return np.array([])
    d = np.maximum(depth, 0.0)
    exp_term = np.exp(-k * d)
    p = 1.0 - np.exp(-A * exp_term * dt)
    return np.clip(p, 0.0, 1.0)

def _fill_prob_scalar(depth, A, k, dt):
    """Fill probability for a scalar depth."""
    if depth <= 0:
        return 0.0
    exp_term = np.exp(-k * depth)
    p = 1.0 - np.exp(-A * exp_term * dt)
    return min(max(p, 0.0), 1.0)

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths using DP with coordinate ascent on depths.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1

    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))

    start_time = time.time()
    deadline = start_time + max(1.0, _TIME_BUDGET - (time.time() - _GLOBAL_START))

    # Max depth for search
    max_depth = 10.0 / k
    min_depth = 1e-5

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break

        V_next = V[n + 1]

        # Transition inventory offsets
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()

        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]

        # Initialize b, a to some reasonable values (e.g., from previous step or 0.5/k)
        if n == steps - 1:
            b_init = np.full(n_q, 0.5 / k)
            a_init = np.full(n_q, 0.5 / k)
        else:
            b_init = opt_bid[n+1, :]
            a_init = opt_ask[n+1, :]

        b_opt = np.zeros(n_q)
        a_opt = np.zeros(n_q)

        for i in range(n_q):
            q = i
            C = C_inv[q]
            
            # Handle boundaries
            if q == n_q - 1: # q = Q, no bid fill
                # Only optimize a
                # J(a) = a p_a(a) - C + V_q(q) + p_a [ (V(q+1)-V(q)) ] 
                # Wait, at q=Q, bid fill is impossible.
                # Transition: 
                # Ask fill (prob p_a): q -> q-1. Reward a.
                # No fill (prob 1-p_a): q -> q.
                # V = max_a [ a p_a(a) - C + p_a V(q-1) + (1-p_a) V(q) ]
                # V = max_a [ a p_a(a) - C + V(q) + p_a (V(q-1) - V(q)) ]
                # Let Delta = V(q-1) - V(q)
                # J(a) = a p_a(a) + p_a Delta + Const
                # J'(a) = p_a'(a)(a + Delta) + p_a(a) = 0
                Delta = V_next[q-1] - V_next[q]
                
                def obj_a_prime(a):
                    pa = _fill_prob_scalar(a, A, k, dt)
                    if pa == 0 or pa == 1:
                        return 0.0 # Handle boundaries
                    # p_a'(a) = k * A * exp(-k a) * dt * exp(-A exp(-k a) dt)
                    u = A * np.exp(-k * a) * dt
                    p_prime = k * u * np.exp(-u)
                    return p_prime * (a + Delta) + pa
                
                try:
                    # Search for root in [min_depth, max_depth]
                    # If no sign change, pick boundary
                    lo, hi = min_depth, max_depth
                    f_lo = obj_a_prime(lo)
                    f_hi = obj_a_prime(hi)
                    if f_lo * f_hi < 0:
                        a_star = brentq(obj_a_prime, lo, hi, xtol=1e-6)
                    elif f_lo > 0:
                        a_star = lo
                    else:
                        a_star = hi
                except:
                    a_star = a_init[q]
                
                a_opt[q] = a_star
                b_opt[q] = np.inf # No bid

            elif q == 0: # q = -Q, no ask fill
                # Only optimize b
                # J(b) = b p_b(b) + p_b (V(q+1) - V(q)) + Const
                Delta = V_next[q+1] - V_next[q]
                
                def obj_b_prime(b):
                    pb = _fill_prob_scalar(b, A, k, dt)
                    if pb == 0 or pb == 1:
                        return 0.0
                    u = A * np.exp(-k * b) * dt
                    p_prime = k * u * np.exp(-u)
                    return p_prime * (b + Delta) + pb

                try:
                    lo, hi = min_depth, max_depth
                    f_lo = obj_b_prime(lo)
                    f_hi = obj_b_prime(hi)
                    if f_lo * f_hi < 0:
                        b_star = brentq(obj_b_prime, lo, hi, xtol=1e-6)
                    elif f_lo > 0:
                        b_star = lo
                    else:
                        b_star = hi
                except:
                    b_star = b_init[q]
                
                b_opt[q] = b_star
                a_opt[q] = np.inf # No ask

            else:
                # Interior q
                # Coordinate ascent
                b_curr = b_init[q]
                a_curr = a_init[q]
                
                for _ in range(10): # 10 iterations of coordinate ascent
                    # Optimize b given a
                    pa = _fill_prob_scalar(a_curr, A, k, dt)
                    Delta_2 = V_next[q+1] - V_next[q-1]
                    # J(b) = p_b(b) (b + pa * Delta_2) + Const
                    # J'(b) = p_b'(b) (b + K) + p_b(b) = 0
                    K = pa * Delta_2
                    
                    def obj_b_prime(b):
                        pb = _fill_prob_scalar(b, A, k, dt)
                        if pb == 0 or pb == 1:
                            return 0.0
                        u = A * np.exp(-k * b) * dt
                        p_prime = k * u * np.exp(-u)
                        return p_prime * (b + K) + pb

                    try:
                        lo, hi = min_depth, max_depth
                        f_lo = obj_b_prime(lo)
                        f_hi = obj_b_prime(hi)
                        if f_lo * f_hi < 0:
                            b_new = brentq(obj_b_prime, lo, hi, xtol=1e-6)
                        elif f_lo > 0:
                            b_new = lo
                        else:
                            b_new = hi
                    except:
                        b_new = b_curr

                    # Optimize a given b
                    pb = _fill_prob_scalar(b_new, A, k, dt)
                    Delta_2_a = V_next[q-1] - V_next[q+1] # Note sign change?
                    # J(a) = p_a(a) (a + pb * (V(q-1) - V(q+1))) + Const?
                    # Let's re-derive for a.
                    # J(a) = a p_a(a) + b p_b(b) - C + V(q) + p_b [ (V(q+1)-V(q)) + p_a (V(q-1) - V(q+1)) ]
                    # Wait, previous derivation:
                    # Trans = V(q) + p_a [ (V(q-1)-V(q)) + p_b (V(q+1) - V(q-1)) ]
                    # This was for fixing a and optimizing b.
                    # Symmetrically, fixing b and optimizing a:
                    # Trans = V(q) + p_b [ (V(q+1)-V(q)) + p_a (V(q-1) - V(q+1)) ]
                    # So J(a) = a p_a(a) + p_b (V(q+1)-V(q)) + p_b p_a (V(q-1) - V(q+1))
                    # J(a) = p_a(a) [ a + p_b (V(q-1) - V(q+1)) ] + Const
                    # Let K_a = p_b (V(q-1) - V(q+1))
                    K_a = pb * (V_next[q-1] - V_next[q+1])
                    
                    def obj_a_prime(a):
                        pa = _fill_prob_scalar(a, A, k, dt)
                        if pa == 0 or pa == 1:
                            return 0.0
                        u = A * np.exp(-k * a) * dt
                        p_prime = k * u * np.exp(-u)
                        return p_prime * (a + K_a) + pa

                    try:
                        lo, hi = min_depth, max_depth
                        f_lo = obj_a_prime(lo)
                        f_hi = obj_a_prime(hi)
                        if f_lo * f_hi < 0:
                            a_new = brentq(obj_a_prime, lo, hi, xtol=1e-6)
                        elif f_lo > 0:
                            a_new = lo
                        else:
                            a_new = hi
                    except:
                        a_new = a_curr

                    if abs(b_new - b_curr) < 1e-5 and abs(a_new - a_curr) < 1e-5:
                        b_curr = b_new
                        a_curr = a_new
                        break
                    b_curr = b_new
                    a_curr = a_new

                b_opt[q] = b_curr
                a_opt[q] = a_curr

        # Compute V with refined optima
        pb = _fill_prob(b_opt, A, k, dt)
        pa = _fill_prob(a_opt, A, k, dt)
        
        # Handle inf
        pb = np.where(np.isinf(b_opt), 0.0, pb)
        pa = np.where(np.isinf(a_opt), 0.0, pa)

        V_refined = np.empty(n_q)
        
        # Interior
        for i in range(1, n_q-1):
            trans = pb[i] * (1 - pa[i]) * V_next[i+1] + \
                    pa[i] * (1 - pb[i]) * V_next[i-1] + \
                    (1 - pb[i]) * (1 - pa[i]) * V_next[i]
            V_refined[i] = b_opt[i] * pb[i] + a_opt[i] * pa[i] - C_inv[i] + trans

        # q = -Q (i=0)
        V_refined[0] = b_opt[0] * pb[0] - C_inv[0] + pb[0] * V_next[1] + (1 - pb[0]) * V_next[0]
        
        # q = Q (i=n_q-1)
        V_refined[-1] = a_opt[-1] * pa[-1] - C_inv[-1] + pa[-1] * V_next[-2] + (1 - pa[-1]) * V_next[-1]

        V[n, :] = V_refined
        opt_bid[n, :] = b_opt
        opt_ask[n, :] = a_opt

    return opt_bid, opt_ask


def _get_policy(params):
    """Retrieves or computes the optimal policy for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Returns optimal bid and ask depths based on pre-computed DP policy."""
    opt_bid, opt_ask = _get_policy(params)

    dt = params['T'] / params['steps']
    n = int(round(t / dt))

    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1

    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].copy()
    ask_depths = opt_ask[n, q_idx].copy()

    # Ensure valid outputs: clip to valid range
    min_depth = 0.001
    max_depth = 10.0 / params['k']

    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)

    # Replace any NaN or inf
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, min_depth)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, min_depth)

    # No bid at q=Q, no ask at q=-Q
    # When inventory == Q, set bid to inf (no bid quote)
    # When inventory == -Q, set ask to inf (no ask quote)
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
