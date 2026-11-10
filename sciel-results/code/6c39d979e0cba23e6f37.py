"""Optimal market making with inventory risk - NEWTON + GRID approach.

Uses a coarse grid for initial estimates, then refines each (bid, ask) pair
with Newton-Raphson on the continuous first-order conditions. This eliminates
most grid-search error while keeping computation tractable.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # leave 10s margin before 330s kill


def _fill_prob(depth, A, k, dt):
    """Fill probability for a given depth."""
    if depth <= 0:
        return 0.0
    exp_term = np.exp(-k * depth)
    p = 1.0 - np.exp(-A * exp_term * dt)
    return min(max(p, 0.0), 1.0)


def _fill_prob_vec(depths, A, k, dt):
    """Vectorized fill probability."""
    depths = np.asarray(depths, dtype=float)
    p = np.where(depths > 0, 1.0 - np.exp(-A * np.exp(-k * depths) * dt), 0.0)
    return np.clip(p, 0.0, 1.0)


def _newton_refine_bid(b_init, a_fixed, q_idx, V_q_plus, V_q_minus, V_q,
                       C_inv, A, k, dt, n_q, max_iter=30, tol=1e-8):
    """
    Refine bid depth b using Newton-Raphson on the first-order condition.
    The objective for bid b (given ask a) is:
        f(b) = b * p_b + a * p_a - C_inv + p_b*(1-p_a)*V_q_plus
              + p_a*(1-p_b)*V_q_minus + (1-p_b-p_a+p_b*p_a)*V_q
    where p_b = 1 - exp(-A*exp(-k*b)*dt), p_a = 1 - exp(-A*exp(-k*a)*dt)
    
    For interior q (not at boundaries), the FOC is:
        p_b + b * dp_b/db = (1-p_a)*V_q_plus - (1-p_b)*V_q_minus
              + (p_a*(1-p_b) - (1-p_b-p_a+p_b*p_a))*0  [simplified]
    
    Actually, let's derive properly:
    f(b) = b*p_b + a*p_a - C + p_b(1-p_a)V+ + p_a(1-p_b)V- + (1-p_b-p_a+p_b p_a)V
    
    df/db = p_b + b*p_b' + p_b'(1-p_a)V+ - p_a*p_b'V- + (-p_b' - 0 + p_b'p_a)V
          = p_b + p_b' [ b + (1-p_a)V+ - p_a V- - V + p_a V ]
          = p_b + p_b' [ b + (1-p_a)V+ - p_a V- - V(1 - p_a) ]
          = p_b + p_b' [ b - V + (1-p_a)(V+ - V) - p_a(V- - V) ]
    
    Wait, let me re-derive more carefully:
    f(b) = b*p_b + a*p_a - C + p_b(1-p_a)V+ + p_a(1-p_b)V- + (1-p_b-p_a+p_b p_a)V
    
    Let me expand the last term: 1 - p_b - p_a + p_b*p_a
    So f(b) = b*p_b + a*p_a - C + p_b(1-p_a)V+ + p_a(1-p_b)V- + V - p_b V - p_a V + p_b p_a V
    
    = b*p_b + a*p_a - C + V + p_b[(1-p_a)V+ - V + p_a V] + p_a[V- - V] + ... 
    wait this is getting complicated. Let me just compute numerically.
    
    For boundary cases (q=0 or q=Q-1), the formula is simpler.
    """
    if q_idx == 0:
        # No ask fill possible. f(b) = b*p_b - C + p_b V+ + (1-p_b) V
        # df/db = p_b + b*p_b' + p_b' V+ - p_b' V = p_b + p_b'(b + V+ - V)
        # p_b' = A*exp(-k*b)*dt*exp(-k*b) ... wait
        # p_b = 1 - exp(-A*exp(-k*b)*dt)
        # dp_b/db = -exp(-A*exp(-k*b)*dt) * A*exp(-k*b)*dt*(-k)
        #         = k*A*exp(-k*b)*dt*exp(-A*exp(-k*b)*dt)
        #         = k*A*exp(-k*b)*dt*p_b... no
        #         = k*A*exp(-k*b)*dt*(1 - p_b)
        
        b = max(b_init, 0.001)
        for _ in range(max_iter):
            pb = _fill_prob(b, A, k, dt)
            dpb = k * A * np.exp(-k * b) * dt * (1.0 - pb) if pb < 1.0 else 0.0
            
            # f'(b) = pb + dpb*(b + V_q_plus[q_idx] - V_q[q_idx])
            fprime = pb + dpb * (b + V_q_plus[q_idx] - V_q[q_idx])
            
            # f''(b) for Newton step
            # dpb/db = k*A*exp(-k*b)*dt*(1-pb) 
            # d2pb/db2 = -k^2*A*exp(-k*b)*dt*(1-pb) + k*A*exp(-k*b)*dt*(-dpb)
            #          = k*A*exp(-k*b)*dt * (-k*(1-pb) - dpb)
            d2pb = k * A * np.exp(-k * b) * dt * (-k * (1.0 - pb) - dpb)
            f2prime = dpb + d2pb * (b + V_q_plus[q_idx] - V_q[q_idx]) + dpb
            
            if abs(f2prime) < 1e-15:
                break
            step = fprime / f2prime
            b_new = b - step
            if b_new <= 0:
                b_new = 0.001
            if abs(b_new - b) < tol:
                b = b_new
                break
            b = b_new
        
        return b
    
    elif q_idx == n_q - 1:
        # No bid fill possible. Bid doesn't matter.
        return b_init
    
    else:
        # Interior: f(b) = b*pb + a*pa - C + pb(1-pa)V+ + pa(1-pb)V- + (1-pb-pa+pb*pa)V
        # Let's compute f'(b) numerically for stability, or analytically:
        # f'(b) = pb + b*pb' + pb'(1-pa)V+ - pa*pb'V- + (-pb' - 0 + pb'*pa)V
        #        = pb + pb'[b + (1-pa)V+ - pa*V- - V + pa*V]
        #        = pb + pb'[b + (1-pa)(V+ - V) - pa(V- - V)]
        
        pa = _fill_prob(a_fixed, A, k, dt)
        b = max(b_init, 0.001)
        for _ in range(max_iter):
            pb = _fill_prob(b, A, k, dt)
            dpb = k * A * np.exp(-k * b) * dt * (1.0 - pb) if pb < 1.0 else 0.0
            
            bracket = b + (1.0 - pa) * (V_q_plus[q_idx] - V_q[q_idx]) - pa * (V_q_minus[q_idx] - V_q[q_idx])
            fprime = pb + dpb * bracket
            
            # Second derivative
            d2pb = k * A * np.exp(-k * b) * dt * (-k * (1.0 - pb) - dpb)
            f2prime = dpb + d2pb * bracket + dpb
            
            if abs(f2prime) < 1e-15:
                break
            step = fprime / f2prime
            b_new = b - step
            if b_new <= 0:
                b_new = 0.001
            if abs(b_new - b) < tol:
                b = b_new
                break
            b = b_new
        
        return b


def _newton_refine_ask(a_init, b_fixed, q_idx, V_q_plus, V_q_minus, V_q,
                       C_inv, A, k, dt, n_q, max_iter=30, tol=1e-8):
    """
    Refine ask depth a using Newton-Raphson on the first-order condition.
    """
    if q_idx == n_q - 1:
        # No ask fill... wait, at q=Q, no bid fill. Ask fill is allowed.
        # f(a) = a*pa - C + pa V- + (1-pa) V
        # f'(a) = pa + pa'(a + V- - V)
        a = max(a_init, 0.001)
        for _ in range(max_iter):
            pa = _fill_prob(a, A, k, dt)
            dpa = k * A * np.exp(-k * a) * dt * (1.0 - pa) if pa < 1.0 else 0.0
            
            fprime = pa + dpa * (a + V_q_minus[q_idx] - V_q[q_idx])
            d2pa = k * A * np.exp(-k * a) * dt * (-k * (1.0 - pa) - dpa)
            f2prime = dpa + d2pa * (a + V_q_minus[q_idx] - V_q[q_idx]) + dpa
            
            if abs(f2prime) < 1e-15:
                break
            step = fprime / f2prime
            a_new = a - step
            if a_new <= 0:
                a_new = 0.001
            if abs(a_new - a) < tol:
                a = a_new
                break
            a = a_new
        
        return a
    
    elif q_idx == 0:
        # At q=-Q, no ask fill. Ask doesn't matter.
        return a_init
    
    else:
        # Interior: f(a) = a*pa + b*pb - C + pb(1-pa)V+ + pa(1-pb)V- + (1-pb-pa+pb*pa)V
        # f'(a) = pa + a*pa' - pb*pa'V+ + pa'(1-pb)V- + (-pa' + pb*pa')V
        #        = pa + pa'[a - pb*V+ + (1-pb)V- - V + pb*V]
        #        = pa + pa'[a - pb(V+ - V) + (1-pb)(V- - V)]
        
        pb = _fill_prob(b_fixed, A, k, dt)
        a = max(a_init, 0.001)
        for _ in range(max_iter):
            pa = _fill_prob(a, A, k, dt)
            dpa = k * A * np.exp(-k * a) * dt * (1.0 - pa) if pa < 1.0 else 0.0
            
            bracket = a - pb * (V_q_plus[q_idx] - V_q[q_idx]) + (1.0 - pb) * (V_q_minus[q_idx] - V_q[q_idx])
            fprime = pa + dpa * bracket
            
            d2pa = k * A * np.exp(-k * a) * dt * (-k * (1.0 - pa) - dpa)
            f2prime = dpa + d2pa * bracket + dpa
            
            if abs(f2prime) < 1e-15:
                break
            step = fprime / f2prime
            a_new = a - step
            if a_new <= 0:
                a_new = 0.001
            if abs(a_new - a) < tol:
                a = a_new
                break
            a = a_new
        
        return a


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths using:
    1. Coarse grid for initial optima
    2. Newton-Raphson refinement on continuous FOCs
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1

    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    # Coarse grid for initial search
    coarse_grid = np.linspace(0.01, 4.0, 100)
    n_coarse = len(coarse_grid)

    # V[n, q_idx]
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))

    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))

    # Pre-compute fill probabilities on coarse grid
    p_coarse = _fill_prob_vec(coarse_grid, A, k, dt)

    # Matrix for coarse grid
    P_b = np.tile(p_coarse[:, np.newaxis], (1, n_coarse))
    P_a = np.tile(p_coarse[np.newaxis, :], (n_coarse, 1))

    Depths_b = np.tile(coarse_grid[:, np.newaxis], (1, n_coarse))
    Depths_a = np.tile(coarse_grid[np.newaxis, :], (n_coarse, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a

    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus

    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n + 1, :] = V[n + 1, :]
                opt_bid[n:n + 1, :] = opt_bid[n + 1, :]
                opt_ask[n:n + 1, :] = opt_ask[n + 1, :]
            break

        V_next = V[n + 1]

        # Transition inventory offsets
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()

        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]

        # Compute values on coarse grid
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        Values += P_plus[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]

        # Boundary handling
        Values[-1, :, :] -= P_plus * V_q_plus[-1, np.newaxis, np.newaxis]
        Values[0, :, :] -= P_minus * V_q_minus[0, np.newaxis, np.newaxis]

        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)

        i_idx, j_idx = np.unravel_index(max_indices, (n_coarse, n_coarse))

        b_coarse = coarse_grid[i_idx]
        a_coarse = coarse_grid[j_idx]

        # Newton-Raphson refinement for each q
        refined_bid = np.zeros(n_q)
        refined_ask = np.zeros(n_q)
        V_refined = np.zeros(n_q)

        for q_idx in range(n_q):
            b_cur = b_coarse[q_idx]
            a_cur = a_coarse[q_idx]

            # Iterate Newton refinement a few times for bid-ask coupling
            for _ in range(3):
                if q_idx != n_q - 1:
                    b_cur = _newton_refine_bid(b_cur, a_cur, q_idx, V_q_plus, V_q_minus,
                                               V_q, C_inv, A, k, dt, n_q)
                if q_idx != 0:
                    a_cur = _newton_refine_ask(a_cur, b_cur, q_idx, V_q_plus, V_q_minus,
                                               V_q, C_inv, A, k, dt, n_q)

            refined_bid[q_idx] = b_cur
            refined_ask[q_idx] = a_cur

            # Compute refined value
            pb = _fill_prob(b_cur, A, k, dt)
            pa = _fill_prob(a_cur, A, k, dt)

            if q_idx == n_q - 1:
                trans = pa * V_q_minus[q_idx] + (1 - pa) * V_q[q_idx]
                V_refined[q_idx] = a_cur * pa - C_inv[q_idx] + trans
            elif q_idx == 0:
                trans = pb * V_q_plus[q_idx] + (1 - pb) * V_q[q_idx]
                V_refined[q_idx] = b_cur * pb - C_inv[q_idx] + trans
            else:
                trans = pb * (1 - pa) * V_q_plus[q_idx] + \
                        pa * (1 - pb) * V_q_minus[q_idx] + \
                        (1 - pb - pa + pb * pa) * V_q[q_idx]
                V_refined[q_idx] = b_cur * pb + a_cur * pa - C_inv[q_idx] + trans

        V[n, :] = V_refined
        opt_bid[n, :] = refined_bid
        opt_ask[n, :] = refined_ask

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

    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]

    # Ensure valid outputs
    bid_depths = np.where(np.isnan(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.01, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.01, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
