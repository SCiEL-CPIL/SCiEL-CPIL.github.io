"""Optimal market making with inventory risk - Vectorized continuous-depth solver.

Combines the vectorized grid approach (fast, robust) with Newton-Raphson
refinement of the first-order condition to eliminate grid discretization error.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 325.0  # seconds, leave 5s margin before 330s kill


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes optimal bid/ask depths via dynamic programming.
    Uses a vectorized grid search followed by Newton-Raphson refinement
    of the first-order condition for continuous optimality.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1

    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    # Depth grid for initial search
    # Use a wide enough range to capture the optimum
    depth_grid = np.linspace(0.005, 8.0, 400)
    n_depths = len(depth_grid)

    # Pre-compute fill probabilities on the grid
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depth_grid)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    p_depths = np.clip(p_depths, 0.0, 1.0)

    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))

    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))

    # Precompute for Newton refinement
    # We'll refine each scalar depth using Newton on the FOC

    for n in range(steps - 1, -1, -1):
        # Time check
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

        # --- Vectorized grid search for initial depths ---
        # For each q, we want to maximize over (b, a):
        #   b * p(b) + a * p(a) - C_inv[q]
        #   + (1-p(b))(1-p(a)) V[q]
        #   + p(b)(1-p(a)) V[q+1]
        #   + (1-p(b)) p(a) V[q-1]
        #
        # This is a 2D optimization. We use coordinate descent:
        # First optimize b given a, then a given b, iterate.

        # Initialize with symmetric depths
        b_cur = np.full(n_q, 1.0 / k)
        a_cur = np.full(n_q, 1.0 / k)

        # Coordinate descent with vectorized grid search
        max_cd_iters = 5
        for cd_iter in range(max_cd_iters):
            # Optimize bid (b) given ask (a)
            exp_a = np.exp(-k * a_cur)
            p_a = 1.0 - np.exp(-A * exp_a * dt)
            one_minus_p_a = 1.0 - p_a

            # For each q, maximize over b:
            #   b * p(b) + a * p(a) - C_inv[q]
            #   + (1-p(b)) * one_minus_p_a * V[q]
            #   + p(b) * one_minus_p_a * V[q+1]
            #   + (1-p(b)) * p_a * V[q-1]
            # = a*p(a) - C_inv[q] + one_minus_p_a*p_a*V[q-1] + one_minus_p_a*(1-p_a)*V[q]
            #   + p(b) * [b + one_minus_p_a * (V[q+1] - V[q])]
            #   + (1-p(b)) * [one_minus_p_a * V[q] - one_minus_p_a*p_a*V[q-1]]
            # = const + p(b) * [b + K(q)] where K(q) = one_minus_p_a*(V[q+1]-V[q])
            #   + (1-p(b)) * M(q) where M(q) = one_minus_p_a*(V[q] - p_a*V[q-1])
            # = M(q) + p(b)*[b + K(q) - M(q)]
            # So maximize p(b) * (b + K(q) - M(q)) over b

            K = one_minus_p_a * (V_q_plus - V_q)
            M = one_minus_p_a * (V_q - p_a * V_q_minus)
            # Objective for b: p(b) * (b + K - M)
            # Vectorized: for each q, compute p_depths * (depth_grid + K[q] - M[q])
            # Shape: (n_q, n_depths)
            obj_b = p_depths[np.newaxis, :] * (depth_grid[np.newaxis, :] + (K - M)[:, np.newaxis])

            # Boundary: at q=Q, no bid fill, so p(b)=0 effectively
            obj_b[-1, :] = 0.0  # will pick depth 0

            i_idx = np.argmax(obj_b, axis=1)
            b_new = depth_grid[i_idx]

            # Optimize ask (a) given bid (b_new)
            exp_b = np.exp(-k * b_new)
            p_b = 1.0 - np.exp(-A * exp_b * dt)
            one_minus_p_b = 1.0 - p_b

            # For each q, maximize over a:
            #   b*p(b) + a*p(a) - C_inv[q]
            #   + (1-p_b)*one_minus_p_a*V[q]  -- wait, we need to redo
            # Actually let's re-derive:
            # V = b*p_b + a*p_a - C_inv
            #   + (1-p_b)(1-p_a) V[q]
            #   + p_b(1-p_a) V[q+1]
            #   + (1-p_b) p_a V[q-1]
            # Given b fixed, maximize over a:
            # = b*p_b - C_inv + (1-p_b)*V[q]
            #   + p_a * [a + (1-p_b)*(V[q-1] - V[q])]
            #   + (1-p_a) * [(1-p_b)*V[q] - (1-p_b)*(V[q-1])]  -- this is getting complex
            # Let me just do:
            # = const + p_a * [a + L(q)] where L(q) = (1-p_b)*(V[q-1] - V[q])
            #   + (1-p_a) * N(q) where N(q) = (1-p_b)*(V[q] - V[q-1]) ... no
            #
            # Let me be more careful:
            # Terms with a:
            #   a * p_a
            #   + (1-p_b)(1-p_a) V[q]
            #   + (1-p_b) p_a V[q-1]
            # = a*p_a + (1-p_b) V[q] - (1-p_b) p_a V[q] + (1-p_b) p_a V[q-1]
            # = (1-p_b) V[q] + p_a * [a + (1-p_b)(V[q-1] - V[q])]
            # So maximize p_a * (a + L(q)) where L(q) = (1-p_b)*(V[q-1] - V[q])

            L = one_minus_p_b * (V_q_minus - V_q)
            obj_a = p_depths[np.newaxis, :] * (depth_grid[np.newaxis, :] + L[:, np.newaxis])

            # Boundary: at q=-Q, no ask fill
            obj_a[0, :] = 0.0

            j_idx = np.argmax(obj_a, axis=1)
            a_new = depth_grid[j_idx]

            b_cur = b_new
            a_cur = a_new

        # --- Newton-Raphson refinement for continuous optimality ---
        # Refine each depth using the FOC: p'(b)*(b+K) + p(b) = 0
        # where K depends on the other side's fill probability

        # Recompute with final b_cur, a_cur
        exp_a_final = np.exp(-k * a_cur)
        p_a_final = 1.0 - np.exp(-A * exp_a_final * dt)
        one_minus_p_a_final = 1.0 - p_a_final

        exp_b_final = np.exp(-k * b_cur)
        p_b_final = 1.0 - np.exp(-A * exp_b_final * dt)
        one_minus_p_b_final = 1.0 - p_b_final

        # Refine bid depths
        K_bid = one_minus_p_a_final * (V_q_plus - V_q)
        for qi in range(n_q):
            if qi == n_q - 1:
                continue
            b = b_cur[qi]
            if b <= 0:
                continue
            # Newton iterations
            for _ in range(20):
                e = np.exp(-k * b)
                inner = A * e * dt
                one_minus_p = np.exp(-inner)
                p = 1.0 - one_minus_p
                p_prime = -A * k * dt * e * one_minus_p

                f_prime = p_prime * (b + K_bid[qi]) + p
                # f''(b) = p''(b)*(b+K) + 2*p'(b)
                # p''(b) = p_prime * (-k + inner)
                p_double_prime = p_prime * (-k + inner)
                f_double_prime = p_double_prime * (b + K_bid[qi]) + 2 * p_prime

                if abs(f_double_prime) < 1e-15:
                    break

                delta = f_prime / f_double_prime
                b_new = b - delta
                if b_new <= 0:
                    b_new = b * 0.5
                    if b_new <= 0:
                        b_new = 1e-6
                if abs(b_new - b) < 1e-10:
                    b = b_new
                    break
                b = b_new
            b_cur[qi] = max(b, 0.0)

        # Refine ask depths
        L_ask = one_minus_p_b_final * (V_q_minus - V_q)
        for qi in range(n_q):
            if qi == 0:
                continue
            a = a_cur[qi]
            if a <= 0:
                continue
            for _ in range(20):
                e = np.exp(-k * a)
                inner = A * e * dt
                one_minus_p = np.exp(-inner)
                p = 1.0 - one_minus_p
                p_prime = -A * k * dt * e * one_minus_p

                f_prime = p_prime * (a + L_ask[qi]) + p
                p_double_prime = p_prime * (-k + inner)
                f_double_prime = p_double_prime * (a + L_ask[qi]) + 2 * p_prime

                if abs(f_double_prime) < 1e-15:
                    break

                delta = f_prime / f_double_prime
                a_new = a - delta
                if a_new <= 0:
                    a_new = a * 0.5
                    if a_new <= 0:
                        a_new = 1e-6
                if abs(a_new - a) < 1e-10:
                    a = a_new
                    break
                a = a_new
            a_cur[qi] = max(a, 0.0)

        # Compute final value for this step
        exp_b_final = np.exp(-k * b_cur)
        p_b_final = 1.0 - np.exp(-A * exp_b_final * dt)

        exp_a_final = np.exp(-k * a_cur)
        p_a_final = 1.0 - np.exp(-A * exp_a_final * dt)

        # Calculate V[n, q]
        trans_plus = p_b_final * (1.0 - p_a_final) * V_q_plus
        trans_minus = p_a_final * (1.0 - p_b_final) * V_q_minus
        trans_same = (1.0 - p_b_final - p_a_final + p_b_final * p_a_final) * V_q

        V_n = b_cur * p_b_final + a_cur * p_a_final - C_inv + trans_plus + trans_minus + trans_same

        V[n, :] = V_n
        opt_bid[n, :] = b_cur
        opt_ask[n, :] = a_cur

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

    # Ensure valid outputs: must be >= 0 and finite
    bid_depths = np.where(np.isnan(bid_depths) | ~np.isfinite(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | ~np.isfinite(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.01, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.01, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
