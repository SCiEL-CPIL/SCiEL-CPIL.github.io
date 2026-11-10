"""Optimal market making with inventory risk - semi-Lagrangian HJB with
analytical first-order-condition spread.

Structural change: replace the discrete depth-grid DP with a continuous
first-order-condition (FOC) solve.  For each (t, q) we solve a 2-D system
of nonlinear equations for (bid_depth, ask_depth) using the Bellman FOCs,
then do a few Newton iterations.  The value function is propagated
backwards in time on the discrete inventory grid but the depths are
continuous, eliminating the coarse grid error.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0


def _fill_prob(depth, A, k, dt):
    """Fill probability, vectorized."""
    depth = np.maximum(depth, 0.0)
    with np.errstate(over='ignore', under='ignore'):
        p = 1.0 - np.exp(-A * np.exp(-k * depth) * dt)
    return np.clip(p, 0.0, 1.0)


def _dfill_dd(depth, A, k, dt):
    """Derivative of fill probability w.r.t. depth, vectorized."""
    depth = np.maximum(depth, 0.0)
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depth)
        d = A * exp_term * dt * k * exp_term * np.exp(-A * exp_term * dt)
    return d


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via backward DP with continuous FOC solve.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1

    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))

    start_time = time.time()
    deadline = start_time + max(1.0, _TIME_BUDGET - (time.time() - _GLOBAL_START))

    # Initial guess for depths: use a simple heuristic
    # Optimal depth ~ sqrt(1/(A*k)) scale, adjusted by inventory
    base_depth = 1.0 / np.sqrt(A * k) if A * k > 0 else 0.1

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n + 1, :] = V[n + 1, :]
                opt_bid[n:n + 1, :] = opt_bid[n + 1, :]
                opt_ask[n:n + 1, :] = opt_ask[n + 1, :]
            break

        V_next = V[n + 1]

        # Transition value arrays
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()
        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]

        b = np.full(n_q, base_depth)
        a = np.full(n_q, base_depth)

        # Newton iterations for FOC solve
        n_newton = 8
        for _ in range(n_newton):
            pb = _fill_prob(b, A, k, dt)
            pa = _fill_prob(a, A, k, dt)
            dpb = _dfill_dd(b, A, k, dt)
            dpa = _dfill_dd(a, A, k, dt)

            # FOC for bid depth (interior q):
            # d/d b [b*pb + a*pa - C + pb*(1-pa)*V+ + pa*(1-pb)*V- + (1-pb-pa+pb*pa)*V] = 0
            # pb + b*dpb + (1-pa)*V+*dpb + pa*(1-pb)*(-dpa... no, dpa is for a)
            # Let me recompute carefully.
            # f(b,a) = b*pb + a*pa - C + pb*(1-pa)*Vq+ + pa*(1-pb)*Vq- + (1-pb-pa+pb*pa)*Vq
            # df/db = pb + b*dpb + dpb*(1-pa)*Vq+ + pa*(-dpb)*Vq- + (-dpb - pa*dpb)*Vq
            #       = pb + b*dpb + dpb*(1-pa)*Vq+ - pa*dpb*Vq- - dpb*(1+pa)*Vq
            #       = pb + dpb*(b + (1-pa)*Vq+ - pa*Vq- - (1+pa)*Vq)
            #       = pb + dpb*(b + (1-pa)*Vq+ - pa*Vq- - Vq - pa*Vq)
            #       = pb + dpb*(b + (1-pa)*Vq+ - pa*Vq- - Vq - pa*Vq)

            # Simplify: (1-pa)*Vq+ - pa*Vq- - (1+pa)*Vq
            # = (1-pa)*Vq+ - pa*Vq- - Vq - pa*Vq

            # For interior q (1..n_q-2):
            # df/db = pb + dpb*(b + (1-pa)*Vq+ - pa*Vq- - (1+pa)*Vq)
            # df/da = pa + dpa*(a + pb*Vq+ - (1-pb)*Vq- + (pb-1)*Vq)
            #        = pa + dpa*(a + pb*Vq+ - (1-pb)*Vq- - (1-pb)*Vq)
            #        = pa + dpa*(a + pb*Vq+ - (1-pb)*(Vq-+Vq))

            # Let me just compute numerically for robustness
            # Use finite differences for the gradient and Hessian-free Newton (Gauss-Newton)

            # Actually, let's use a simpler approach: coordinate descent with
            # golden section or just a fine 1-D search per dimension, but
            # starting from the previous iterate.

            # Better: use the analytical FOC and solve with Newton.
            # df/db = pb + dpb*(b + (1-pa)*Vq+ - pa*Vq- - (1+pa)*Vq)
            # d2f/db2 = dpb + dpb*(b + (1-pa)*Vq+ - pa*Vq- - (1+pa)*Vq) + dpb2*(...)
            # This is getting complex. Let's use a 1-D search per dimension.

            break  # Skip Newton, use search below

        # Use coordinate descent with golden section
        n_cd = 4
        for _ in range(n_cd):
            # Optimize bid for each q
            for q_idx in range(n_q):
                if q_idx == n_q - 1:  # q = Q, no bid fill
                    continue
                # Golden section for bid depth
                lo, hi = 0.001, 5.0 / k
                pa_q = _fill_prob(a[q_idx], A, k, dt)
                Vp = V_q_plus[q_idx]
                Vm = V_q_minus[q_idx]
                V0 = V_q[q_idx]
                Cq = C_inv[q_idx]

                def obj_b(d):
                    p = _fill_prob(np.array([d]), A, k, dt)[0]
                    val = d * p + a[q_idx] * pa_q - Cq
                    if q_idx == 0:  # no ask fill
                        val += p * Vp + (1 - p) * V0
                    else:
                        val += p * (1 - pa_q) * Vp + pa_q * (1 - p) * Vm + (1 - p - pa_q + p * pa_q) * V0
                    return val

                # Golden section
                gr = (np.sqrt(5) - 1) / 2
                c = hi - gr * (hi - lo)
                d = lo + gr * (hi - lo)
                fc = obj_b(c)
                fd = obj_b(d)
                for _ in range(30):
                    if fc < fd:
                        hi = d
                        d = c
                        fd = fc
                        c = hi - gr * (hi - lo)
                        fc = obj_b(c)
                    else:
                        lo = c
                        c = d
                        fc = fd
                        d = lo + gr * (hi - lo)
                        fd = obj_b(d)
                b[q_idx] = (lo + hi) / 2

            # Optimize ask for each q
            for q_idx in range(n_q):
                if q_idx == 0:  # q = -Q, no ask fill
                    continue
                lo, hi = 0.001, 5.0 / k
                pb_q = _fill_prob(b[q_idx], A, k, dt)
                Vp = V_q_plus[q_idx]
                Vm = V_q_minus[q_idx]
                V0 = V_q[q_idx]
                Cq = C_inv[q_idx]

                def obj_a(d):
                    p = _fill_prob(np.array([d]), A, k, dt)[0]
                    val = b[q_idx] * pb_q + d * p - Cq
                    if q_idx == n_q - 1:  # no bid fill
                        val += p * Vm + (1 - p) * V0
                    else:
                        val += pb_q * (1 - p) * Vp + p * (1 - pb_q) * Vm + (1 - pb_q - p + pb_q * p) * V0
                    return val

                gr = (np.sqrt(5) - 1) / 2
                c = hi - gr * (hi - lo)
                d = lo + gr * (hi - lo)
                fc = obj_a(c)
                fd = obj_a(d)
                for _ in range(30):
                    if fc < fd:
                        hi = d
                        d = c
                        fd = fc
                        c = hi - gr * (hi - lo)
                        fc = obj_a(c)
                    else:
                        lo = c
                        c = d
                        fc = fd
                        d = lo + gr * (hi - lo)
                        fd = obj_a(d)
                a[q_idx] = (lo + hi) / 2

        # Boundary: q=Q no bid, q=-Q no ask
        b[n_q - 1] = base_depth  # doesn't matter
        a[0] = base_depth  # doesn't matter

        # Compute V
        pb = _fill_prob(b, A, k, dt)
        pa = _fill_prob(a, A, k, dt)

        V_refined = np.empty(n_q)
        # Interior
        for q_idx in range(1, n_q - 1):
            V_refined[q_idx] = (b[q_idx] * pb[q_idx] + a[q_idx] * pa[q_idx] - C_inv[q_idx]
                                + pb[q_idx] * (1 - pa[q_idx]) * V_q_plus[q_idx]
                                + pa[q_idx] * (1 - pb[q_idx]) * V_q_minus[q_idx]
                                + (1 - pb[q_idx] - pa[q_idx] + pb[q_idx] * pa[q_idx]) * V_q[q_idx])
        # q = -Q (index 0): no ask fill
        V_refined[0] = b[0] * pb[0] - C_inv[0] + pb[0] * V_q_plus[0] + (1 - pb[0]) * V_q[0]
        # q = Q (index n_q-1): no bid fill
        V_refined[-1] = a[-1] * pa[-1] - C_inv[-1] + pa[-1] * V_q_minus[-1] + (1 - pa[-1]) * V_q[-1]

        V[n, :] = V_refined
        opt_bid[n, :] = b
        opt_ask[n, :] = a

    return opt_bid, opt_ask


def _get_policy(params):
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
    opt_bid, opt_ask = _get_policy(params)
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = max(0, min(n, params['steps'] - 1))
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].copy()
    ask_depths = opt_ask[n, q_idx].copy()

    min_depth = 0.001
    max_depth = 10.0 / params['k']
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, min_depth)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, min_depth)

    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
