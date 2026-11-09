# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute edge incidence for each demand for both directions
    # clockwise_mask[d] : boolean array of length n, True if edge e is on clockwise path
    # ccw_mask[d] : boolean array of length n, True if edge e is on counter-clockwise path
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw = np.zeros(n, dtype=bool)
        ccw = np.zeros(n, dtype=bool)
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            for e in range(s, t):
                cw[e] = True
        else:
            for e in range(s, n):
                cw[e] = True
            for e in range(0, t):
                cw[e] = True
        # Counter-clockwise: all other edges
        ccw = ~cw
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    cw_masks = np.array(cw_masks)  # shape (D, n)
    ccw_masks = np.array(ccw_masks)  # shape (D, n)
    weights = np.array([w for s, t, w in demands])  # shape (D,)

    def objective(x):
        """
        x: continuous routing variables in [0, 1], shape (D,)
        x[d] = 0 -> clockwise, x[d] = 1 -> counter-clockwise
        Load on edge e = sum_d weights[d] * ( (1-x[d]) * cw_masks[d,e] + x[d] * ccw_masks[d,e] )
        We minimize the log-sum-exp approximation of max load.
        """
        # Compute load for each edge
        # load[e] = sum_d w[d] * ((1-x[d]) * cw[d,e] + x[d] * ccw[d,e])
        # = sum_d w[d] * cw[d,e] + sum_d w[d] * (ccw[d,e] - cw[d,e]) * x[d]
        base_load = weights @ cw_masks  # shape (n,)
        delta = weights @ (ccw_masks - cw_masks)  # shape (n,)
        loads = base_load + delta * x  # shape (n,)

        # Log-sum-exp approximation of max load
        alpha = 50.0
        # Use stable log-sum-exp
        m = np.max(loads)
        if m == 0:
            return 0.0
        lse = m + np.log(np.sum(np.exp(alpha * (loads - m)))) / alpha
        return lse

    def gradient(x):
        base_load = weights @ cw_masks
        delta = weights @ (ccw_masks - cw_masks)
        loads = base_load + delta * x
        alpha = 50.0
        m = np.max(loads)
        if m == 0:
            return np.zeros(D)
        exp_terms = np.exp(alpha * (loads - m))
        softmax = exp_terms / np.sum(exp_terms)
        # d(loads)/dx[d] = delta
        # d(lse)/dx[d] = sum_e softmax[e] * delta[e] ... wait
        # lse = m + (1/alpha) * log(sum(exp(alpha*(loads-m))))
        # d(lse)/d(loads[e]) = exp(alpha*(loads[e]-m)) / sum(exp(alpha*(loads'-m))) = softmax[e]
        # d(loads[e])/dx[d] = delta[e]
        # So d(lse)/dx[d] = sum_e softmax[e] * delta[e] ... no, that's wrong.
        # d(lse)/dx[d] = sum_e (d lse/d loads[e]) * (d loads[e]/dx[d])
        # = sum_e softmax[e] * delta[e]
        # But delta[e] is the same for all d? No, delta is shape (n,), and x is shape (D,).
        # loads = base_load + delta * x where delta is (n,) and x is (D,).
        # This is element-wise: loads[e] = base_load[e] + sum_d delta[e] * x[d]? No.
        # Wait, my formulation: loads = base_load + delta * x
        # base_load: (n,), delta: (n,), x: (D,)
        # This doesn't work dimensionally unless I'm doing something else.
        
        # Let me recompute. 
        # load[e] = sum_d w[d] * ((1-x[d]) * cw[d,e] + x[d] * ccw[d,e])
        # = sum_d w[d] * cw[d,e] + sum_d w[d] * (ccw[d,e] - cw[d,e]) * x[d]
        # So load[e] = base_load[e] + sum_d (w[d] * (ccw[d,e] - cw[d,e])) * x[d]
        # Let A[e,d] = w[d] * (ccw[d,e] - cw[d,e])
        # Then load = base_load + A @ x
        # d(load[e])/dx[d] = A[e,d]
        # d(lse)/dx[d] = sum_e softmax[e] * A[e,d]
        
        # A = (ccw_masks - cw_masks).T * weights  -> shape (n, D)
        # Actually A[e,d] = w[d] * (ccw[d,e] - cw[d,e])
        A = (ccw_masks - cw_masks).T * weights  # shape (n, D)
        grad = softmax @ A  # shape (D,)
        return grad

    # Initial guess: try all 0s, all 1s, and a random start
    best_x = None
    best_val = np.inf

    initial_guesses = [
        np.zeros(D),
        np.ones(D),
        np.random.rand(D),
    ]

    for x0 in initial_guesses:
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * D,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        return np.zeros(D)

    # Round to binary
    routing = (best_x >= 0.5).astype(float)

    # Try local search: flip each bit and see if it improves max load
    def max_load(routing_vec):
        loads = np.zeros(n)
        for d in range(D):
            if routing_vec[d] >= 0.5:
                loads += weights[d] * ccw_masks[d]
            else:
                loads += weights[d] * cw_masks[d]
        return np.max(loads)

    current_load = max_load(routing)
    improved = True
    while improved:
        improved = False
        for d in range(D):
            new_routing = routing.copy()
            new_routing[d] = 1.0 - new_routing[d]
            new_load = max_load(new_routing)
            if new_load < current_load - 1e-10:
                routing = new_routing
                current_load = new_load
                improved = True

    return routing


# EVOLVE-BLOCK-END
