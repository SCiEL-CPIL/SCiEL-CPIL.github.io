import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Build edge masks
    mask_cw = np.zeros((D, n), dtype=int)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            mask_cw[d, e % n] = 1
            e = (e + 1) % n
    mask_ccw = 1 - mask_cw

    weights = np.array([w for _, _, w in demands], dtype=float)

    start_time = time.time()
    deadline = start_time + 600.0  # safety margin below the 630s kill

    def get_loads(x):
        # x: 0/1 array; 1 => CCW
        return (weights[:, None] * (x[:, None].astype(float) * mask_ccw + (1 - x[:, None].astype(float)) * mask_cw)).sum(axis=0)

    # ILP via scipy.optimize.milp
    best_x = None
    best_val = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception:
        milp = None

    if milp is not None:
        # c: minimize t (last variable), x vars have 0 cost
        c = np.zeros(D + 1)
        c[D] = 1.0

        # Variables: x[0..D-1] in {0,1}, t continuous in [0, big]
        integrality = np.ones(D + 1)
        integrality[D] = 0

        big = float(weights.sum())  # upper bound on any edge load

        A = np.zeros((n, D + 1))
        b_lo = np.zeros(n)
        b_hi = np.full(n, np.inf)
        for e in range(n):
            # load_e = sum_d w_d*(cw_mask[d,e] + (ccw_mask[d,e]-cw_mask[d,e])*x_d) <= t
            for d in range(D):
                A[e, d] = weights[d] * (mask_ccw[d, e] - mask_cw[d, e])
            A[e, D] = -1.0
            b_lo[e] = -np.inf
            b_hi[e] = -float((weights[:, None] * mask_cw[:, e]).sum())  # subtract constant
        # Rearranged: sum_d w_d*(ccw_mask-cw_mask)*x_d - t <= -const
        # i.e. A_row . vars <= b_hi  and A_row . vars >= b_lo
        # We set b_lo = -inf, b_hi = -const
        # But LinearConstraint wants (lb, ub) on A@x.
        # A@x = sum_d A[e,d]*x_d + A[e,D]*t = sum_d w_d*(ccw-cw)*x_d - t
        # We want sum_d w_d*(ccw-cw)*x_d - t + const <= 0  =>  A@x <= -const
        # So lb=-inf, ub=-const. Good.

        lc = LinearConstraint(A, lb=-np.inf, ub=b_hi)
        bounds = Bounds(lb=np.zeros(D + 1), ub=np.concatenate([np.ones(D), [big]]))

        res = milp(c=c, integrality=integrality, constraints=lc, bounds=bounds)

        if res.status == 0 and res.x is not None:
            x_ilp = (res.x[:D] > 0.5).astype(int)
            best_x = x_ilp
            best_val = float(res.fun)
        else:
            best_x = None

    # Fallback / polish: local search to guarantee a valid answer and possibly improve
    def get_max_load(x):
        return float(np.max(get_loads(x)))

    if best_x is None:
        # Start from all CW
        x = np.zeros(D, dtype=int)
    else:
        x = best_x.copy()

    cur_score = get_max_load(x)
    best_local = x.copy()
    best_local_score = cur_score

    # Deterministic first-improvement local search: flip any single demand if it reduces max load
    improved = True
    while improved and (time.time() < deadline - 5.0):
        improved = False
        for d in range(D):
            x[d] = 1 - x[d]
            s = get_max_load(x)
            if s < best_local_score - 1e-12:
                best_local_score = s
                best_local = x.copy()
                cur_score = s
                improved = True
            else:
                x[d] = 1 - x[d]
    x = best_local.copy()
    best_local_score = get_max_load(x)

    # If ILP gave a better result, prefer it
    if best_x is not None:
        ilp_score = get_max_load(best_x)
        if ilp_score < best_local_score - 1e-12:
            x = best_x.copy()
            best_local_score = ilp_score

    # Convert to required format: >=0.5 for CCW, <0.5 for CW
    result = x.astype(float)
    return result
