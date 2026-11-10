import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # 16 types: (v1, v2) with v1, v2 in {0,1,2,3}
    # Index: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n = 16

    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(n) / n
    else:
        probs = probs / total

    # LP formulation:
    # Variables: q1[0:16], q2[16:32], p[32:48]
    # Maximize: sum_t probs[t] * p[t]
    # Subject to:
    #   IC: p[t] - p[t'] >= v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t']) for all t != t'
    #   IR: p[t] <= v1t*q1[t] + v2t*q2[t]
    #   p[t] >= 0
    #   0 <= q1[t] <= 1, 0 <= q2[t] <= 1

    # Objective: minimize -sum_t probs[t] * p[t]
    c = np.zeros(48)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_ub = []
    b_ub = []

    # IC constraints: p[t] - p[t'] - v1t*q1[t] + v1t*q1[t'] - v2t*q2[t] + v2t*q2[t'] <= 0
    for t in range(n):
        v1t, v2t = types[t]
        for tp in range(n):
            if tp == t:
                continue
            row = np.zeros(48)
            row[t] = -v1t
            row[tp] = v1t
            row[16 + t] = -v2t
            row[16 + tp] = v2t
            row[32 + t] = 1.0
            row[32 + tp] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints: p[t] - v1t*q1[t] - v2t*q2[t] <= 0
    for t in range(n):
        v1t, v2t = types[t]
        row = np.zeros(48)
        row[t] = -v1t
        row[16 + t] = -v2t
        row[32 + t] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Nonnegative payments: -p[t] <= 0
    for t in range(n):
        row = np.zeros(48)
        row[32 + t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Check time before solving
    if time.time() > deadline - 5:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    # Solve LP with HiGHS
    try:
        options = {'presolve': True,
                   'dual_feasibility_tolerance': 1e-13,
                   'primal_feasibility_tolerance': 1e-13,
                   'time_limit': 30.0}
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs', options=options)
    except Exception:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if time.time() > deadline - 2:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if not result.success or result.x is None:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    q1 = np.clip(result.x[0:16], 0, 1)
    q2 = np.clip(result.x[16:32], 0, 1)
    p = np.clip(result.x[32:48], 0, None)

    # Post-process: enforce monotonicity of allocations (required for IC in this structure)
    # For additive values, IC implies q1 is non-decreasing in v1 and q2 is non-decreasing in v2
    # Actually, IC on the full grid implies both monotonicity and cross-consistency
    # Let's enforce monotonicity to help with numerical stability
    for v1 in range(3):
        for v2 in range(4):
            t_curr = v1 * 4 + v2
            t_next = (v1 + 1) * 4 + v2
            if q1[t_curr] > q1[t_next] + 1e-12:
                # Swap to enforce monotonicity (average them)
                avg = (q1[t_curr] + q1[t_next]) / 2
                q1[t_curr] = min(q1[t_curr], avg + 1e-12)
                q1[t_next] = max(q1[t_next], avg - 1e-12)

    for v1 in range(4):
        for v2 in range(3):
            t_curr = v1 * 4 + v2
            t_next = v1 * 4 + (v2 + 1)
            if q2[t_curr] > q2[t_next] + 1e-12:
                avg = (q2[t_curr] + q2[t_next]) / 2
                q2[t_curr] = min(q2[t_curr], avg + 1e-12)
                q2[t_next] = max(q2[t_next], avg - 1e-12)

    # Re-solve with the monotone allocations fixed, optimizing only payments
    # This is a smaller LP: just optimize p given q
    # Maximize sum_t probs[t] * p[t]
    # Subject to:
    #   IC: p[t] - p[t'] >= v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t'])
    #   IR: p[t] <= v1t*q1[t] + v2t*q2[t]
    #   p[t] >= 0

    if time.time() < deadline - 8:
        c_p = np.zeros(n)
        for t in range(n):
            c_p[t] = -probs[t]

        A_ub_p = []
        b_ub_p = []

        # IC constraints for fixed q
        for t in range(n):
            v1t, v2t = types[t]
            for tp in range(n):
                if tp == t:
                    continue
                row = np.zeros(n)
                row[t] = 1.0
                row[tp] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(v1t * (q1[tp] - q1[t]) + v2t * (q2[tp] - q2[t]))

        # IR constraints
        for t in range(n):
            v1t, v2t = types[t]
            row = np.zeros(n)
            row[t] = 1.0
            A_ub_p.append(row)
            b_ub_p.append(v1t * q1[t] + v2t * q2[t])

        # Nonnegative payments
        for t in range(n):
            row = np.zeros(n)
            row[t] = -1.0
            A_ub_p.append(row)
            b_ub_p.append(0.0)

        A_ub_p = np.array(A_ub_p)
        b_ub_p = np.array(b_ub_p)
        bounds_p = [(0, None)] * n

        try:
            result_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p,
                               method='highs',
                               options={'presolve': True,
                                        'dual_feasibility_tolerance': 1e-14,
                                        'primal_feasibility_tolerance': 1e-14,
                                        'time_limit': 15.0})
            if result_p.success and result_p.x is not None:
                p_new = np.clip(result_p.x, 0, None)
                # Check if this is better
                if np.dot(probs, p_new) >= np.dot(probs, p) - 1e-12:
                    p = p_new
        except Exception:
            pass

    # Verify constraints and compute regret
    max_regret = 0.0
    for t in range(n):
        v1t, v2t = types[t]
        util_t = v1t * q1[t] + v2t * q2[t] - p[t]
        for tp in range(n):
            if tp == t:
                continue
            util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
            regret = util_tp - util_t
            if regret > max_regret:
                max_regret = regret

    if max_regret > 1e-8:
        # Iterative regret fix: adjust payments to ensure IC
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            changed = False
            for t in range(n):
                v1t, v2t = types[t]
                min_p = 0.0
                for tp in range(n):
                    if tp == t:
                        continue
                    required = p[tp] + v1t * (q1[t] - q1[tp]) + v2t * (q2[t] - q2[tp])
                    if required > min_p:
                        min_p = required
                max_p = v1t * q1[t] + v2t * q2[t]
                if min_p > max_p:
                    # Infeasible; reduce allocation slightly
                    # Scale down allocations for this type
                    scale = max_p / max(min_p, 1e-12) * 0.99
                    q1[t] *= scale
                    q2[t] *= scale
                    min_p = v1t * q1[t] + v2t * q2[t]
                if min_p > p[t] + 1e-12:
                    p[t] = min_p
                    changed = True
            if not changed:
                break
        p = np.clip(p, 0, None)

    allocations = np.column_stack([q1, q2])
    payments = p

    return {'allocations': allocations, 'payments': payments}
