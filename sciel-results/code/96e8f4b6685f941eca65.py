"""Incentive-compatible finite auction design solver.

Strategy: enumerate monotone threshold-based allocation rules for a single
additive bidder with two goods over values {0,1,2,3}, compute the unique
payment rule implied by IC/IR, and select the highest expected revenue.
"""
import time
import itertools
import numpy as np


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0

    values = [0, 1, 2, 3]
    n = 4

    # Parse distributions
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    if p1.sum() > 0:
        p1 = p1 / p1.sum()
    else:
        p1 = np.ones(4) / 4
    if p2.sum() > 0:
        p2 = p2 / p2.sum()
    else:
        p2 = np.ones(4) / 4

    # Joint probability over (v1, v2), flattened with v1 major
    joint = np.outer(p1, p2).flatten()  # index t = v1*4 + v2
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = 16

    def eval_rule(x1, x2):
        """x1[v1] = prob of good 1, x2[v2] = prob of good 2 (monotone).
        Returns (allocations, payments, revenue) or None if infeasible."""
        # Compute implied payments via path integral along canonical path
        # u(v1, v2) = sum_{i=1}^{v1} x1[i] + sum_{j=1}^{v2} x2[j]
        # p(v1, v2) = v1*x1[v1] + v2*x2[v2] - u(v1, v2)
        payments = np.zeros(n_types)
        allocations = np.zeros((n_types, 2))
        for v1 in range(n):
            for v2 in range(n):
                idx = v1 * n + v2
                allocations[idx, 0] = x1[v1]
                allocations[idx, 1] = x2[v2]
                u = 0.0
                for i in range(1, v1 + 1):
                    u += x1[i]
                for j in range(1, v2 + 1):
                    u += x2[j]
                p = v1 * x1[v1] + v2 * x2[v2] - u
                payments[idx] = p

        # Check IR: p <= value
        for t in range(n_types):
            v1_t, v2_t = types[t]
            val = v1_t * x1[v1_t] + v2_t * x2[v2_t]
            if payments[t] > val + 1e-8:
                return None
        # Check IC: for all t, tp
        for t in range(n_types):
            v1_t, v2_t = types[t]
            u_tt = v1_t * x1[v1_t] + v2_t * x2[v2_t] - payments[t]
            for tp in range(n_types):
                v1_tp, v2_tp = types[tp]
                u_tt_prime = v1_t * x1[v1_tp] + v2_t * x2[v2_tp] - payments[tp]
                if u_tt < u_tt_prime - 1e-8:
                    return None

        # Check nonnegative payments
        if np.any(payments < -1e-8):
            return None

        revenue = float(np.dot(joint, payments))
        return allocations, payments, revenue

    best_alloc = np.zeros((16, 2))
    best_pay = np.zeros(16)
    best_rev = -np.inf

    # Enumerate monotone allocation rules: x1[v] and x2[v] in {0,1} with
    # x1[v] <= x1[v+1], x2[v] <= x2[v+1]. This is equivalent to choosing
    # a threshold k1 for good 1 and k2 for good 2.
    # But also allow the "separable" family where allocation depends only
    # on each good's value independently (which is the natural structure).
    # For a single additive bidder, the IC-optimal mechanism is of the form:
    # x1(v1) = 1 if v1 >= t1, else 0; x2(v2) = 1 if v2 >= t2, else 0
    # for some thresholds t1, t2 in {0,1,2,3,4} (4 means never allocate).

    for t1 in range(n + 1):
        for t2 in range(n + 1):
            x1 = np.array([1.0 if v >= t1 else 0.0 for v in range(n)])
            x2 = np.array([1.0 if v >= t2 else 0.0 for v in range(n)])
            res = eval_rule(x1, x2)
            if res is not None:
                alloc, pay, rev = res
                if rev > best_rev:
                    best_rev = rev
                    best_alloc = alloc.copy()
                    best_pay = pay.copy()

    # Also try a broader family: x1[v] in {0,1} monotone (threshold), but
    # allow "both goods" to be allocated based on a joint rule that is still
    # separable. The threshold family above should capture the main cases.
    # Let's also try allowing x1[v] to be in {0, 0.5, 1} for more flexibility,
    # but maintain monotonicity. This increases the search space.

    # For monotone {0, 0.5, 1} rules: x1[v] can be 0, 0.5, or 1, non-decreasing.
    # Number of such sequences: C(n+3, 3) = C(7,3) = 35 per good. 35*35 = 1225.
    # This is still very fast.

    for seq1 in itertools.product([0, 0.5, 1], repeat=n):
        if time.time() > deadline:
            break
        # Check monotonicity
        mono = all(seq1[i] <= seq1[i+1] for i in range(n-1))
        if not mono:
            continue
        for seq2 in itertools.product([0, 0.5, 1], repeat=n):
            if time.time() > deadline:
                break
            mono2 = all(seq2[i] <= seq2[i+1] for i in range(n-1))
            if not mono2:
                continue
            x1 = np.array(seq1, dtype=float)
            x2 = np.array(seq2, dtype=float)
            res = eval_rule(x1, x2)
            if res is not None:
                alloc, pay, rev = res
                if rev > best_rev:
                    best_rev = rev
                    best_alloc = alloc.copy()
                    best_pay = pay.copy()

    # Final safety: ensure we return a valid mechanism
    if best_rev < 0:
        # Fallback: zero allocation
        best_alloc = np.zeros((16, 2))
        best_pay = np.zeros(16)
    else:
        # Final validation
        alloc = best_alloc
        pay = best_pay
        ok = True
        for t in range(n_types):
            v1_t, v2_t = types[t]
            u_tt = v1_t * alloc[t, 0] + v2_t * alloc[t, 1] - pay[t]
            if u_tt < -1e-8:
                ok = False
                break
            for tp in range(n_types):
                u_tt_prime = v1_t * alloc[tp, 0] + v2_t * alloc[tp, 1] - pay[tp]
                if u_tt < u_tt_prime - 1e-8:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            best_alloc = np.zeros((16, 2))
            best_pay = np.zeros(16)

    return {'allocations': best_alloc, 'payments': best_pay}
