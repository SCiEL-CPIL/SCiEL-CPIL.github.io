"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, three fixed distributions.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    t0 = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    # Type enumeration: 16 types, (v1, v2) with vi in {0,1,2,3}
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    # types[i] = (v1_i, v2_i), i = 0..15

    # Parse distribution_spec
    probs = _parse_distribution(distribution_spec)

    # Solve the separable Myerson LP exactly
    allocations, payments = _solve_separable_lp(probs, types)

    # Verify and repair
    allocations, payments, is_valid = _verify_and_repair(allocations, payments, types, probs)

    if not is_valid:
        payments[:] = 0.0
        allocations, payments, is_valid = _verify_and_repair(allocations, payments, types, probs)

    # Polish pass: try to increase payments while maintaining IC
    if time.time() - t0 < DEADLINE - 15:
        allocations, payments = _polish_pass(allocations, payments, types, probs, time_budget=10.0)
        allocations, payments, is_valid = _verify_and_repair(allocations, payments, types, probs)
        if not is_valid:
            # Revert to pre-polish
            allocations, payments = _solve_separable_lp(probs, types)
            allocations, payments, _ = _verify_and_repair(allocations, payments, types, probs)

    # Final cleanup
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    allocations[allocations < 1e-12] = 0.0
    allocations[allocations > 1.0 - 1e-12] = 1.0
    payments[payments < 1e-12] = 0.0

    return {'allocations': allocations, 'payments': payments}


def _parse_distribution(distribution_spec):
    """Parse distribution_spec into a probability vector of length 16."""
    if isinstance(distribution_spec, dict):
        if 'type_probs' in distribution_spec:
            probs = np.array(distribution_spec['type_probs'], dtype=float)
        elif 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            for k, v in distribution_spec.items():
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 16:
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == 16:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0

    probs = np.maximum(probs, 0.0)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.ones(16) / 16.0

    return probs


def _solve_separable_lp(probs, types):
    """
    Solve the separable Myerson LP for a single additive bidder with two goods.
    
    For each good j independently, choose q_j(v) in [0,1] to maximize:
        sum_{v=0}^{3} q_j(v) * [v * P(v_j = v) - P(v_j > v)]
    
    This is a simple LP: set q_j(v) = 1 if coefficient > 0, else 0.
    Payments are derived from the IC formula.
    """
    p1 = np.zeros(4)
    p2 = np.zeros(4)
    for i in range(16):
        v1, v2 = types[i]
        p1[v1] += probs[i]
        p2[v2] += probs[i]

    def optimal_q(p):
        q = np.zeros(4)
        for v in range(4):
            if v == 3:
                coeff = 3.0 * p[3]
            else:
                coeff = v * p[v] - np.sum(p[v+1:4])
            q[v] = 1.0 if coeff > 1e-14 else 0.0
        return q

    q1 = optimal_q(p1)
    q2 = optimal_q(p2)

    allocations = np.zeros((16, 2))
    payments = np.zeros(16)

    for i in range(16):
        v1, v2 = types[i]
        allocations[i, 0] = q1[v1]
        allocations[i, 1] = q2[v2]

        pay1 = v1 * q1[v1] - np.sum(q1[:v1]) if v1 > 0 else 0.0
        pay2 = v2 * q2[v2] - np.sum(q2[:v2]) if v2 > 0 else 0.0
        payments[i] = pay1 + pay2

    return allocations, payments


def _verify_and_repair(allocations, payments, types, probs):
    """
    Verify IC, IR, and feasibility constraints. If violated, repair by
    projecting payments to satisfy IC via the payment formula.
    """
    n = 16
    # Check allocations in [0,1]
    if np.any(allocations < -1e-12) or np.any(allocations > 1.0 + 1e-12):
        allocations = np.clip(allocations, 0.0, 1.0)

    # Check nonnegative payments
    if np.any(payments < -1e-12):
        payments = np.maximum(payments, 0.0)

    # Check IC: for all i, j:
    #   v_i . q_i - p_i >= v_i . q_j - p_j
    #   i.e., p_i - p_j <= v_i . (q_i - q_j)
    ic_violated = False
    max_violation = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lhs = payments[i] - payments[j]
            rhs = np.dot(np.array(types[i], dtype=float),
                         allocations[i] - allocations[j])
            if lhs - rhs > 1e-8:
                ic_violated = True
                max_violation = max(max_violation, lhs - rhs)

    if ic_violated:
        # Repair: recompute payments from allocations using the IC formula
        # For a single bidder, the payment is determined by the allocation rule
        # p(v) = v . q(v) - sum over lower values
        # But since we have a general allocation, use the formula:
        # p(i) = v_i . q_i - min_j (v_i . q_j - p_j) ... 
        # Actually, the standard approach: set p(0,0) = 0, then for each type,
        # p(i) = v_i . q_i - max over "lower" types of (v_i . q_j - p_j)
        # For a lattice, we can use the formula based on the allocation rule.
        
        # For a single additive bidder, if allocation is separable (q_j depends only on v_j),
        # we can recover payments. Otherwise, use the general IC constraint.
        
        # General approach: set p(0) = 0 (IR for type (0,0)), then for each type i,
        # p(i) = v_i . q_i - max_{j} (v_i . q_j - p(j)) where the max is over types j
        # that are "below" i in some sense. Actually, the correct formula is:
        # p(i) = v_i . q_i - max_j [v_i . q_j - p(j)]
        # But this is circular. The standard solution:
        # p(i) = v_i . q_i - sum over the path from (0,0) to i of marginal contributions.
        
        # Simpler: for a single bidder, the payment is:
        # p(v1, v2) = v1*q1(v1) + v2*q2(v2) - sum_{k=0}^{v1-1} q1(k, v2) - sum_{k=0}^{v2-1} q2(v1, k)
        # But if allocation is not separable, this doesn't work.
        
        # Most robust: set p(0,0) = 0, and for each other type, set
        # p(i) = v_i . q_i - max_j (v_i . q_j - p(j))
        # Process in order of increasing v1+v2.
        
        p_new = np.zeros(n)
        # Process types in order of increasing v1 + v2
        order = sorted(range(n), key=lambda i: types[i][0] + types[i][1])
        p_new[order[0]] = 0.0  # type (0,0)
        
        for idx in range(1, n):
            i = order[idx]
            v_i = np.array(types[i], dtype=float)
            # p(i) = v_i . q_i - max_j (v_i . q_j - p(j))
            min_term = np.inf
            for j in range(n):
                if j == i:
                    continue
                term = np.dot(v_i, allocations[j]) - p_new[j]
                if term < min_term:
                    min_term = term
            p_new[i] = np.dot(v_i, allocations[i]) - min_term
            # IR: p(i) >= 0
            if p_new[i] < 0:
                p_new[i] = 0.0
        
        payments = p_new
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        # Re-check IC
        ic_violated = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                lhs = payments[i] - payments[j]
                rhs = np.dot(np.array(types[i], dtype=float),
                             allocations[i] - allocations[j])
                if lhs - rhs > 1e-8:
                    ic_violated = True
                    break
            if ic_violated:
                break
        
        if ic_violated:
            # Last resort: zero payments
            payments = np.zeros(n)
            return allocations, payments, False
        
        return allocations, payments, True

    # Check IR: for all i, v_i . q_i - p_i >= 0
    for i in range(n):
        util = np.dot(np.array(types[i], dtype=float), allocations[i]) - payments[i]
        if util < -1e-8:
            # Repair: reduce payment
            payments[i] = np.dot(np.array(types[i], dtype=float), allocations[i])
    
    return allocations, payments, True


def _polish_pass(allocations, payments, types, probs, time_budget=10.0):
    """
    Try to increase payments while maintaining IC and IR constraints.
    For each type, try to increase its payment by a small amount and check IC.
    """
    n = 16
    t_start = time.time()
    step = 1e-6
    improved = True

    while improved and time.time() - t_start < time_budget:
        improved = False
        for i in range(n):
            if time.time() - t_start >= time_budget:
                break
            # Try to increase payment[i] by step
            new_payments = payments.copy()
            new_payments[i] += step
            # Check IC
            ic_ok = True
            for a in range(n):
                for b in range(n):
                    if a == b:
                        continue
                    lhs = new_payments[a] - new_payments[b]
                    rhs = np.dot(np.array(types[a], dtype=float),
                                 allocations[a] - allocations[b])
                    if lhs - rhs > 1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break
            # Check IR
            if ic_ok:
                for a in range(n):
                    util = np.dot(np.array(types[a], dtype=float), allocations[a]) - new_payments[a]
                    if util < -1e-8:
                        ic_ok = False
                        break
            if ic_ok:
                payments = new_payments
                improved = True
            else:
                # Try smaller step
                new_payments = payments.copy()
                new_payments[i] += step * 0.1
                ic_ok = True
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        lhs = new_payments[a] - new_payments[b]
                        rhs = np.dot(np.array(types[a], dtype=float),
                                     allocations[a] - allocations[b])
                        if lhs - rhs > 1e-8:
                            ic_ok = False
                            break
                    if not ic_ok:
                        break
                if ic_ok:
                    for a in range(n):
                        util = np.dot(np.array(types[a], dtype=float), allocations[a]) - new_payments[a]
                        if util < -1e-8:
                            ic_ok = False
                            break
                if ic_ok:
                    payments = new_payments
                    improved = True

    return allocations, payments
# EVOLVE-BLOCK-END
