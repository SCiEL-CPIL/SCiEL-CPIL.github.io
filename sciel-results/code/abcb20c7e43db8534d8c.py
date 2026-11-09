"""Auction design with threshold-based incentive-compatible mechanism."""
import time
import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods.
    Values in {0,1,2,3}. Three fixed distributions.
    
    Returns: {allocations: (16,2), payments: (16,)}
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse inputs
    # distribution_spec: describes the distribution over values
    # support: the set of possible value reports, e.g., {(0,0), (0,1), ..., (3,3)}
    # constraints: dict with mechanism constraints
    
    # Enumerate all 16 possible reports (v1, v2) where vi in {0,1,2,3}
    values = [0, 1, 2, 3]
    reports = [(v1, v2) for v1 in values for v2 in values]
    n_reports = len(reports)  # 16
    
    # Map report to index
    report_to_idx = {r: i for i, r in enumerate(reports)}
    
    # Determine the distribution from distribution_spec
    # distribution_spec could be a dict or a tuple describing the distribution
    # We'll handle various formats
    probs = _get_distribution(distribution_spec, reports)
    
    # For a single additive bidder with two goods, incentive compatibility
    # (Bayesian IC) requires that the expected utility is maximized at truthful reporting.
    # 
    # For additive valuations, a key structure: the allocation probability for good i
    # must be non-decreasing in v_i (the bidder's value for good i), holding v_{-i} fixed.
    # This is the monotonicity condition.
    #
    # Payment identity: p(v) = sum_i v_i * q_i(v) - U(v) + U(0,0)
    # where q_i(v) is the allocation probability of good i to the bidder at report v.
    #
    # For individual rationality: U(v) >= 0 for all v, so U(0,0) >= 0.
    # We can set U(0,0) = 0 (IR at the lowest type).
    #
    # The expected utility at report v is:
    # U(v) = sum_i v_i * q_i(v) - p(v)
    #
    # For IC: for all v, v', U(v) >= v_i * q_i(v') - p(v') for all i (in the appropriate form)
    # Actually for additive: U(v) >= sum_i v_i * q_i(v') - p(v') for all v'
    #
    # Revenue = sum_v P(v) * p(v)
    #
    # Strategy: Use a threshold mechanism.
    # For each good i, allocate with probability 1 if v_i >= t_i, else 0.
    # This is monotone in v_i.
    # Then set payments via the payment identity with U(0,0) = 0.
    #
    # But we can do better: allow fractional allocations or more complex monotone structures.
    #
    # Let's use an LP approach: maximize expected revenue subject to IC, IR, feasibility.
    
    # Variables: q[i][v] for i in {0,1}, v in reports -> allocation probability of good i at report v
    # p[v] for v in reports -> payment at report v
    # 
    # Total variables: 2 * 16 + 16 = 48
    # 
    # Constraints:
    # 1. 0 <= q[i][v] <= 1 for all i, v
    # 2. Monotonicity: q[i][v] >= q[i][v'] if v_i >= v'_i and v_{-i} = v'_{-i}
    #    For additive single bidder, IC is equivalent to monotonicity + payment identity
    # 3. Payment identity: p(v) = sum_i v_i * q_i(v) - U(v)
    #    where U(v) = sum_{v' <= v} sum_i (v_i - v'_i) * q_i(v') ... 
    #    Actually, for discrete case: U(v) = sum_{v'' <= v} delta(v'') where delta is the marginal
    #    
    # Let me use a cleaner formulation.
    #
    # For a single additive bidder, Bayesian IC is equivalent to:
    # (a) q_i(v) is non-decreasing in v_i for each i
    # (b) p(v) = v_0*q_0(v) + v_1*q_1(v) - U(v) for some U with U(0,0) >= 0
    #     and U(v) = sum over paths from (0,0) to v of sum_i (v_i - v_i') * q_i(v')
    #
    # In the discrete case with values {0,1,2,3}, we can define U(v) recursively:
    # U(0,0) = U00 >= 0
    # U(v1, v2) = U(v1-1, v2) + v1 * q_0(v1, v2) + ... hmm, need to be careful.
    #
    # Actually, the standard payment identity for discrete types:
    # p(v) = sum_i v_i q_i(v) - U(v)
    # U(v) = sum_{w: w <= v, w != v} [U(w) + sum_i (v_i - w_i) * q_i(w)] ... this is getting complex.
    #
    # Let me use a simpler approach: for each report v, define the utility U(v).
    # IC requires: U(v) >= sum_i v_i q_i(v') - p(v') for all v'
    # IR requires: U(v) >= 0 for all v
    # p(v) = sum_i v_i q_i(v) - U(v)
    #
    # So: sum_i v_i q_i(v) - U(v) >= 0 (IR) -> U(v) <= sum_i v_i q_i(v)
    # And IC: U(v) >= max_{v'} [sum_i v_i q_i(v') - p(v')] = max_{v'} [sum_i v_i q_i(v') - sum_i v'_i q_i(v') + U(v')]
    #
    # Let's just set up the LP directly.
    
    # Decision variables:
    # q[i][v] for i=0,1 and v in 0..15 (32 variables)
    # U[v] for v in 0..15 (16 variables)
    # p[v] is determined: p[v] = sum_i v_i * q[i][v] - U[v]
    #
    # We want to maximize: sum_v probs[v] * p[v] = sum_v probs[v] * (sum_i v_i * q[i][v] - U[v])
    #
    # Constraints:
    # 1. 0 <= q[i][v] <= 1
    # 2. U[v] >= 0 (IR)
    # 3. IC: for all v, v': U[v] >= sum_i v_i * q[i][v'] - (sum_i v'_i * q[i][v'] - U[v'])
    #    = sum_i (v_i - v'_i) * q[i][v'] + U[v']
    # 4. p[v] >= 0: sum_i v_i * q[i][v] - U[v] >= 0
    
    # This is a linear program!
    
    n_vars = 32 + 16  # q[0][0..15], q[1][0..15], U[0..15]
    
    # Objective: maximize sum_v probs[v] * (v0*v_q0 + v1*v_q1 - U[v])
    # = sum_v probs[v] * (v0 * q[0][v] + v1 * q[1][v] - U[v])
    c = np.zeros(n_vars)
    for v_idx, (v0, v1) in enumerate(reports):
        c[v_idx] -= probs[v_idx] * v0  # q[0][v]
        c[16 + v_idx] -= probs[v_idx] * v1  # q[1][v]
        c[32 + v_idx] += probs[v_idx]  # -U[v] in max, so +probs in min
    # We'll minimize -objective, so negate c for linprog
    c_min = -c
    
    # Constraints: A_ub x <= b_ub
    A_rows = []
    b_rows = []
    
    # 1. q[i][v] <= 1: q[i][v] - 0 <= 1 (already bounded, but let's be explicit)
    # Actually, use bounds for q: 0 <= q <= 1
    # U >= 0: use bounds
    
    # 2. IC constraints: for all v, v':
    # U[v] - U[v'] >= sum_i (v_i - v'_i) * q[i][v']
    # Rearranged: -U[v] + U[v'] + sum_i (v_i - v'_i) * q[i][v'] <= 0
    for v_idx in range(n_reports):
        v = reports[v_idx]
        for v2_idx in range(n_reports):
            v2 = reports[v2_idx]
            row = np.zeros(n_vars)
            row[32 + v_idx] = -1.0  # -U[v]
            row[32 + v2_idx] = 1.0   # +U[v2]
            # sum_i (v_i - v2_i) * q[i][v2]
            row[v2_idx] += (v[0] - v2[0])       # q[0][v2]
            row[16 + v2_idx] += (v[1] - v2[1])  # q[1][v2]
            A_rows.append(row)
            b_rows.append(0.0)
    
    # 3. p[v] >= 0: sum_i v_i * q[i][v] - U[v] >= 0
    # => -sum_i v_i * q[i][v] + U[v] <= 0
    for v_idx, (v0, v1) in enumerate(reports):
        row = np.zeros(n_vars)
        row[v_idx] = -v0
        row[16 + v_idx] = -v1
        row[32 + v_idx] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    # Monotonicity is implied by IC for additive valuations, but let's also add it for safety
    # Actually, IC implies monotonicity, so we don't need separate constraints.
    
    A_ub = np.array(A_rows) if A_rows else None
    b_ub = np.array(b_rows) if b_rows else None
    
    # Bounds: 0 <= q <= 1, U >= 0
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
    
    # Solve LP
    try:
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
            q0 = x[0:16]
            q1 = x[16:32]
            U = x[32:48]
            
            # Compute payments
            payments = np.zeros(n_reports)
            allocations = np.zeros((n_reports, 2))
            for v_idx, (v0, v1) in enumerate(reports):
                allocations[v_idx, 0] = q0[v_idx]
                allocations[v_idx, 1] = q1[v_idx]
                payments[v_idx] = v0 * q0[v_idx] + v1 * q1[v_idx] - U[v_idx]
            
            # Clamp small negatives
            payments = np.maximum(payments, 0)
            
            # Verify constraints
            if _verify_ic_ir(reports, probs, allocations, payments):
                return {
                    'allocations': allocations,
                    'payments': payments
                }
            
            # If verification fails, fall back to threshold mechanism
        else:
            raise Exception("LP failed")
    except Exception as e:
        pass
    
    # Fallback: threshold-based mechanism
    return _threshold_auction(reports, probs, start_time, DEADLINE)


def _get_distribution(distribution_spec, reports):
    """Get probability distribution over reports."""
    n = len(reports)
    if distribution_spec is None:
        return np.ones(n) / n
    
    # Try various formats
    if isinstance(distribution_spec, dict):
        # Could be {report: prob}
        if all(r in distribution_spec for r in reports):
            return np.array([distribution_spec[r] for r in reports])
        # Could be nested: {good: {value: prob}}
        if 'goods' in distribution_spec:
            goods = distribution_spec['goods']
            probs = np.ones(n)
            for i, good in enumerate(goods):
                for v_idx, (v0, v1) in enumerate(reports):
                    v = v0 if i == 0 else v1
                    if v in good:
                        probs[v_idx] *= good[v]
            probs /= probs.sum()
            return probs
        # Could be {0: [p0,p1,p2,p3], 1: [p0,p1,p2,p3]}
        if 0 in distribution_spec and 1 in distribution_spec:
            p0 = distribution_spec[0]
            p1 = distribution_spec[1]
            probs = np.ones(n)
            for v_idx, (v0, v1) in enumerate(reports):
                probs[v_idx] = p0[v0] * p1[v1]
            probs /= probs.sum()
            return probs
    
    if isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == n:
            p = np.array(distribution_spec, dtype=float)
            p /= p.sum()
            return p
        if len(distribution_spec) == 2:
            p0, p1 = distribution_spec
            probs = np.ones(n)
            for v_idx, (v0, v1) in enumerate(reports):
                probs[v_idx] = p0[v0] * p1[v1]
            probs /= probs.sum()
            return probs
    
    # Default: uniform
    return np.ones(n) / n


def _verify_ic_ir(reports, probs, allocations, payments, tol=1e-8):
    """Verify IC, IR, and feasibility."""
    n = len(reports)
    
    # Feasibility: 0 <= allocations <= 1
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    # IR: utility >= 0 for all types
    for v_idx, (v0, v1) in enumerate(reports):
        utility = v0 * allocations[v_idx, 0] + v1 * allocations[v_idx, 1] - payments[v_idx]
        if utility < -tol:
            return False
    
    # IC: for all v, v', utility(v, v') >= utility(v, v)
    # utility when true type is v but report is v': v0*q0[v'] + v1*q1[v'] - p[v']
    for v_idx, (v0, v1) in enumerate(reports):
        true_utility = v0 * allocations[v_idx, 0] + v1 * allocations[v_idx, 1] - payments[v_idx]
        for v2_idx, (v2_0, v2_1) in enumerate(reports):
            misreport_utility = v0 * allocations[v2_idx, 0] + v1 * allocations[v2_idx, 1] - payments[v2_idx]
            if misreport_utility > true_utility + tol:
                return False
    
    # Nonnegative payments
    if np.any(payments < -tol):
        return False
    
    return True


def _threshold_auction(reports, probs, start_time, deadline):
    """Fallback threshold-based auction."""
    values = [0, 1, 2, 3]
    report_to_idx = {r: i for i, r in enumerate(reports)}
    
    best_revenue = -1
    best_allocations = None
    best_payments = None
    
    # Try all threshold combinations
    for t0 in values:
        for t1 in values:
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
            for v_idx, (v0, v1) in enumerate(reports):
                allocations[v_idx, 0] = 1.0 if v0 >= t0 else 0.0
                allocations[v_idx, 1] = 1.0 if v1 >= t1 else 0.0
            
            # Set payments via payment identity with U(0,0) = 0
            # U(v) = sum over all w <= v of marginal contributions
            # For threshold mechanism, U(v) = sum_i max(0, v_i - t_i) if we use the right payment
            # Actually, for a threshold: payment = sum_i v_i * 1[v_i >= t_i] - U(v)
            # U(v) for threshold: the utility is sum_i max(0, v_i - t_i)
            # So payment = sum_i v_i * 1[v_i >= t_i] - sum_i max(0, v_i - t_i)
            # = sum_i [v_i - max(0, v_i - t_i)] * 1[v_i >= t_i] ... let me compute directly
            
            U = np.zeros(16)
            for v_idx, (v0, v1) in enumerate(reports):
                U[v_idx] = max(0, v0 - t0) + max(0, v1 - t1)
                payments[v_idx] = v0 * allocations[v_idx, 0] + v1 * allocations[v_idx, 1] - U[v_idx]
            
            # Check IR
            if np.any(payments < -1e-8):
                continue
            
            # Check IC
            if not _verify_ic_ir(reports, probs, allocations, payments):
                continue
            
            revenue = np.sum(probs * payments)
            if revenue > best_revenue:
                best_revenue = revenue
                best_allocations = allocations.copy()
                best_payments = payments.copy()
            
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if best_allocations is not None:
        return {
            'allocations': best_allocations,
            'payments': best_payments
        }
    
    # Final fallback: no trade
    return {
        'allocations': np.zeros((16, 2)),
        'payments': np.zeros(16)
    }

# EVOLVE-BLOCK-END
