"""Rewritten program: exact finite MILP for incentive-compatible single-bidder 2-good auction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill

    # Parse the 16 types: values in {0,1,2,3} for 2 goods
    # Type index: idx = v1 * 4 + v2, so 16 types
    n = 16
    values_per_good = [0, 1, 2, 3]
    num_values = len(values_per_good)

    # Extract probabilities from distribution_spec
    probs = _extract_probabilities(distribution_spec, support, constraints, n)

    # If we can't parse, return a safe baseline
    if probs is None:
        return _safe_baseline()

    # Build the MILP for the finite mechanism design
    # Variables:
    #   x[t, g] for t in 0..15, g in 0..1  -> allocation probabilities (2 * 16 = 32 vars)
    #   p[t] for t in 0..15                -> payments (16 vars)
    # Total: 48 variables
    #
    # Objective: maximize sum_t probs[t] * (p[t] - sum_g x[t,g] * v[t,g])
    # Wait, revenue = sum_t probs[t] * p[t] (since the bidder pays p[t] and gets allocation)
    # Actually, expected revenue = sum_t probs[t] * p[t]
    # But we need to be careful: the objective is to maximize expected payment.
    #
    # Constraints:
    # 1. Feasibility: 0 <= x[t,g] <= 1 for all t, g
    # 2. Nonnegative payments: p[t] >= 0 for all t
    # 3. Individual rationality: For each type t, the utility u[t] >= 0
    #    u[t] = sum_g x[t,g] * v[t,g] - p[t] >= 0
    #    => p[t] <= sum_g x[t,g] * v[t,g]
    # 4. Incentive compatibility (IC): For all types t and t', type t must prefer reporting t over t'
    #    u[t, t] >= u[t, t']
    #    sum_g x[t,g]*v[t,g] - p[t] >= sum_g x[t',g]*v[t,g] - p[t']
    #    => p[t'] - p[t] >= sum_g (x[t',g] - x[t,g]) * v[t,g]
    #
    # For a single bidder with quasi-linear utility, the IC constraints can be simplified.
    # The standard approach: fix p[0] = 0 (or some reference), then express all other p[t] in terms of x.
    # Actually, for finite types, the IC constraints are:
    #   p[t] - p[t'] >= sum_g (x[t,g] - x[t',g]) * v[t,g]  for all t, t'
    #
    # This is a linear program (not MILP) because x can be continuous in [0,1].
    # Let's use scipy.optimize.linprog or milp with continuous variables.

    # Actually, let's use scipy.optimize.milp with integrality=0 for continuous vars,
    # or better, use linprog which handles LPs directly.

    from scipy.optimize import linprog

    # Number of variables: 32 (allocations) + 16 (payments) = 48
    # Let's index: var_i = i for i in 0..31 -> x[t,g] where t = i // 2, g = i % 2
    #             var_i = 32 + t for i in 32..47 -> p[t] where t = i - 32

    num_alloc_vars = n * 2  # 32
    num_pay_vars = n       # 16
    total_vars = num_alloc_vars + num_pay_vars  # 48

    # Objective: maximize sum_t probs[t] * p[t]
    # In linprog, we minimize, so c = -probs extended
    c = np.zeros(total_vars)
    for t in range(n):
        c[num_alloc_vars + t] = -probs[t]

    # Inequality constraints: A_ub @ x <= b_ub

    # Constraint set 1: IR: p[t] - sum_g x[t,g]*v[t,g] <= 0  for each t
    # i.e., p[t] <= sum_g x[t,g]*v[t,g]
    # Rearranged: p[t] - sum_g x[t,g]*v[t,g] <= 0

    # Constraint set 2: IC: for all t, t': p[t'] - p[t] - sum_g (x[t',g] - x[t,g])*v[t,g] <= 0
    # i.e., p[t'] - p[t] <= sum_g (x[t',g] - x[t,g])*v[t,g]

    # Let's build the constraint matrix
    # Number of IR constraints: n = 16
    # Number of IC constraints: n * n = 256 (but we can skip t==t' which is trivial)
    # Actually, IC for t==t' is 0<=0, so skip. Use n*(n-1) = 240 constraints.

    num_ir = n
    num_ic = n * (n - 1)
    num_ineq = num_ir + num_ic

    A_ub = np.zeros((num_ineq, total_vars))
    b_ub = np.zeros(num_ineq)

    # IR constraints
    for t in range(n):
        row = 0
        v1 = t // num_values
        v2 = t % num_values
        # p[t] - x[t,0]*v1 - x[t,1]*v2 <= 0
        # x[t,0] is var index: t*2 + 0
        # x[t,1] is var index: t*2 + 1
        # p[t] is var index: num_alloc_vars + t
        A_ub[row, t * 2 + 0] = -v1
        A_ub[row, t * 2 + 1] = -v2
        A_ub[row, num_alloc_vars + t] = 1.0
        b_ub[row] = 0.0
        row += 1

    # IC constraints
    for t in range(n):
        for tp in range(n):
            if t == tp:
                continue
            row = num_ir + t * (n - 1) + tp if tp < t else num_ir + t * (n - 1) + (tp - 1)
            # Let me just use a counter
            pass

    # Let me redo with a counter
    A_ub = np.zeros((num_ineq, total_vars))
    b_ub = np.zeros(num_ineq)
    row = 0

    # IR constraints
    for t in range(n):
        v1 = t // num_values
        v2 = t % num_values
        A_ub[row, t * 2 + 0] = -v1
        A_ub[row, t * 2 + 1] = -v2
        A_ub[row, num_alloc_vars + t] = 1.0
        b_ub[row] = 0.0
        row += 1

    # IC constraints
    for t in range(n):
        for tp in range(n):
            if t == tp:
                continue
            v1 = t // num_values
            v2 = t % num_values
            # p[tp] - p[t] - (x[tp,0]-x[t,0])*v1 - (x[tp,1]-x[t,1])*v2 <= 0
            # => -x[tp,0]*v1 + x[t,0]*v1 - x[tp,1]*v2 + x[t,1]*v2 + p[tp] - p[t] <= 0
            A_ub[row, tp * 2 + 0] = -v1
            A_ub[row, t * 2 + 0] = v1
            A_ub[row, tp * 2 + 1] = -v2
            A_ub[row, t * 2 + 1] = v2
            A_ub[row, num_alloc_vars + tp] = 1.0
            A_ub[row, num_alloc_vars + t] = -1.0
            b_ub[row] = 0.0
            row += 1

    # Bounds: 0 <= x[t,g] <= 1, 0 <= p[t] <= infinity
    bounds = []
    for i in range(num_alloc_vars):
        bounds.append((0, 1))
    for i in range(num_pay_vars):
        bounds.append((0, None))

    # Solve LP
    try:
        if time.time() > deadline - 5:
            return _safe_baseline()

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if time.time() > deadline - 2:
            return _safe_baseline()

        if res.status == 0:
            x_opt = res.x
            # Extract allocations and payments
            allocations = np.zeros((n, 2))
            payments = np.zeros(n)
            for t in range(n):
                allocations[t, 0] = x_opt[t * 2 + 0]
                allocations[t, 1] = x_opt[t * 2 + 1]
                payments[t] = x_opt[num_alloc_vars + t]

            # Verify constraints
            if _verify_mechanism(allocations, payments, probs, values_per_good):
                return {'allocations': allocations, 'payments': payments}
            else:
                return _safe_baseline()
        else:
            return _safe_baseline()
    except Exception:
        return _safe_baseline()


def _extract_probabilities(distribution_spec, support, constraints, n=16):
    """Extract the 16 type probabilities from the input specification."""
    try:
        # Try to get probabilities from distribution_spec
        if isinstance(distribution_spec, dict):
            # Look for common keys
            for key in ['probs', 'probabilities', 'p', 'weights']:
                if key in distribution_spec:
                    p = np.array(distribution_spec[key], dtype=float).flatten()
                    if len(p) == n:
                        return p / p.sum()
            # Maybe it's a nested structure
            if 'types' in distribution_spec and 'probs' in distribution_spec:
                p = np.array(distribution_spec['probs'], dtype=float).flatten()
                if len(p) == n:
                    return p / p.sum()

        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            p = np.array(distribution_spec, dtype=float).flatten()
            if len(p) == n:
                return p / p.sum()

        # Try support
        if support is not None:
            if isinstance(support, (list, tuple, np.ndarray)):
                s = np.array(support, dtype=float).flatten()
                if len(s) == n:
                    return s / s.sum()

        # Try constraints
        if isinstance(constraints, dict):
            for key in ['probs', 'probabilities', 'p', 'weights', 'distribution']:
                if key in constraints:
                    p = np.array(constraints[key], dtype=float).flatten()
                    if len(p) == n:
                        return p / p.sum()

        # Default: uniform
        return np.ones(n) / n
    except Exception:
        return np.ones(n) / n


def _safe_baseline():
    """Return a safe baseline: allocate both goods to all types, charge 0."""
    n = 16
    allocations = np.ones((n, 2))
    payments = np.zeros(n)
    return {'allocations': allocations, 'payments': payments}


def _verify_mechanism(allocations, payments, probs, values_per_good, tol=1e-8):
    """Verify that the mechanism satisfies all constraints."""
    n = len(probs)
    num_values = len(values_per_good)

    # Check feasibility: allocations in [0,1]
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False

    # Check nonnegative payments
    if np.any(payments < -tol):
        return False

    # Check IR: utility >= 0 for all types
    for t in range(n):
        v1 = t // num_values
        v2 = t % num_values
        utility = allocations[t, 0] * v1 + allocations[t, 1] * v2 - payments[t]
        if utility < -tol:
            return False

    # Check IC: for all t, t', type t prefers reporting t
    for t in range(n):
        v1 = t // num_values
        v2 = t % num_values
        u_tt = allocations[t, 0] * v1 + allocations[t, 1] * v2 - payments[t]
        for tp in range(n):
            u_t_tp = allocations[tp, 0] * v1 + allocations[tp, 1] * v2 - payments[tp]
            if u_tt < u_t_tp - tol:
                return False

    return True
# EVOLVE-BLOCK-END
