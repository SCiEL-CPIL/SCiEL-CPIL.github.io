"""Incentive-compatible finite auction design for two goods, one bidder, values in {0,1,2,3}."""
import time
import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START
def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Parameters:
    -----------
    distribution_spec : str or dict - one of 'uniform', 'increasing', 'decreasing' or equivalent
    support : set or list - the value support, expected {0,1,2,3} for each good
    constraints : dict - constraint specifications (e.g., 'regret_bound': 1e-8)
    
    Returns:
    --------
    dict with keys:
        'allocations': (16, 2) array - allocation for each of 16 report types
        'payments': (16,) array - payment for each of 16 report types
    """
    t_start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Parse value support
    if support is None:
        vals = [0, 1, 2, 3]
    else:
        vals = sorted(set(int(v) for v in support))
    if len(vals) != 4:
        vals = [0, 1, 2, 3]

    n = len(vals)  # 4
    num_types = n * n  # 16

    # Map (v1, v2) to index
    def type_idx(v1, v2):
        return vals.index(v1) * n + vals.index(v2)

    # Parse distribution
    def get_prob_dist(spec):
        """Return probability vector over 16 types."""
        if spec is None:
            spec = 'uniform'
        
        if isinstance(spec, str):
            s = spec.lower().strip()
            if 'uniform' in s:
                return np.ones(num_types) / num_types
            elif 'increasing' in s:
                # Increasing: higher values more likely
                weights = []
                for v1 in vals:
                    for v2 in vals:
                        w = (v1 + 1) * (v2 + 1)
                        weights.append(w)
                weights = np.array(weights, dtype=float)
                return weights / weights.sum()
            elif 'decreasing' in s:
                # Decreasing: lower values more likely
                weights = []
                for v1 in vals:
                    for v2 in vals:
                        w = (4 - v1) * (4 - v2)
                        weights.append(w)
                weights = np.array(weights, dtype=float)
                return weights / weights.sum()
            else:
                return np.ones(num_types) / num_types
        elif isinstance(spec, (list, tuple, np.ndarray)):
            arr = np.array(spec, dtype=float).flatten()
            if len(arr) == num_types:
                return arr / arr.sum()
            return np.ones(num_types) / num_types
        elif isinstance(spec, dict):
            probs = np.zeros(num_types)
            for key, p in spec.items():
                if isinstance(key, (list, tuple)):
                    i = type_idx(key[0], key[1])
                    probs[i] = p
                elif isinstance(key, int):
                    probs[key] = p
            if probs.sum() > 0:
                return probs / probs.sum()
            return np.ones(num_types) / num_types
        else:
            return np.ones(num_types) / num_types

    dist = get_prob_dist(distribution_spec)

    # Build type lists
    types = []
    for v1 in vals:
        for v2 in vals:
            types.append((v1, v2))

    # Type indices
    t_idx = {t: i for i, t in enumerate(types)}

    # ============================================================
    # Formulate the mechanism design LP
    # ============================================================
    # Decision variables:
    #   x[i, j] : allocation of good j when reporting type i, for i in 0..15, j in 0,1
    #   p[i]    : payment when reporting type i, for i in 0..15
    #
    # Total variables: 16*2 + 16 = 48
    #
    # Objective: maximize expected revenue = sum_i dist[i] * p[i]
    #            (linprog minimizes, so we negate)
    #
    # Constraints:
    # 1. Feasibility: 0 <= x[i,j] <= 1 for all i,j
    # 2. Nonnegative payments: p[i] >= 0
    # 3. IR: For each type t, utility when reporting truthfully >= 0
    #    u_i(x_i, p_i) = v_i . x_i - p_i >= 0
    #    => p_i <= v_i . x_i
    # 4. IC (incentive compatibility): For each pair (i, j) where i is true type, j is report
    #    u_i(x_i, p_i) >= u_i(x_j, p_j)
    #    => v_i . x_i - p_i >= v_i . x_j - p_j
    #    => v_i . (x_i - x_j) >= p_i - p_j
    #    => p_j - p_i >= v_i . (x_j - x_i)
    #    i.e., -v_i . x_j + v_i . x_i + p_j - p_i >= 0  ... wait let me redo
    #    v_i . x_i - p_i >= v_i . x_j - p_j
    #    => v_i . x_i - v_i . x_j - p_i + p_j >= 0
    #    => -p_i + p_j + v_i . x_i - v_i . x_j >= 0

    # Variable ordering: [x[0,0], x[0,1], x[1,0], x[1,1], ..., x[15,0], x[15,1], p[0], p[1], ..., p[15]]
    # x var index for type i, good j: i * 2 + j
    # p var index for type i: 16 * 2 + i = 32 + i

    def xvar(i, j):
        return i * 2 + j

    def pvar(i):
        return 32 + i

    num_vars = 48

    # Objective: maximize sum dist[i]*p[i] => minimize -sum dist[i]*p[i]
    c = np.zeros(num_vars)
    for i in range(num_types):
        c[pvar(i)] = -dist[i]

    # Inequality constraints: A_ub @ x_vec <= b_ub
    A_rows = []
    b_rows = []

    # IR constraints: p_i <= v_i . x_i  =>  -v_i . x_i + p_i <= 0
    for i in range(num_types):
        v1, v2 = types[i]
        row = np.zeros(num_vars)
        row[xvar(i, 0)] = -v1
        row[xvar(i, 1)] = -v2
        row[pvar(i)] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # IC constraints: for each true type i and report j
    # v_i . x_i - v_i . x_j - p_i + p_j >= 0
    # => -v_i . x_i + v_i . x_j + p_i - p_j <= 0
    for i in range(num_types):
        v1, v2 = types[i]
        for j in range(num_types):
            if i == j:
                continue
            row = np.zeros(num_vars)
            row[xvar(i, 0)] = -v1
            row[xvar(i, 1)] = -v2
            row[xvar(j, 0)] = v1
            row[xvar(j, 1)] = v2
            row[pvar(i)] = 1.0
            row[pvar(j)] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)

    # Upper bound constraints: x[i,j] <= 1 (bounds handle this)
    # Lower bound: x[i,j] >= 0, p[i] >= 0

    A_ub = np.array(A_rows) if A_rows else np.zeros((0, num_vars))
    b_ub = np.array(b_rows) if b_rows else np.array([])

    bounds = []
    for i in range(num_types):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(num_types):
        bounds.append((0.0, None))

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        # Fallback: uniform allocation, zero payments (trivially IC and IR)
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    x_vec = result.x

    # Extract allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)

    for i in range(num_types):
        allocations[i, 0] = max(0.0, x_vec[xvar(i, 0)])
        allocations[i, 1] = max(0.0, x_vec[xvar(i, 1)])
        payments[i] = max(0.0, x_vec[pvar(i)])

    # Clip allocations to [0, 1]
    allocations = np.clip(allocations, 0.0, 1.0)

    # ============================================================
    # Post-processing: verify and fix any numerical issues
    # ============================================================
    regret_bound = 1e-8
    if isinstance(constraints, dict):
        if 'regret_bound' in constraints:
            regret_bound = float(constraints['regret_bound'])

    # Verify IC and IR; if violated beyond tolerance, project back
    # Check IR
    for i in range(num_types):
        v1, v2 = types[i]
        util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        if util < -1e-10:
            # Reduce payment
            payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1]
            if payments[i] < 0:
                payments[i] = 0.0

    # Check IC
    for i in range(num_types):
        v1, v2 = types[i]
        truth_util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        for j in range(num_types):
            if i == j:
                continue
            lie_util = v1 * allocations[j, 0] + v2 * allocations[j, 1] - payments[j]
            if lie_util - truth_util > regret_bound:
                # Increase payment for report j or decrease allocation for j
                # Simplest: increase p_j to make lie unprofitable
                needed = lie_util - truth_util + 1e-12
                payments[j] += needed

    # Re-verify IR after adjustments
    for i in range(num_types):
        v1, v2 = types[i]
        util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        if util < 0:
            payments[i] = max(0.0, v1 * allocations[i, 0] + v2 * allocations[i, 1] - 1e-12)

    # Final IC check with a second pass
    for _pass in range(5):
        max_violation = 0.0
        for i in range(num_types):
            v1, v2 = types[i]
            truth_util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
            for j in range(num_types):
                if i == j:
                    continue
                lie_util = v1 * allocations[j, 0] + v2 * allocations[j, 1] - payments[j]
                viol = lie_util - truth_util
                if viol > regret_bound:
                    max_violation = max(max_violation, viol)
                    payments[j] += viol + 1e-12
        if max_violation <= regret_bound:
            break

    # Ensure payments nonnegative
    payments = np.maximum(payments, 0.0)

    # Final IR check
    for i in range(num_types):
        v1, v2 = types[i]
        util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        if util < 0:
            payments[i] = max(0.0, v1 * allocations[i, 0] + v2 * allocations[i, 1])

    # Check deadline
    if time.time() - t_start > DEADLINE:
        pass  # return what we have

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
