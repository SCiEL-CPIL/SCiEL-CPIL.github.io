"""Auction design via LP with envelope-theorem IC constraints."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2

    # Parse distribution probabilities
    probs = np.ones(n_types, dtype=float) / n_types
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, (list, tuple, np.ndarray)):
                probs = np.array(dist, dtype=float)
            elif isinstance(dist, dict):
                probs = np.zeros(n_types, dtype=float)
                for i, t in enumerate(types):
                    probs[i] = dist.get(t, 0.0)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)

    # Normalize probabilities
    probs = probs / probs.sum()

    # Parse support: which types are in the report/value support
    # Default: all 16 types
    in_support = np.ones(n_types, dtype=bool)
    if support is not None:
        if isinstance(support, (list, tuple, np.ndarray)):
            support_list = list(support)
            in_support = np.zeros(n_types, dtype=bool)
            for s in support_list:
                if isinstance(s, (tuple, list)) and len(s) == 2:
                    try:
                        idx = types.index(tuple(s))
                        in_support[idx] = True
                    except ValueError:
                        pass
                elif isinstance(s, int):
                    if 0 <= s < n_types:
                        in_support[s] = True

    # Variables:
    # x[i, g] for i in 0..15, g in 0..1  -> 32 allocation probabilities
    # u[i] for i in 0..15                 -> 16 utilities
    # Total: 48 variables
    #
    # Objective: maximize expected revenue = sum_i probs[i] * (v_i . x_i - u_i)
    #   where v_i is the value vector for type i
    #
    # Constraints:
    # 1. 0 <= x[i, g] <= 1 for all i, g
    # 2. u[i] >= 0 for all i (individual rationality: expected utility >= 0)
    #    Actually, IR means the utility from reporting truthfully is >= 0.
    #    With envelope theorem, u[i] is the utility of type i when reporting truthfully.
    # 3. IC constraints: u[j] <= u[i] + v_j . x_i - v_i . x_i for all type pairs (i, j)
    #    This is the discrete envelope condition: the utility of type j from
    #    reporting as type i should be <= the utility of type i from reporting truthfully.
    #    Wait, let me re-derive:
    #    Utility of type j when reporting type i: v_j . x_i - p_i
    #    Utility of type i when reporting type i: v_i . x_i - p_i = u_i
    #    So p_i = v_i . x_i - u_i
    #    Utility of type j reporting type i: v_j . x_i - (v_i . x_i - u_i) = u_i + v_j . x_i - v_i . x_i
    #    IC: u_j >= u_i + v_j . x_i - v_i . x_i for all i, j
    #    i.e., u_j - u_i >= v_j . x_i - v_i . x_i
    #    Or equivalently: u_i + v_j . x_i - v_i . x_i <= u_j
    #    Which is: u_j - u_i - (v_j - v_i) . x_i >= 0
    #    Let me write it as: u_j >= u_i + (v_j - v_i) . x_i
    #    Rearranged: u_i - u_j + (v_j - v_i) . x_i <= 0
    #    Or: -u_i + u_j - (v_j - v_i) . x_i >= 0
    #    For linprog (minimization), we need <= constraints.
    #    Let's use: u_i + (v_j - v_i) . x_i - u_j <= 0
    #
    # Revenue: sum_i probs[i] * (v_i . x_i - u_i)
    # = sum_i probs[i] * v_i . x_i - sum_i probs[i] * u_i

    # Build LP
    # Variables: [x_0_0, x_0_1, x_1_0, x_1_1, ..., x_15_0, x_15_1, u_0, u_1, ..., u_15]
    # Index: x[i, g] at index 2*i + g, u[i] at index 32 + i

    def x_idx(i, g):
        return 2 * i + g

    def u_idx(i):
        return 32 + i

    # Objective: minimize -expected_revenue = -sum_i probs[i] * (v_i . x_i - u_i)
    c = np.zeros(48)
    for i in range(n_types):
        v1, v2 = types[i]
        c[x_idx(i, 0)] -= probs[i] * v1
        c[x_idx(i, 1)] -= probs[i] * v2
        c[u_idx(i)] += probs[i]

    # Inequality constraints: A_ub @ vars <= b_ub
    # 1. IC constraints for all pairs (i, j): u_i + (v_j - v_i) . x_i - u_j <= 0
    #    Only for types in support
    A_ub_rows = []
    b_ub_vals = []

    support_indices = [i for i in range(n_types) if in_support[i]]

    for i in support_indices:
        vi1, vi2 = types[i]
        for j in support_indices:
            vj1, vj2 = types[j]
            if i == j:
                continue
            row = np.zeros(48)
            # u_i + (v_j - v_i) . x_i - u_j <= 0
            row[u_idx(i)] = 1.0
            row[x_idx(i, 0)] = vj1 - vi1
            row[x_idx(i, 1)] = vj2 - vi2
            row[u_idx(j)] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # 2. Individual rationality: u_i >= 0 => -u_i <= 0
    for i in support_indices:
        row = np.zeros(48)
        row[u_idx(i)] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # 3. Allocation bounds: 0 <= x[i, g] <= 1
    #    x[i, g] <= 1 => x[i, g] <= 1
    #    -x[i, g] <= 0
    for i in support_indices:
        for g in range(n_goods):
            row = np.zeros(48)
            row[x_idx(i, g)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)

            row2 = np.zeros(48)
            row2[x_idx(i, g)] = -1.0
            A_ub_rows.append(row2)
            b_ub_vals.append(0.0)

    # For types not in support, fix x[i,g] = 0 and u[i] = 0 (or handle via bounds)
    # Actually, for types not in support, we can just set their allocations to 0
    # and not include them in IC constraints. But we should still bound them.
    for i in range(n_types):
        if not in_support[i]:
            for g in range(n_goods):
                row = np.zeros(48)
                row[x_idx(i, g)] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(0.0)

                row2 = np.zeros(48)
                row2[x_idx(i, g)] = -1.0
                A_ub_rows.append(row2)
                b_ub_vals.append(0.0)
            row = np.zeros(48)
            row[u_idx(i)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            row2 = np.zeros(48)
            row2[u_idx(i)] = -1.0
            A_ub_rows.append(row2)
            b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((1, 48))
    b_ub = np.array(b_ub_vals) if b_ub_vals else np.zeros(1)

    # Bounds
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            if in_support[i]:
                bounds.append((0.0, 1.0))
            else:
                bounds.append((0.0, 0.0))
        if in_support[i]:
            bounds.append((0.0, None))
        else:
            bounds.append((0.0, 0.0))

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback: return zero allocations
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    x_opt = result.x

    # Extract allocations and compute payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for i in range(n_types):
        allocations[i, 0] = x_opt[x_idx(i, 0)]
        allocations[i, 1] = x_opt[x_idx(i, 1)]
        v1, v2 = types[i]
        u_i = x_opt[u_idx(i)]
        payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - u_i

    # Clip to ensure numerical validity
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # Check deadline
    if time.time() > deadline - 2.0:
        return {'allocations': allocations, 'payments': payments}

    # Verify IC and IR, and check regret
    # Regret: for each type j, the max utility from misreporting i minus true utility u_j
    # Should be <= 1e-8
    # We trust the LP solution satisfies IC, so regret should be ~0

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
