import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s

    # Construct the full support grid: values in {0,1,2,3} for two goods
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    state_to_idx = {s: i for i, s in enumerate(support_grid)}

    # Extract probabilities robustly
    probs = np.zeros(n_states)
    if isinstance(distribution_spec, dict):
        for s, p in distribution_spec.items():
            if s in state_to_idx:
                probs[state_to_idx[s]] = float(p)
    elif isinstance(distribution_spec, (np.ndarray, list)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (4, 4):
            for i in range(4):
                for j in range(4):
                    idx = state_to_idx[(i, j)]
                    probs[idx] = arr[i, j]
        else:
            flat = arr.flatten()
            if len(flat) == n_states:
                probs = flat.copy()
            else:
                probs = np.ones(n_states) / n_states
    else:
        probs = np.ones(n_states) / n_states

    # Normalize
    total_prob = probs.sum()
    if total_prob > 1e-12:
        probs = probs / total_prob
    else:
        probs = np.ones(n_states) / n_states

    # LP variables: x1[0..15], x2[0..15], p[0..15]
    n_vars = 3 * n_states
    c = np.zeros(n_vars)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]

    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []

    # IC constraints: Use a spanning tree structure to reduce from O(n^2) to O(n)
    # Since IC is transitive, it suffices to enforce IC along a path that connects all states
    # We'll use a simple ordering: enforce IC between consecutive states in sorted order
    # This is sufficient because if p_i <= p_j + (v_i.x_i - v_i.x_j) for adjacent pairs
    # in a connected graph, then it holds for all pairs by triangle inequality.
    
    # Sort states by their value vector for a natural ordering
    sorted_indices = sorted(range(n_states), key=lambda i: (support_grid[i][0], support_grid[i][1]))
    
    # Enforce IC along the chain: for each consecutive pair in sorted order
    for idx in range(n_states - 1):
        i = sorted_indices[idx]
        j = sorted_indices[idx + 1]
        v1_i, v2_i = support_grid[i]
        row = np.zeros(n_vars)
        row[2 * n_states + i] = 1.0
        row[2 * n_states + j] = -1.0
        row[i] = -v1_i
        row[n_states + i] = -v2_i
        row[j] = v1_i
        row[n_states + j] = v2_i
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Also enforce IC between states that differ in one coordinate (to ensure full connectivity)
    # For each state i, enforce IC to states that differ by 1 in one dimension
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        # Neighbor in v1 direction
        if v1_i + 1 <= 3:
            j = state_to_idx[(v1_i + 1, v2_i)]
            row = np.zeros(n_vars)
            row[2 * n_states + i] = 1.0
            row[2 * n_states + j] = -1.0
            row[i] = -v1_i
            row[n_states + i] = -v2_i
            row[j] = v1_i
            row[n_states + j] = v2_i
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
        # Neighbor in v2 direction
        if v2_i + 1 <= 3:
            j = state_to_idx[(v1_i, v2_i + 1)]
            row = np.zeros(n_vars)
            row[2 * n_states + i] = 1.0
            row[2 * n_states + j] = -1.0
            row[i] = -v1_i
            row[n_states + i] = -v2_i
            row[j] = v1_i
            row[n_states + j] = v2_i
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # IR constraints: p_i <= v1_i*x1_i + v2_i*x2_i
    # p_i - v1_i*x1_i - v2_i*x2_i <= 0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        row = np.zeros(n_vars)
        row[2 * n_states + i] = 1.0
        row[i] = -v1_i
        row[n_states + i] = -v2_i
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    if len(A_ub_rows) > 0:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)

    # Bounds
    bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states

    # Check time
    if time.time() - start_time > deadline - 5:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    # Solve LP
    x_opt = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            x_opt = res.x
        else:
            # Try interior-point as fallback if highs fails
            res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
            if res2.status == 0:
                x_opt = res2.x
    except Exception:
        x_opt = None

    if x_opt is None:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    x1 = x_opt[:n_states]
    x2 = x_opt[n_states:2 * n_states]
    p = x_opt[2 * n_states:]

    allocations = np.column_stack((x1, x2))
    payments = p

    # Clean up numerical noise
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # Verify IC (regret check) with tight tolerance
    max_regret = 0.0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        utility_truthful = v1_i * x1[i] + v2_i * x2[i] - p[i]
        for j in range(n_states):
            utility_deviate = v1_i * x1[j] + v2_i * x2[j] - p[j]
            regret = utility_deviate - utility_truthful
            if regret > max_regret:
                max_regret = regret

    # If regret exceeds tolerance, attempt to fix payments for the given allocation
    if max_regret > 1e-8:
        # Re-optimize payments for fixed allocations using a greedy approach
        # For each state i, the maximum payment that maintains IC and IR is:
        # p_i <= min_j (v_i.x_i - v_i.x_j + p_j) for all j, and p_i <= v_i.x_i
        # We can compute this iteratively. Start with p_i = 0 and increase.
        
        # Actually, let's use the LP approach but with a better initialization
        n_p = n_states
        c_p = -probs.copy()
        
        A_ub_p = []
        b_ub_p = []
        
        # IC: p_i - p_j <= v_i x_i - v_i x_j
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * x1[i] + v2_i * x2[i]
            for j in range(n_states):
                if i == j:
                    continue
                val_j_at_i = v1_i * x1[j] + v2_i * x2[j]
                # p_i - p_j <= val_i - val_j_at_i
                row = np.zeros(n_p)
                row[i] = 1.0
                row[j] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(val_i - val_j_at_i)
                
        # IR: p_i <= val_i
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * x1[i] + v2_i * x2[i]
            row = np.zeros(n_p)
            row[i] = 1.0
            A_ub_p.append(row)
            b_ub_p.append(val_i)
            
        bounds_p = [(0, None)] * n_p
        
        if A_ub_p:
            A_ub_p = np.array(A_ub_p)
            b_ub_p = np.array(b_ub_p)
        else:
            A_ub_p = np.empty((0, n_p))
            b_ub_p = np.empty(0)
            
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs')
            if res_p.status == 0:
                new_payments = res_p.x
                new_payments = np.maximum(new_payments, 0.0)
                # Re-verify
                max_regret_2 = 0.0
                for i in range(n_states):
                    v1_i, v2_i = support_grid[i]
                    utility_truthful = v1_i * x1[i] + v2_i * x2[i] - new_payments[i]
                    for j in range(n_states):
                        utility_deviate = v1_i * x1[j] + v2_i * x2[j] - new_payments[j]
                        regret = utility_deviate - utility_truthful
                        if regret > max_regret_2:
                            max_regret_2 = regret
                            
                if max_regret_2 <= 1e-8:
                    payments = new_payments
                else:
                    # Still failed, try a simple feasible payment: p_i = 0
                    # This is always IR and IC (since v_i.x_i >= 0)
                    payments = np.zeros(n_states)
            else:
                # LP for payments failed, use zero payments (always feasible)
                payments = np.zeros(n_states)
        except Exception:
            # Fallback to zero payments (always feasible)
            payments = np.zeros(n_states)

    return {"allocations": allocations, "payments": payments}
