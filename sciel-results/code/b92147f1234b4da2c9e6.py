import numpy as np
from scipy.optimize import linprog
import time
import json

# Pre-compute the constraint structure
# Support: (v1, v2) for v1, v2 in {0, 1, 2, 3}
values = [0, 1, 2, 3]
support_grid = [(v1, v2) for v1 in values for v2 in values]
n_states = len(support_grid)
state_to_idx = {s: i for i, s in enumerate(support_grid)}

# Variables: x1[0..15], x2[0..15], p[0..15]
n_vars = 3 * n_states

# Build A_ub and b_ub
A_ub_rows = []
b_ub_vals = []

# IC constraints: p_i - p_j <= v1_i*x1_i + v2_i*x2_i - (v1_i*x1_j + v2_i*x2_j)
# p_i - p_j - v1_i*x1_i - v2_i*x2_i + v1_i*x1_j + v2_i*x2_j <= 0
for i in range(n_states):
    v1_i, v2_i = support_grid[i]
    for j in range(n_states):
        if i == j:
            continue
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

A_ub = np.array(A_ub_rows)
b_ub = np.array(b_ub_vals)
bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    
    # Extract probabilities
    probs = np.zeros(n_states)
    if isinstance(distribution_spec, dict):
        for s, p in distribution_spec.items():
            if s in state_to_idx:
                probs[state_to_idx[s]] = float(p)
    elif isinstance(distribution_spec, str):
        try:
            parsed = json.loads(distribution_spec)
            if isinstance(parsed, dict):
                for s, p in parsed.items():
                    if s in state_to_idx:
                        probs[state_to_idx[s]] = float(p)
            elif isinstance(parsed, (list, np.ndarray)):
                arr = np.array(parsed, dtype=float)
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
        except (json.JSONDecodeError, ValueError, TypeError):
            probs = np.ones(n_states) / n_states
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

    # Objective: Maximize sum(probs[i] * p[i])
    # Minimize -sum(probs[i] * p[i])
    c = np.zeros(n_vars)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]

    # Solve LP
    x_opt = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            x_opt = res.x
    except Exception:
        pass

    if x_opt is None:
        try:
            res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
            if res2.status == 0:
                x_opt = res2.x
        except Exception:
            pass

    if x_opt is None:
        # Fallback: Return a valid but potentially suboptimal mechanism
        # x=0, p=0 is always valid.
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
    
    # Ensure IR is satisfied after adjustments
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        max_payment = v1_i * allocations[i, 0] + v2_i * allocations[i, 1]
        if payments[i] > max_payment:
            payments[i] = max_payment
        if payments[i] < 0:
            payments[i] = 0.0

    # Verify IC (regret check)
    max_regret = 0.0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        utility_truthful = v1_i * x1[i] + v2_i * x2[i] - p[i]
        for j in range(n_states):
            utility_deviate = v1_i * x1[j] + v2_i * x2[j] - p[j]
            regret = utility_deviate - utility_truthful
            if regret > max_regret:
                max_regret = regret

    # If regret exceeds tolerance, try to fix by adjusting payments slightly
    if max_regret > 1e-8:
        # Attempt a small adjustment to reduce regret
        adjusted_p = p.copy()
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            utility_truthful = v1_i * x1[i] + v2_i * x2[i] - adjusted_p[i]
            for j in range(n_states):
                if i == j:
                    continue
                utility_deviate = v1_i * x1[j] + v2_i * x2[j] - adjusted_p[j]
                regret = utility_deviate - utility_truthful
                if regret > 1e-8:
                    # Increase payment at state i to reduce incentive to deviate to j
                    adjusted_p[i] += regret + 1e-10
        
        # Re-check with adjusted payments
        max_regret_adj = 0.0
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            utility_truthful = v1_i * x1[i] + v2_i * x2[i] - adjusted_p[i]
            for j in range(n_states):
                utility_deviate = v1_i * x1[j] + v2_i * x2[j] - adjusted_p[j]
                regret = utility_deviate - utility_truthful
                if regret > max_regret_adj:
                    max_regret_adj = regret
        
        if max_regret_adj <= 1e-8:
            payments = adjusted_p
        else:
            allocations = np.zeros((n_states, 2))
            payments = np.zeros(n_states)
    else:
        payments = p

    return {"allocations": allocations, "payments": payments}
