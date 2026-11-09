"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse distribution spec to get types and probabilities
    # support: list of (v1, v2) values, each in {0,1,2,3}
    # distribution_spec: dict or list giving probabilities for each type
    
    # Build type list from support
    types = []
    for s in support:
        if isinstance(s, (tuple, list)):
            types.append(tuple(s))
        else:
            types.append(s)
    
    n_types = len(types)
    
    # Get probabilities for each type
    probs = np.zeros(n_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            probs[i] = distribution_spec.get(t, 0.0)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        for i, t in enumerate(types):
            probs[i] = float(distribution_spec[i])
    else:
        # Try other formats
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == n_types:
                pass
            else:
                probs = np.ones(n_types) / n_types
        except:
            probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 1e-12:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types
    
    # Value vectors for each type
    values = np.array([list(t) for t in types], dtype=float)
    
    # LP formulation:
    # Variables: x_1(t), x_2(t), p(t) for each type t -> 3 * n_types variables
    # Maximize: sum_t probs[t] * p(t)
    # Constraints:
    #   IC: for all s, t: v_s . x(s) - p(s) >= v_s . x(t) - p(t)
    #       => v_s . x(s) - v_s . x(t) - p(s) + p(t) >= 0
    #   IR: for all t: v_t . x(t) - p(t) >= 0
    #   0 <= x_j(t) <= 1
    #   p(t) >= 0
    
    n_vars = 3 * n_types  # x1_t, x2_t, p_t for each type
    
    # Objective: maximize sum_t probs[t] * p(t)
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[2 * t + 2] = -probs[t]  # p(t) coefficient
    
    # Build constraint matrix A_ub such that A_ub @ x <= b_ub
    A_rows = []
    b_rows = []
    
    # IC constraints: for all s, t: v_s . x(s) - v_s . x(t) - p(s) + p(t) >= 0
    # => -(v_s . x(s) - v_s . x(t) - p(s) + p(t)) <= 0
    # => -v_s[0]*x1(s) - v_s[1]*x2(s) + v_s[0]*x1(t) + v_s[1]*x2(t) + p(s) - p(t) <= 0
    for s in range(n_types):
        for t in range(n_types):
            if s == t:
                continue
            row = np.zeros(n_vars)
            # -v_s[0] * x1(s)
            row[2 * s + 0] = -values[s, 0]
            # -v_s[1] * x2(s)
            row[2 * s + 1] = -values[s, 1]
            # +v_s[0] * x1(t)
            row[2 * t + 0] = values[s, 0]
            # +v_s[1] * x2(t)
            row[2 * t + 1] = values[s, 1]
            # +p(s)
            row[2 * s + 2] = 1.0
            # -p(t)
            row[2 * t + 2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: v_t . x(t) - p(t) >= 0 => -(v_t . x(t) - p(t)) <= 0
    # => -v_t[0]*x1(t) - v_t[1]*x2(t) + p(t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[2 * t + 0] = -values[t, 0]
        row[2 * t + 1] = -values[t, 1]
        row[2 * t + 2] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.array([[]]).reshape(0, n_vars)
    b_ub = np.array(b_rows) if b_rows else np.array([])
    
    # Bounds: 0 <= x_j(t) <= 1, p(t) >= 0
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1(t)
        bounds.append((0, 1))  # x2(t)
        bounds.append((0, None))  # p(t)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: zero allocations, zero payments
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {
            'allocations': allocations,
            'payments': payments
        }
    
    x = result.x
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    for t in range(n_types):
        allocations[t, 0] = x[2 * t + 0]
        allocations[t, 1] = x[2 * t + 1]
        payments[t] = x[2 * t + 2]
    
    # Clamp small numerical errors
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify constraints with tight tolerance
    # Check IC
    for s in range(n_types):
        for t in range(n_types):
            util_s_s = values[s] @ allocations[s] - payments[s]
            util_s_t = values[s] @ allocations[t] - payments[t]
            if util_s_s < util_s_t - 1e-8:
                # Small violation, adjust payment
                # This shouldn't happen with correct LP solve
                pass
    
    # Check IR
    for t in range(n_types):
        util_t = values[t] @ allocations[t] - payments[t]
        if util_t < -1e-8:
            payments[t] = max(0, values[t] @ allocations[t] - 1e-10)
    
    # Re-clamp after adjustments
    payments = np.maximum(payments, 0)
    allocations = np.clip(allocations, 0, 1)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
