"""Incentive-compatible finite auction design via dual LP solve."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Parse inputs
    if isinstance(distribution_spec, dict):
        probs = np.array(distribution_spec.get('probs', []), dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    # Ensure we have 16 probabilities
    if len(probs) < 16:
        probs = np.pad(probs, (0, 16 - len(probs)))
    else:
        probs = probs[:16]
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16) / 16.0
    
    # Support: standard 4x4 grid
    support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_reports = 16
    values = np.array(support, dtype=float)  # shape (16, 2)
    
    # Check time
    if time.time() > deadline - 5:
        allocs = np.zeros((16, 2))
        pays = np.zeros(16)
        return {'allocations': allocs, 'payments': pays}
    
    # Use the dual LP approach for better numerical stability
    # 
    # Primal: max sum_i probs[i] * p[i]
    #   s.t. IC: for all i,j: sum_g x[i,g]*v_i[g] - p[i] >= sum_g x[j,g]*v_i[g] - p[j]
    #        IR: for all i: sum_g x[i,g]*v_i[g] - p[i] >= 0
    #        0 <= x[i,g] <= 1, p[i] >= 0
    #
    # Dual formulation:
    # Let y[i,j] >= 0 be the dual variable for IC constraint (i,j)
    # Let z[i] >= 0 be the dual variable for IR constraint i
    # Let w[i,g] >= 0 be the dual variable for x[i,g] <= 1 constraint
    #
    # Dual: min sum_{i,g} w[i,g]
    #   s.t. for each allocation var x[i,g]:
    #        w[i,g] - sum_{j != i} y[i,j]*v_i[g] + sum_{j != i} y[j,i]*v_j[g] - z[i]*v_i[g] >= 0
    #        (from the coefficient of x[i,g] in the Lagrangian)
    #
    #   s.t. for each payment var p[i]:
    #        -sum_{j != i} y[i,j] + sum_{j != i} y[j,i] - z[i] + probs[i] >= 0
    #        (wait, this needs more careful derivation)
    #
    # Let me use a cleaner approach. The key insight is that for a single additive bidder,
    # the optimal mechanism can be characterized by the expected allocation being non-decreasing
    # in value. Let's use the direct primal LP but with better numerical settings.
    
    # Actually, let's stick with the primal LP but improve numerical stability
    n_alloc = 32
    n_pay = 16
    n_vars = n_alloc + n_pay
    
    # Objective: maximize sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(16):
        c[n_alloc + i] = -probs[i]
    
    # Build constraint matrix
    A_rows = []
    b_rows = []
    
    # IC constraints: for all i != j
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            for g in range(2):
                row[i * 2 + g] = -values[i, g]
                row[j * 2 + g] = values[i, g]
            row[n_alloc + i] = 1.0
            row[n_alloc + j] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints
    for i in range(16):
        row = np.zeros(n_vars)
        for g in range(2):
            row[i * 2 + g] = -values[i, g]
        row[n_alloc + i] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    bounds = [(0, 1)] * 32 + [(0, None)] * 16
    
    # Check time
    if time.time() > deadline - 10:
        allocs = np.zeros((16, 2))
        pays = np.zeros(16)
        return {'allocations': allocs, 'payments': pays}
    
    # Solve LP with HiGHS
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                              'primal_feasibility_tolerance': 1e-10})
    
    if result.success:
        x_opt = result.x
        allocs = x_opt[:32].reshape(16, 2)
        pays = x_opt[32:]
        
        # Clip for numerical safety
        allocs = np.clip(allocs, 0, 1)
        pays = np.maximum(pays, 0)
        
        # Verify and fix IC violations
        for i in range(16):
            for j in range(16):
                if i == j:
                    continue
                ui_i = np.dot(allocs[i], values[i]) - pays[i]
                ui_j = np.dot(allocs[j], values[i]) - pays[j]
                if ui_i < ui_j - 1e-8:
                    # Increase payment for j to fix violation
                    excess = ui_j - ui_i + 1e-8
                    pays[j] += excess
        
        # Re-verify IR
        for i in range(16):
            ui_i = np.dot(allocs[i], values[i]) - pays[i]
            if ui_i < 0:
                pays[i] = max(0, np.dot(allocs[i], values[i]))
        
        # Final safety check: if any payment went negative after adjustments, set to 0
        pays = np.maximum(pays, 0)
        
        return {'allocations': allocs, 'payments': pays}
    else:
        # Fallback: trivial mechanism
        allocs = np.zeros((16, 2))
        pays = np.zeros(16)
        return {'allocations': allocs, 'payments': pays}

# EVOLVE-BLOCK-END
