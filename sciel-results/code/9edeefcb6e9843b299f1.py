"""Direct convex LP over allocation and payment variables for optimal finite-support mechanism."""
import numpy as np
from scipy.optimize import linprog
import time


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    # Value matrix: val[t] = (v1, v2)
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    # Direct LP formulation:
    # Variables: p_t (16), x1_t (16), x2_t (16) -> 48 total
    # Maximize: sum_t probs[t] * p_t
    # Subject to:
    #   IC: p_t - p_t' <= v_t . x_t - v_t . x_t'  for all t, t' (use only t < t2 to halve constraints)
    #   IR: p_t <= v_t . x_t  for all t
    #   0 <= p_t
    #   0 <= x1_t, x2_t <= 1
    
    n_vars = 48
    c = np.zeros(n_vars)
    # Maximize revenue -> minimize -revenue
    c[:16] = -probs
    
    A_rows = []
    b_rows = []
    
    # IC constraints: p_t - p_t' - v_t1*x_t1 + v_t1*x_t1' - v_t2*x_t2 + v_t2*x_t2' <= 0
    # Only need t < t2 since the constraint for (t2, t) is the negative of (t, t2)
    # But wait: IC requires BOTH directions. Let me reconsider.
    # Actually, IC says: for all t, t', p_t - p_t' <= v_t . x_t - v_t . x_t'
    # This is NOT symmetric. We need all pairs. But we can note that
    # p_t - p_t' <= v_t.x_t - v_t.x_t' is equivalent to
    # p_t' - p_t <= v_t'.x_t' - v_t'.x_t
    # These are DIFFERENT constraints (different LHS and RHS). So we need all 16*15=240.
    # However, we can reduce: the constraint p_t - p_t' <= v_t.x_t - v_t.x_t' 
    # combined with p_t' - p_t <= v_t'.x_t' - v_t'.x_t 
    # means |p_t - p_t'| <= |v_t.x_t - v_t.x_t'| only when values align.
    # Actually no, they're independent. Keep all 240.
    
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = 1.0
            row[t2] = -1.0
            row[16 + t] = -val_matrix[t, 0]
            row[16 + t2] = val_matrix[t, 0]
            row[32 + t] = -val_matrix[t, 1]
            row[32 + t2] = val_matrix[t, 1]
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: p_t - v_t1*x_t1 - v_t2*x_t2 <= 0
    for t in range(n):
        row = np.zeros(n_vars)
        row[t] = 1.0
        row[16 + t] = -val_matrix[t, 0]
        row[32 + t] = -val_matrix[t, 1]
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    # Bounds: p_t >= 0, x in [0,1]
    bounds = [(0, None)] * 16
    bounds += [(0, 1)] * 32
    
    # Check time before LP
    if time.time() > deadline - 5.0:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    try:
        options = {'presolve': True, 'primal_feasibility_tolerance': 1e-10,
                   'dual_feasibility_tolerance': 1e-10, 'time_limit': 75.0}
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        if result.success:
            p = result.x[:16]
            x1 = result.x[16:32]
            x2 = result.x[32:48]
            
            allocations = np.column_stack((x1, x2))
            payments = p
            
            # Clip to handle numerical noise
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            # Verification and repair pass: check IC/IR to 1e-8
            # Compute regret: for each t, t': p_t - p_t' - (v_t.x_t - v_t.x_t') should be <= 0
            # i.e., p_t - v_t.x_t <= p_t' - v_t.x_t' for all t, t'
            # Define u_t = p_t - v_t . x_t (utility when reporting truthfully)
            # IC: u_t >= u_t' for all t, t' (truthful reporting is optimal)
            # Wait, let me re-derive. 
            # A bidder with true type t reports t'. They get:
            # v_t . x_{t'} - p_{t'}
            # Truthful: v_t . x_t - p_t
            # IC: v_t . x_t - p_t >= v_t . x_{t'} - p_{t'} for all t, t'
            # => p_t - v_t . x_t <= p_{t'} - v_t . x_{t'} for all t, t'
            # This is NOT the same as p_t - p_{t'} <= v_t . x_t - v_t . x_{t'}
            # Let me recheck the LP constraint.
            # Our LP has: p_t - p_{t'} <= v_t . x_t - v_t . x_{t'}
            # Which is: p_t - v_t . x_t <= p_{t'} - v_t . x_{t'}
            # Yes, that's correct.
            
            # For verification, check: for all t, t': p[t] - p[t2] <= val[t]@x[t] - val[t]@x[t2]
            # With tolerance 1e-8
            tol = 1e-8
            max_violation = 0.0
            for t in range(n):
                for t2 in range(n):
                    lhs = payments[t] - payments[t2]
                    rhs = np.dot(val_matrix[t], allocations[t]) - np.dot(val_matrix[t], allocations[t2])
                    viol = lhs - rhs
                    if viol > max_violation:
                        max_violation = viol
            
            # Check IR: p_t <= v_t . x_t
            for t in range(n):
                viol = payments[t] - np.dot(val_matrix[t], allocations[t])
                if viol > max_violation:
                    max_violation = viol
            
            if max_violation > tol:
                # Try to repair by slightly reducing payments
                # Scale down payments proportionally
                scale = 1.0 - min(max_violation / (max(payments.max(), 1e-15)) * 2.0, 0.5)
                payments_repaired = payments * max(scale, 0.0)
                # Re-verify
                max_violation2 = 0.0
                for t in range(n):
                    for t2 in range(n):
                        lhs = payments_repaired[t] - payments_repaired[t2]
                        rhs = np.dot(val_matrix[t], allocations[t]) - np.dot(val_matrix[t], allocations[t2])
                        viol = lhs - rhs
                        if viol > max_violation2:
                            max_violation2 = viol
                for t in range(n):
                    viol = payments_repaired[t] - np.dot(val_matrix[t], allocations[t])
                    if viol > max_violation2:
                        max_violation2 = viol
                
                if max_violation2 <= tol:
                    payments = payments_repaired
                # If still violating, use fallback
                elif max_violation2 > 1e-6:
                    return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
            
            return {'allocations': allocations, 'payments': payments}
        else:
            # Fallback: trivial mechanism
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    except Exception:
        # Fallback: trivial mechanism
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
