"""Incentive-compatible finite auction design via direct LP solve."""
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
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        probs = np.array(distribution_spec.get('probs', distribution_spec.get('probabilities', np.ones(16)/16.0)))
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Support: 16 type profiles (v1, v2) where v1, v2 in {0, 1, 2, 3}
    if support is None:
        support = [(i, j) for i in range(4) for j in range(4)]
    support = list(support)
    
    n_types = len(support)
    # We assume n_types = 16 based on the task description
    
    # Index mapping: type t has value profile (v1_t, v2_t)
    # Values are in {0, 1, 2, 3}
    
    # Extract values
    values = np.array(support, dtype=float)  # shape (16, 2)
    v1 = values[:, 0]
    v2 = values[:, 1]
    
    # Decision variables:
    # x[t, g] = probability of allocating good g to bidder when type is t
    # p[t] = payment when type is t
    # 
    # Total variables: 16*2 + 16 = 48
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # In linprog form (minimize), we minimize -sum_t probs[t] * p[t]
    
    # Constraints:
    # 1. Feasibility: 0 <= x[t, g] <= 1 for all t, g
    # 2. Individual Rationality: v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= 0 for all t
    #    i.e., -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    # 3. Incentive Compatibility (truthfulness):
    #    For all t, t':
    #    v1[t]*x[t',0] + v2[t]*x[t',1] - p[t'] <= v1[t']*x[t',0] + v2[t']*x[t',1] - p[t']
    #    i.e., (v1[t] - v1[t'])*x[t',0] + (v2[t] - v2[t'])*x[t',1] + p[t'] - p[t'] <= 0
    #    Wait, let me re-derive.
    #    
    #    Utility of type t reporting t' is: v1[t]*x[t',0] + v2[t]*x[t',1] - p[t']
    #    Utility of type t reporting truthfully t is: v1[t]*x[t,0] + v2[t]*x[t,1] - p[t]
    #    
    #    IC: v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= v1[t]*x[t',0] + v2[t]*x[t',1] - p[t']
    #    for all t, t'
    #    
    #    Rearranging:
    #    v1[t]*(x[t,0] - x[t',0]) + v2[t]*(x[t,1] - x[t',1]) - (p[t] - p[t']) >= 0
    #    i.e., -v1[t]*(x[t,0] - x[t',0]) - v2[t]*(x[t,1] - x[t',1]) + (p[t] - p[t']) <= 0
    
    # Let's set up the LP properly.
    # Variables: x[0..15, 0..1], p[0..15]
    # Index: variable i corresponds to:
    #   i in [0, 32): x[i//2, i%2]
    #   i in [32, 48): p[i-32]
    
    n_x = n_types * 2  # 32
    n_p = n_types  # 16
    n_vars = n_x + n_p  # 48
    
    # Objective: minimize -sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    # IR constraints: for each t, -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    # This gives n_types = 16 constraints
    
    # IC constraints: for each pair (t, t'), 
    # -v1[t]*x[t,0] - v2[t]*x[t,1] + v1[t]*x[t',0] + v2[t]*x[t',1] + p[t] - p[t'] <= 0
    # Wait, let me redo this carefully.
    
    # IC(t, t'): utility(t, t) >= utility(t, t')
    # v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= v1[t]*x[t',0] + v2[t]*x[t',1] - p[t']
    # v1[t]*x[t,0] + v2[t]*x[t,1] - v1[t]*x[t',0] - v2[t]*x[t',1] - p[t] + p[t'] >= 0
    # Multiply by -1:
    # -v1[t]*x[t,0] - v2[t]*x[t,1] + v1[t]*x[t',0] + v2[t]*x[t',1] + p[t] - p[t'] <= 0
    
    # Number of IC constraints: n_types * n_types = 256
    # But we can reduce: IC(t,t) is trivially 0 <= 0, so skip those.
    # Actually, let's include all for simplicity, or just t != t'.
    
    # Total constraints: 16 (IR) + 256 (IC) = 272
    n_constraints = n_types + n_types * n_types  # 16 + 256 = 272
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # IR constraints
    for t in range(n_types):
        row = t
        A_ub[row, t] = -v1[t]      # coefficient of x[t, 0]
        A_ub[row, t + n_types] = -v2[t]  # coefficient of x[t, 1]
        A_ub[row, n_x + t] = 1.0   # coefficient of p[t]
        b_ub[row] = 0.0
    
    # IC constraints
    constraint_idx = n_types
    for t in range(n_types):
        for tp in range(n_types):
            row = constraint_idx
            constraint_idx += 1
            if t == tp:
                # Trivial constraint: 0 <= 0, skip or set to 0
                continue
            # -v1[t]*x[t,0] - v2[t]*x[t,1] + v1[t]*x[tp,0] + v2[t]*x[tp,1] + p[t] - p[tp] <= 0
            A_ub[row, t] = -v1[t]
            A_ub[row, t + n_types] = -v2[t]
            A_ub[row, tp] = v1[t]
            A_ub[row, tp + n_types] = v2[t]
            A_ub[row, n_x + t] = 1.0
            A_ub[row, n_x + tp] = -1.0
            b_ub[row] = 0.0
    
    # Bounds: 0 <= x[t,g] <= 1, 0 <= p[t] (nonnegative payments)
    bounds = [(0, 1)] * n_x + [(0, None)] * n_p
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a simple mechanism
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x_sol = result.x[:n_x].reshape(n_types, 2)
        p_sol = result.x[n_x:]
        
        # Ensure numerical precision
        x_sol = np.clip(x_sol, 0, 1)
        p_sol = np.maximum(p_sol, 0)
        
        # Check time
        if time.time() > deadline:
            return {'allocations': x_sol, 'payments': p_sol}
        
        # Verify and polish if needed
        # Check IC and IR constraints with tolerance
        max_violation = 0.0
        for t in range(n_types):
            # IR
            util_t = v1[t]*x_sol[t, 0] + v2[t]*x_sol[t, 1] - p_sol[t]
            max_violation = max(max_violation, -util_t)
            
            # IC
            for tp in range(n_types):
                util_tp = v1[t]*x_sol[tp, 0] + v2[t]*x_sol[tp, 1] - p_sol[tp]
                violation = util_t - util_tp
                max_violation = max(max_violation, -violation)
        
        if max_violation > 1e-8:
            # Fallback to a simple valid mechanism if LP solution is invalid
            # Mechanism: Allocate good g if value >= 1, pay 1.
            # This is IC and IR.
            alloc_fallback = np.zeros((n_types, 2))
            pay_fallback = np.zeros(n_types)
            for t in range(n_types):
                if v1[t] >= 1.0:
                    alloc_fallback[t, 0] = 1.0
                    pay_fallback[t] += 1.0
                if v2[t] >= 1.0:
                    alloc_fallback[t, 1] = 1.0
                    pay_fallback[t] += 1.0
            
            # Verify fallback (should be valid)
            max_violation_fb = 0.0
            for t in range(n_types):
                util_t = v1[t]*alloc_fallback[t, 0] + v2[t]*alloc_fallback[t, 1] - pay_fallback[t]
                max_violation_fb = max(max_violation_fb, -util_t)
                for tp in range(n_types):
                    util_tp = v1[t]*alloc_fallback[tp, 0] + v2[t]*alloc_fallback[tp, 1] - pay_fallback[tp]
                    violation = util_t - util_tp
                    max_violation_fb = max(max_violation_fb, -violation)
            
            if max_violation_fb <= 1e-8:
                return {'allocations': alloc_fallback, 'payments': pay_fallback}
            else:
                # Last resort: Zero mechanism
                return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
        
        return {'allocations': x_sol, 'payments': p_sol}
    
    except Exception as e:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
