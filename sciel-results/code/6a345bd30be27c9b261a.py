"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 50  # 50 second deadline with 40s safety margin
    
    # Parse the support: 16 reports, each with 2 values in {0,1,2,3}
    if support is None or len(support) != 16:
        support = [(i, j) for i in range(4) for j in range(4)]
    
    # Parse distribution_spec to get probabilities for each report
    if isinstance(distribution_spec, dict) and 'probs' in distribution_spec:
        probs = np.array(distribution_spec['probs'], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Parse constraints
    if constraints is None:
        constraints = {}
    
    # Extract constraint parameters
    max_regret = constraints.get('max_regret', 1e-8)
    
    n_reports = 16
    n_goods = 2
    
    # Values matrix: 16 x 2
    values = np.array(support, dtype=float)
    
    # Decision variables:
    # x: 16*2 = 32 variables (allocations)
    # p: 16 variables (payments)
    # Total: 48 variables
    
    # Objective: maximize sum_t probs[t] * p[t]
    # In linprog, we minimize, so minimize -sum_t probs[t] * p[t]
    c = np.zeros(48)
    c[32:48] = -probs
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    # Constraint set 1: Incentive compatibility
    # For all t, t': p[t] - p[t'] <= v[t] · x[t] - v[t] · x[t']
    # Rearranged: -v[t] · x[t] + v[t] · x[t'] + p[t] - p[t'] <= 0
    # In terms of variables:
    # For each t: coefficient of x[t][g] = -v[t][g]
    # For each t': coefficient of x[t'][g] = +v[t][g]  <-- NOTE: v[t], NOT v[t']
    # Coefficient of p[t] = 1
    # Coefficient of p[t'] = -1
    
    A_rows = []
    b_rows = []
    
    for t in range(n_reports):
        for t_prime in range(n_reports):
            row = np.zeros(48)
            # x[t][g] coefficients: -v[t][g]
            for g in range(n_goods):
                row[t * n_goods + g] = -values[t][g]
            # x[t'][g] coefficients: +v[t][g]
            for g in range(n_goods):
                row[t_prime * n_goods + g] = values[t][g]
            # p[t] coefficient: 1
            row[32 + t] = 1
            # p[t'] coefficient: -1
            row[32 + t_prime] = -1
            A_rows.append(row)
            b_rows.append(0.0)
    
    # Constraint set 2: Individual rationality
    # v[t] · x[t] - p[t] >= 0 => p[t] - v[t] · x[t] <= 0
    # In terms of variables:
    # For each t: coefficient of x[t][g] = -v[t][g]
    # Coefficient of p[t] = 1
    
    for t in range(n_reports):
        row = np.zeros(48)
        for g in range(n_goods):
            row[t * n_goods + g] = -values[t][g]
        row[32 + t] = 1
        A_rows.append(row)
        b_rows.append(0.0)
    
    # Constraint set 3: Nonnegative payments
    # p[t] >= 0 => -p[t] <= 0
    for t in range(n_reports):
        row = np.zeros(48)
        row[32 + t] = -1
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds:
    # x[t][g]: 0 <= x <= 1
    # p[t]: p >= 0 (unbounded above, but bounded by IR)
    bounds = []
    for i in range(32):  # allocations
        bounds.append((0, 1))
    for i in range(16):  # payments
        bounds.append((0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        
        if result.success:
            x_flat = result.x[:32]
            p = result.x[32:48]
            
            # Reshape allocations
            allocations = x_flat.reshape(n_reports, n_goods)
            
            # Clip to ensure feasibility strictly
            allocations = np.clip(allocations, 0, 1)
            p = np.clip(p, 0, None)
            
            # Post-processing to ensure strict IC and IR satisfaction within tolerance
            # Re-calculate payments to ensure IR holds exactly if possible, or adjust slightly
            # However, LP solution should be optimal. We just need to ensure no numerical drift violates regret.
            
            # Verify and fix if necessary
            # Check IR: p[t] <= v[t] dot x[t]
            for t in range(n_reports):
                max_p = np.dot(values[t], allocations[t])
                if p[t] > max_p + 1e-9:
                    p[t] = max_p
            
            # Check IC: p[t] - p[t'] <= v[t] dot (x[t] - x[t'])
            # If violated, we might need to adjust. But LP should satisfy it.
            # Given the small size, we can do a quick check.
            
            return {
                'allocations': allocations,
                'payments': p
            }
        else:
            # Fallback: simple mechanism
            allocations = np.zeros((n_reports, n_goods))
            p = np.zeros(n_reports)
            return {
                'allocations': allocations,
                'payments': p
            }
    except Exception as e:
        # Fallback on any error
        allocations = np.zeros((n_reports, n_goods))
        p = np.zeros(n_reports)
        return {
            'allocations': allocations,
            'payments': p
        }
    
    # Check deadline
    if time.time() > deadline:
        allocations = np.zeros((n_reports, n_goods))
        p = np.zeros(n_reports)
        return {
            'allocations': allocations,
            'payments': p
        }
# EVOLVE-BLOCK-END
