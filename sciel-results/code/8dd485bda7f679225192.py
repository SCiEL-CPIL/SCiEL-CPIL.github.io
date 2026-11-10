"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_values = np.array(types, dtype=float)  # shape (16, 2)
    
    # Parse distribution_spec to get probability vector
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == 16:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, dict):
        probs = np.ones(16) / 16.0
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    prob_sum = probs.sum()
    if prob_sum > 1e-12:
        probs = probs / prob_sum
    else:
        probs = np.ones(16) / 16.0
    
    # LP formulation:
    # Variables: x = [q1(0), q2(0), p(0), q1(1), q2(1), p(1), ..., q1(15), q2(15), p(15)]
    n_vars = 3 * n_types  # 48
    
    # Objective: maximize sum(probs[i] * p(i))
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[3*i + 2] = -probs[i]
    
    # Build constraints efficiently
    A_rows = []
    b_vals = []
    
    # IC constraints: For all i, j
    # v_i . x_i - p_i >= v_i . x_j - p_j
    # p_i - p_j <= v_i . (x_i - x_j)
    # -v_i . x_i + v_i . x_j + p_i - p_j <= 0
    # Coefficients for type i: x_i: -v_i, p_i: 1
    # Coefficients for type j: x_j: v_i, p_j: -1
    for i in range(n_types):
        v_i = type_values[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # Term for type i
            row[3*i + 0] = -v_i[0]
            row[3*i + 1] = -v_i[1]
            row[3*i + 2] = 1.0
            # Term for type j
            row[3*j + 0] = v_i[0]
            row[3*j + 1] = v_i[1]
            row[3*j + 2] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
            
    # IR constraints: p_i <= v_i . x_i
    # p_i - v_i . x_i <= 0
    for i in range(n_types):
        v_i = type_values[i]
        row = np.zeros(n_vars)
        row[3*i + 0] = -v_i[0]
        row[3*i + 1] = -v_i[1]
        row[3*i + 2] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)
        
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    # Bounds
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, None))
    
    # Solve LP
    if time.time() > deadline - 5:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'dual_stopping_tolerance': 1e-12,
                              'primal_stopping_tolerance': 1e-12,
                              'time_limit': max(0, deadline - time.time() - 2)})
    
    if not result.success:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    x_sol = result.x
    
    # Extract initial allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for i in range(n_types):
        allocations[i, 0] = x_sol[3*i + 0]
        allocations[i, 1] = x_sol[3*i + 1]
        payments[i] = x_sol[3*i + 2]
        
    # Clip allocations to [0,1] to handle numerical noise
    allocations = np.clip(allocations, 0.0, 1.0)
    
    # Refinement: Tighten payments to maximum feasible values given the allocations
    # Constraints:
    # 1. p_i <= v_i . x_i (IR)
    # 2. p_i <= p_j + v_i . (x_i - x_j) (IC, rearranged: p_i - p_j <= v_i(x_i - x_j))
    #    Note: The IC constraint is symmetric in a way, but we must satisfy it for all i, j.
    #    Specifically: v_i . x_i - p_i >= v_i . x_j - p_j  => p_i <= p_j + v_i . (x_i - x_j)
    
    # We want to maximize sum(probs[i] * p_i).
    # This is a linear program again, but much smaller (16 vars, ~272 constraints).
    # However, we can solve it iteratively or just run linprog again on the payment sub-problem.
    
    if time.time() < deadline - 3:
        # Sub-LP for payments given fixed allocations
        n_p_vars = n_types
        c_p = -probs  # Maximize sum probs[i] p[i]
        
        A_p_rows = []
        b_p_vals = []
        
        # IR: p_i <= v_i . x_i
        for i in range(n_types):
            row = np.zeros(n_p_vars)
            row[i] = 1.0
            A_p_rows.append(row)
            b_p_vals.append(float(np.dot(type_values[i], allocations[i])))
            
        # IC: p_i - p_j <= v_i . (x_i - x_j)
        for i in range(n_types):
            v_i = type_values[i]
            x_i = allocations[i]
            for j in range(n_types):
                if i == j:
                    continue
                x_j = allocations[j]
                # p_i - p_j <= v_i . (x_i - x_j)
                row = np.zeros(n_p_vars)
                row[i] = 1.0
                row[j] = -1.0
                A_p_rows.append(row)
                b_p_vals.append(float(np.dot(v_i, x_i - x_j)))
                
        A_p = np.array(A_p_rows)
        b_p = np.array(b_p_vals)
        
        bounds_p = [(0, None)] * n_p_vars
        
        try:
            res_p = linprog(c_p, A_ub=A_p, b_ub=b_p, bounds=bounds_p, method='highs',
                            options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                     'primal_feasibility_tolerance': 1e-14,
                                     'dual_stopping_tolerance': 1e-14,
                                     'primal_stopping_tolerance': 1e-14,
                                     'time_limit': max(0, deadline - time.time() - 1)})
            
            if res_p.success:
                # Check if this improves revenue
                new_payments = np.maximum(res_p.x, 0)
                old_rev = np.sum(probs * payments)
                new_rev = np.sum(probs * new_payments)
                
                # Verify feasibility of new payments with current allocations
                feasible = True
                for i in range(n_types):
                    # IR
                    if new_payments[i] > np.dot(type_values[i], allocations[i]) + 1e-8:
                        feasible = False
                        break
                    # IC
                    for j in range(n_types):
                        util_i = np.dot(type_values[i], allocations[i]) - new_payments[i]
                        util_j = np.dot(type_values[i], allocations[j]) - new_payments[j]
                        if util_i < util_j - 1e-8:
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if feasible and new_rev > old_rev - 1e-12:
                    payments = new_payments
        except Exception:
            pass

    # Final sanity check and clip
    payments = np.maximum(payments, 0.0)
    allocations = np.clip(allocations, 0.0, 1.0)
    
    return {"allocations": allocations, "payments": payments}

# EVOLVE-BLOCK-END
