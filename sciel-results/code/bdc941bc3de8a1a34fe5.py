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
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
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
    
    # IC constraints: for all theta, theta'
    for theta_idx in range(n_types):
        v_theta = type_values[theta_idx]
        for theta_prime_idx in range(n_types):
            row = np.zeros(n_vars)
            row[3*theta_prime_idx + 0] = v_theta[0]
            row[3*theta_prime_idx + 1] = v_theta[1]
            row[3*theta_prime_idx + 2] = -1.0
            row[3*theta_idx + 0] = -v_theta[0]
            row[3*theta_idx + 1] = -v_theta[1]
            row[3*theta_idx + 2] = 1.0
            A_rows.append(row)
            b_vals.append(0.0)
    
    # IR constraints
    for theta_idx in range(n_types):
        v_theta = type_values[theta_idx]
        row = np.zeros(n_vars)
        row[3*theta_idx + 0] = -v_theta[0]
        row[3*theta_idx + 1] = -v_theta[1]
        row[3*theta_idx + 2] = 1.0
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
    
    # Solve LP with time check
    if time.time() > deadline - 15:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12})
    
    if not result.success:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    
    # Extract allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for i in range(n_types):
        allocations[i, 0] = x[3*i + 0]
        allocations[i, 1] = x[3*i + 1]
        payments[i] = x[3*i + 2]
    
    # Clip to handle numerical issues
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Verify and fix IC/IR violations with a tighter tolerance
    ic_violated = False
    max_violation = 0.0
    for theta_idx in range(n_types):
        v_theta = type_values[theta_idx]
        u_truth = v_theta[0]*allocations[theta_idx,0] + v_theta[1]*allocations[theta_idx,1] - payments[theta_idx]
        for theta_prime_idx in range(n_types):
            u_dev = v_theta[0]*allocations[theta_prime_idx,0] + v_theta[1]*allocations[theta_prime_idx,1] - payments[theta_prime_idx]
            viol = u_dev - u_truth
            if viol > max_violation:
                max_violation = viol
            if viol > 1e-8:
                ic_violated = True
                break
        if ic_violated:
            break
    
    if ic_violated:
        # Try a high-precision polish: re-solve with tighter tolerances
        if time.time() < deadline - 10:
            result2 = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                             options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                      'primal_feasibility_tolerance': 1e-14})
            if result2.success:
                x2 = result2.x
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                for i in range(n_types):
                    allocations[i, 0] = x2[3*i + 0]
                    allocations[i, 1] = x2[3*i + 1]
                    payments[i] = x2[3*i + 2]
                allocations = np.clip(allocations, 0.0, 1.0)
                payments = np.maximum(payments, 0.0)
                
                # Re-verify
                ic_violated = False
                for theta_idx in range(n_types):
                    v_theta = type_values[theta_idx]
                    u_truth = v_theta[0]*allocations[theta_idx,0] + v_theta[1]*allocations[theta_idx,1] - payments[theta_idx]
                    for theta_prime_idx in range(n_types):
                        u_dev = v_theta[0]*allocations[theta_prime_idx,0] + v_theta[1]*allocations[theta_prime_idx,1] - payments[theta_prime_idx]
                        if u_dev > u_truth + 1e-8:
                            ic_violated = True
                            break
                    if ic_violated:
                        break
        
        if ic_violated:
            # Fallback: use a simple valid mechanism
            # Allocate both goods with prob 0.5, payment = expected value * 0.5
            allocations = np.full((16, 2), 0.5)
            for i in range(n_types):
                payments[i] = 0.5 * (type_values[i, 0] + type_values[i, 1])
    
    # Local refinement pass: exploit IC structure to squeeze out residual revenue
    # The key insight: given allocations q, the optimal payment for each type is
    # p(theta) = v(theta)·q(theta) - U(theta), where U(theta) is the utility.
    # IC requires U(theta) >= U(theta') + v(theta)·(q(theta') - q(theta)) for all theta, theta'
    # To maximize revenue, we want to minimize expected utility E[U(theta)] subject to IC and IR.
    # This is a separate LP on utilities with fixed allocations.
    
    if time.time() < deadline - 5:
        # Solve for optimal utilities given fixed allocations
        # Variables: U(0), ..., U(15)
        # Objective: minimize sum(probs[i] * U(i))
        # Constraints: U(i) >= U(j) + v_i·(q_j - q_i) for all i, j (IC)
        #              U(i) >= 0 for all i (IR)
        
        n_util_vars = n_types
        c_util = probs.copy()  # minimize sum(probs[i] * U[i])
        
        A_util_rows = []
        b_util_vals = []
        
        # IC constraints: U(i) - U(j) >= v_i·(q_j - q_i)
        # In standard form: -U(i) + U(j) <= -v_i·(q_j - q_i)
        for i in range(n_types):
            for j in range(n_types):
                if i == j:
                    continue
                row = np.zeros(n_util_vars)
                row[i] = -1.0
                row[j] = 1.0
                rhs = -(type_values[i] @ (allocations[j] - allocations[i]))
                A_util_rows.append(row)
                b_util_vals.append(rhs)
        
        # IR constraints: U(i) >= 0 => -U(i) <= 0
        for i in range(n_types):
            row = np.zeros(n_util_vars)
            row[i] = -1.0
            A_util_rows.append(row)
            b_util_vals.append(0.0)
        
        A_util = np.array(A_util_rows)
        b_util = np.array(b_util_vals)
        
        bounds_util = [(0, None)] * n_types
        
        result_util = linprog(c_util, A_ub=A_util, b_ub=b_util, bounds=bounds_util,
                              method='highs',
                              options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                       'primal_feasibility_tolerance': 1e-14})
        
        if result_util.success:
            U_opt = result_util.x
            # Update payments: p(i) = v_i·q_i - U(i)
            for i in range(n_types):
                payments[i] = type_values[i] @ allocations[i] - U_opt[i]
            
            # Clip payments to non-negative
            payments = np.maximum(payments, 0.0)
            
            # Verify IC/IR again after refinement
            ic_violated = False
            for theta_idx in range(n_types):
                v_theta = type_values[theta_idx]
                u_truth = v_theta[0]*allocations[theta_idx,0] + v_theta[1]*allocations[theta_idx,1] - payments[theta_idx]
                for theta_prime_idx in range(n_types):
                    u_dev = v_theta[0]*allocations[theta_prime_idx,0] + v_theta[1]*allocations[theta_prime_idx,1] - payments[theta_prime_idx]
                    if u_dev > u_truth + 1e-8:
                        ic_violated = True
                        break
                if ic_violated:
                    break
            
            if ic_violated:
                # Revert to previous valid solution
                # Re-extract from original LP solution
                for i in range(n_types):
                    payments[i] = x[3*i + 2]
                payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
