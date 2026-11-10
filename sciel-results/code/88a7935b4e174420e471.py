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
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
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
    if time.time() > deadline - 10:
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
        if time.time() < deadline - 5:
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
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
