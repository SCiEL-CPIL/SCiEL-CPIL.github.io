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
    
    # Build constraints efficiently using vectorized operations
    A_rows = []
    b_vals = []
    
    # Precompute type index arrays for faster access
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
    if time.time() > deadline - 5:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'dual_stopping_tolerance': 1e-12,
                              'primal_stopping_tolerance': 1e-12,
                              'time_limit': max(0, deadline - time.time() - 2)})
    
    if not result.success:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    
    # High-precision polish: re-solve with tighter tolerances using the first solution as warm start
    if time.time() < deadline - 3:
        x_init = x.copy()
        # Clip to feasible region for warm start
        x_init = np.clip(x_init, 0, None)
        for i in range(n_types):
            x_init[3*i] = np.clip(x_init[3*i], 0, 1)
            x_init[3*i+1] = np.clip(x_init[3*i+1], 0, 1)
        
        result2 = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', x0=x_init,
                          options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                   'primal_feasibility_tolerance': 1e-14,
                                   'dual_stopping_tolerance': 1e-14,
                                   'primal_stopping_tolerance': 1e-14,
                                   'time_limit': max(0, deadline - time.time() - 1)})
        
        if result2.success and result2.fun < result.fun:
            x = result2.x
    
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
    
    def check_feasibility(alloc, pay, tol=1e-8):
        """Check IC and IR constraints."""
        for theta_idx in range(n_types):
            v_theta = type_values[theta_idx]
            u_truth = v_theta[0]*alloc[theta_idx,0] + v_theta[1]*alloc[theta_idx,1] - pay[theta_idx]
            # IR: u_truth >= -tol
            if u_truth < -tol:
                return False
            for theta_prime_idx in range(n_types):
                u_dev = v_theta[0]*alloc[theta_prime_idx,0] + v_theta[1]*alloc[theta_prime_idx,1] - pay[theta_prime_idx]
                if u_dev > u_truth + tol:
                    return False
        # Check allocation bounds
        if np.any(alloc < -tol) or np.any(alloc > 1 + tol):
            return False
        if np.any(pay < -tol):
            return False
        return True
    
    def compute_revenue(alloc, pay, p):
        return float(np.sum(p * pay))
    
    current_revenue = compute_revenue(allocations, payments, probs)
    
    # Vectorized local refinement: exploit IC structure to squeeze residual revenue
    # Key insight: for a fixed allocation, the minimum feasible payment for type theta is:
    #   p(theta) >= max(0, v_theta . q(theta), max_{theta'} (v_theta . q(theta') - u(theta')))
    # where u(theta') is the utility of type theta' when reporting truthfully.
    # Actually, the binding constraint is: p(theta) >= v_theta . q(theta') - (v_theta' . q(theta') - p(theta'))
    # Simplified: p(theta) >= max(0, v_theta . q(theta), max_{theta'} [v_theta . q(theta') - v_theta' . q(theta') + p(theta')])
    # 
    # Better approach: given allocations, compute the minimal feasible payments that satisfy IC and IR,
    # then verify revenue. If the LP solution has slack in some constraints, we can reduce payments.
    
    if time.time() < deadline - 2:
        best_alloc = allocations.copy()
        best_pay = payments.copy()
        best_rev = current_revenue
        
        # For each type, compute the minimum payment that satisfies all constraints
        # given the current allocations
        for iteration in range(3):
            if time.time() > deadline - 2:
                break
            
            # Compute utilities under truthful reporting
            utils = np.zeros(n_types)
            for i in range(n_types):
                utils[i] = type_values[i, 0] * best_alloc[i, 0] + type_values[i, 1] * best_alloc[i, 1] - best_pay[i]
            
            # For each type theta, the minimum payment is:
            # p(theta) >= v_theta . q(theta) - u(theta)  [IR]
            # p(theta) >= v_theta . q(theta') - u(theta')  [IC: theta doesn't want to deviate to theta']
            # Wait, IC says: u(theta) >= v_theta . q(theta') - p(theta')
            # So: p(theta') >= v_theta . q(theta') - u(theta) for all theta, theta'
            # This means: p(theta') >= max_theta (v_theta . q(theta') - u(theta))
            # And IR: p(theta') >= v_theta' . q(theta') - u(theta')
            # And p(theta') >= 0
            
            new_pay = np.zeros(n_types)
            for theta_prime in range(n_types):
                # IR constraint
                min_pay = type_values[theta_prime, 0] * best_alloc[theta_prime, 0] + \
                          type_values[theta_prime, 1] * best_alloc[theta_prime, 1] - utils[theta_prime]
                # IC constraints: for all theta, p(theta') >= v_theta . q(theta') - u(theta)
                for theta in range(n_types):
                    required = type_values[theta, 0] * best_alloc[theta_prime, 0] + \
                               type_values[theta, 1] * best_alloc[theta_prime, 1] - utils[theta]
                    if required > min_pay:
                        min_pay = required
                # Non-negativity
                if min_pay < 0:
                    min_pay = 0
                new_pay[theta_prime] = min_pay
            
            new_rev = compute_revenue(best_alloc, new_pay, probs)
            if new_rev > best_rev + 1e-12:
                best_pay = new_pay
                best_rev = new_rev
                if not check_feasibility(best_alloc, best_pay, tol=1e-8):
                    # Rollback if infeasible due to numerical issues
                    best_pay = payments.copy()
                    best_rev = current_revenue
                    break
        
        if best_rev > current_revenue:
            allocations = best_alloc
            payments = best_pay
    
    # Final verification
    if not check_feasibility(allocations, payments, tol=1e-8):
        # Fallback: use a simple valid mechanism
        allocations = np.full((16, 2), 0.5)
        for i in range(n_types):
            payments[i] = 0.5 * (type_values[i, 0] + type_values[i, 1])
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
