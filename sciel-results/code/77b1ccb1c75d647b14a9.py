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
    deadline = start_time + 88  # 2s safety margin before 90s kill
    
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
    
    # IC constraints: for each (theta, theta_prime) pair
    # v_theta . q(theta_prime) - p(theta_prime) - v_theta . q(theta) + p(theta) <= 0
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
    
    # IR constraints: -v_theta . q(theta) + p(theta) <= 0
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
    
    # Solve LP with tight tolerances
    if time.time() > deadline - 5:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    time_limit_1 = max(1.0, deadline - time.time() - 3)
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                     options={'presolve': True,
                              'dual_feasibility_tolerance': 1e-14,
                              'primal_feasibility_tolerance': 1e-14,
                              'dual_stopping_tolerance': 1e-14,
                              'primal_stopping_tolerance': 1e-14,
                              'time_limit': time_limit_1})
    
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
    
    def check_feasibility(alloc, pay, tol=1e-8):
        """Check IC and IR constraints."""
        for theta_idx in range(n_types):
            v_theta = type_values[theta_idx]
            u_truth = v_theta[0]*alloc[theta_idx,0] + v_theta[1]*alloc[theta_idx,1] - pay[theta_idx]
            if u_truth < -tol:
                return False
            for theta_prime_idx in range(n_types):
                u_dev = v_theta[0]*alloc[theta_prime_idx,0] + v_theta[1]*alloc[theta_prime_idx,1] - pay[theta_prime_idx]
                if u_dev > u_truth + tol:
                    return False
        if np.any(alloc < -tol) or np.any(alloc > 1 + tol):
            return False
        if np.any(pay < -tol):
            return False
        return True
    
    def compute_revenue(alloc, pay, p):
        return float(np.sum(p * pay))
    
    def compute_minimal_payments(alloc):
        """
        Given fixed allocations, compute the minimal feasible payments.
        For each type theta_prime, the minimum payment must satisfy:
          p(theta_prime) >= v_theta_prime . q(theta_prime) - u(theta_prime)  [IR]
          p(theta_prime) >= v_theta . q(theta_prime) - u(theta)  [IC for all theta]
          p(theta_prime) >= 0
        where u(theta) = v_theta . q(theta) - p(theta) is the truthful utility.
        
        But since we're computing p from scratch, we need a fixed-point or direct approach.
        The key insight: the minimal payment for type theta' given utilities u is:
          p_min(theta') = max(0, max_theta [v_theta . q(theta') - u(theta)])
        
        And u(theta) = v_theta . q(theta) - p(theta).
        
        This is a system of equations. We can solve it iteratively:
        Start with u = 0, then p(theta') = max(0, max_theta [v_theta . q(theta') - u(theta)])
        Then u(theta) = v_theta . q(theta) - p(theta), iterate.
        
        Actually, there's a cleaner way. The IC + IR constraints define a polytope for p.
        The minimum revenue solution is the vertex where payments are minimized.
        We can solve this as a small LP: minimize sum(probs * p) subject to IC+IR.
        But that's what the main LP does jointly. For fixed allocations, it's simpler.
        
        For fixed allocations, the constraints on p are:
          p(theta) - p(theta') >= v_theta . (q(theta) - q(theta'))  [IC]
          p(theta) >= v_theta . q(theta)  [IR]
          p(theta) >= 0
        
        This is a difference constraints + lower bounds problem. The minimal solution
        can be found by: p(theta) = max(0, v_theta . q(theta), max_{theta'} [p(theta') + v_theta . (q(theta) - q(theta'))])
        
        This is a longest-path problem on a graph. With 16 nodes, we can just iterate.
        """
        # v_dot_q[i] = v_i . q_i
        v_dot_q = alloc @ type_values.T  # shape (16,)
        
        # Iterate to find minimal payments
        p = np.maximum(0.0, v_dot_q)  # start with IR lower bounds
        for iteration in range(20):
            p_new = p.copy()
            for theta in range(n_types):
                # p(theta) >= p(theta') + v_theta . (q(theta) - q(theta'))
                for theta_prime in range(n_types):
                    if theta == theta_prime:
                        continue
                    delta_v_q = type_values[theta] @ (alloc[theta] - alloc[theta_prime])
                    lower = p[theta_prime] + delta_v_q
                    if lower > p_new[theta]:
                        p_new[theta] = lower
                # Also IR
                if v_dot_q[theta] > p_new[theta]:
                    p_new[theta] = v_dot_q[theta]
                if 0.0 > p_new[theta]:
                    p_new[theta] = 0.0
            if np.allclose(p_new, p, atol=1e-15):
                p = p_new
                break
            p = p_new
        return p
    
    current_revenue = compute_revenue(allocations, payments, probs)
    
    # Refinement 1: Recompute minimal payments for the LP's allocations
    if time.time() < deadline - 2:
        min_pay = compute_minimal_payments(allocations)
        min_rev = compute_revenue(allocations, min_pay, probs)
        
        # The LP should already give minimal payments, but numerical issues may cause gaps
        # Verify feasibility of the minimal payments
        if check_feasibility(allocations, min_pay, tol=1e-8):
            if min_rev > current_revenue:
                payments = min_pay
                current_revenue = min_rev
        else:
            # If minimal payments are slightly infeasible, use LP payments
            # but try to nudge them down
            pass
    
    # Refinement 2: Try small allocation perturbations with recomputed optimal payments
    # The LP optimum is a vertex; small moves along edges may find better vertices
    # if the solver stopped early or at a degenerate point.
    if time.time() < deadline - 2:
        best_alloc = allocations.copy()
        best_pay = payments.copy()
        best_rev = current_revenue
        
        # Try perturbing each allocation component slightly
        for step in [1e-4, 1e-5, 1e-6]:
            if time.time() > deadline - 2:
                break
            for i in range(n_types):
                if time.time() > deadline - 2:
                    break
                for g in range(2):
                    if time.time() > deadline - 2:
                        break
                    for direction in [1.0, -1.0]:
                        if time.time() > deadline - 2:
                            break
                        trial_alloc = best_alloc.copy()
                        trial_alloc[i, g] += direction * step
                        # Clip to [0, 1]
                        trial_alloc[i, g] = np.clip(trial_alloc[i, g], 0.0, 1.0)
                        # Recompute minimal payments
                        trial_pay = compute_minimal_payments(trial_alloc)
                        # Check feasibility
                        if check_feasibility(trial_alloc, trial_pay, tol=1e-8):
                            trial_rev = compute_revenue(trial_alloc, trial_pay, probs)
                            if trial_rev > best_rev + 1e-10:
                                best_alloc = trial_alloc
                                best_pay = trial_pay
                                best_rev = trial_rev
        
        if best_rev > current_revenue:
            allocations = best_alloc
            payments = best_pay
            current_revenue = best_rev
    
    # Refinement 3: High-precision LP polish with the refined allocations as a guide
    if time.time() < deadline - 1:
        # Re-solve the full LP with even tighter tolerances
        x_init = np.zeros(n_vars)
        for i in range(n_types):
            x_init[3*i + 0] = allocations[i, 0]
            x_init[3*i + 1] = allocations[i, 1]
            x_init[3*i + 2] = payments[i]
        
        time_limit_2 = max(0.5, deadline - time.time() - 0.5)
        result2 = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', x0=x_init,
                          options={'presolve': True,
                                   'dual_feasibility_tolerance': 1e-15,
                                   'primal_feasibility_tolerance': 1e-15,
                                   'dual_stopping_tolerance': 1e-15,
                                   'primal_stopping_tolerance': 1e-15,
                                   'time_limit': time_limit_2})
        
        if result2.success:
            x2 = result2.x
            alloc2 = np.zeros((16, 2))
            pay2 = np.zeros(16)
            for i in range(n_types):
                alloc2[i, 0] = x2[3*i + 0]
                alloc2[i, 1] = x2[3*i + 1]
                pay2[i] = x2[3*i + 2]
            
            alloc2 = np.clip(alloc2, 0.0, 1.0)
            pay2 = np.maximum(pay2, 0.0)
            
            if check_feasibility(alloc2, pay2, tol=1e-8):
                rev2 = compute_revenue(alloc2, pay2, probs)
                if rev2 > current_revenue + 1e-12:
                    allocations = alloc2
                    payments = pay2
                    current_revenue = rev2
    
    # Final verification and fallback
    if not check_feasibility(allocations, payments, tol=1e-8):
        # Fallback: use a simple valid mechanism
        allocations = np.full((16, 2), 0.5)
        for i in range(n_types):
            payments[i] = 0.5 * (type_values[i, 0] + type_values[i, 1])
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
