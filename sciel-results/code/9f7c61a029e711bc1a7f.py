"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Specification of the type distribution
        support: Support of types (values in {0,1,2,3} for 2 goods)
        constraints: Additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = list(product(range(4), range(4)))
    n_types = len(types)
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, constraints)
    
    # Try the decoupled single-item approach first (faster, exact)
    result = _solve_decoupled(types, probs)
    
    # If that failed or timed out, fall back to LP
    if result is None or time.time() > deadline:
        result = _solve_auction_utility(types, probs)
    
    # If still failed, return baseline
    if result is None or time.time() > deadline:
        return _baseline_result(types, probs)
    
    # Final verification before returning
    if _verify_mechanism(types, probs, result['allocations'], result['payments'], eps=1e-8):
        return result
    else:
        return _baseline_result(types, probs)

def _solve_decoupled(types, probs):
    """
    Solve the auction by decoupling into two independent single-item auctions.
    
    For a single additive bidder with independent goods, the optimal IC mechanism
    for each good is a Myerson-style auction. With values in {0,1,2,3}, we can
    solve each good's allocation rule and payment rule independently.
    """
    n_types = len(types)
    values = [0, 1, 2, 3]
    n_vals = len(values)
    
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for good in range(2):
        # Get marginal distribution of value for this good
        marg_probs = np.zeros(n_vals)
        for t in range(n_types):
            v = types[t][good]
            marg_probs[v] += probs[t]
        
        # Solve single-item auction for this good
        x, p = _solve_single_item(marg_probs, values)
        
        # Assign to all types with this good's value
        for t in range(n_types):
            v = types[t][good]
            allocations[t, good] = x[v]
            payments[t] += p[v]
    
    # Verify the combined mechanism
    if _verify_mechanism(types, probs, allocations, payments, eps=1e-8):
        return {
            'allocations': allocations,
            'payments': payments
        }
    else:
        return None

def _solve_single_item(marg_probs, values):
    """
    Solve the optimal single-item, single-bidder auction with discrete values.
    """
    n_vals = len(values)
    
    # Variables: x[0], x[1], x[2], x[3] (allocation probabilities)
    # u[0], u[1], u[2], u[3] (utilities)
    # p[v] = v*x[v] - u[v]
    
    # Objective: maximize sum_v marg_probs[v] * p[v]
    c = np.zeros(2 * n_vals)
    for v in range(n_vals):
        c[v] = -marg_probs[v] * values[v]
        c[n_vals + v] = marg_probs[v]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # x non-decreasing
    for v in range(1, n_vals):
        row = np.zeros(2 * n_vals)
        row[v] = -1.0
        row[v-1] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC: u[v] - u[v-1] >= x[v]
    for v in range(1, n_vals):
        row = np.zeros(2 * n_vals)
        row[n_vals + v] = -1.0
        row[n_vals + v-1] = 1.0
        row[v] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IR: u[0] >= 0
    row = np.zeros(2 * n_vals)
    row[n_vals + 0] = -1.0
    A_ub_rows.append(row)
    b_ub_vals.append(0.0)
    
    # p[v] >= 0
    for v in range(n_vals):
        row = np.zeros(2 * n_vals)
        row[v] = -values[v]
        row[n_vals + v] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = []
    for v in range(n_vals):
        bounds.append((0, 1))
    for v in range(n_vals):
        bounds.append((0, None))
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'primal_feasibility_tolerance': 1e-12,
                             'dual_feasibility_tolerance': 1e-12, 'ipm_optimality_tolerance': 1e-12})
        
        if not res.success:
            return None, None
        
        x = res.x[:n_vals]
        u = res.x[n_vals:]
        p = np.array([values[v] * x[v] - u[v] for v in range(n_vals)])
        
        x = np.clip(x, 0.0, 1.0)
        p = np.maximum(p, 0.0)
        
        return x, p
        
    except Exception:
        return None, None

def _parse_distribution(distribution_spec, support, constraints):
    """Parse distribution spec into probability vector over 16 types."""
    n_types = 16
    
    if isinstance(distribution_spec, dict):
        dist_type = distribution_spec.get('type', 'uniform')
        if dist_type == 'uniform':
            return np.ones(n_types) / n_types
        elif dist_type == 'product':
            marginals = distribution_spec.get('marginals', None)
            if marginals is not None and len(marginals) == 2:
                p1, p2 = marginals
                probs = np.zeros(n_types)
                for i, (v1, v2) in enumerate(product(range(4), range(4))):
                    probs[i] = p1[v1] * p2[v2]
                return probs
            else:
                return np.ones(n_types) / n_types
        elif dist_type == 'explicit':
            explicit_probs = distribution_spec.get('probs', None)
            if explicit_probs is not None and len(explicit_probs) == n_types:
                probs = np.array(explicit_probs, dtype=float)
                probs = np.maximum(probs, 0)
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(n_types) / n_types
                return probs
            else:
                return np.ones(n_types) / n_types
    
    return np.ones(n_types) / n_types

def _solve_auction_utility(types, probs):
    """
    Solve the LP using utility-based formulation with monotonicity constraints.
    """
    n_types = len(types)
    n_goods = 2
    
    n_vars = 3 * n_types
    
    # Variable indexing:
    # x_1(t): index t
    # x_2(t): index n_types + t  
    # p(t): index 2 * n_types + t
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[2 * n_types + t] = -probs[t]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            
            v_t = np.array(types[t], dtype=float)
            row = np.zeros(n_vars)
            
            row[t] = -v_t[0]
            row[n_types + t] = -v_t[1]
            row[2 * n_types + t] = 1.0
            row[t_prime] += v_t[0]
            row[n_types + t_prime] += v_t[1]
            row[2 * n_types + t_prime] -= 1.0
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        row = np.zeros(n_vars)
        
        row[t] = -v_t[0]
        row[n_types + t] = -v_t[1]
        row[2 * n_types + t] = 1.0
        
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Add monotonicity constraints
    for j in range(n_goods):
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                if types[t][j] >= types[t_prime][j]:
                    row = np.zeros(n_vars)
                    row[j * n_types + t] = -1.0
                    row[j * n_types + t_prime] = 1.0
                    A_ub_rows.append(row)
                    b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'presolve': True, 'primal_feasibility_tolerance': 1e-12,
                                'dual_feasibility_tolerance': 1e-12, 'ipm_optimality_tolerance': 1e-12})
        
        if not result.success:
            return None
        
        x = result.x
        
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        
        for t in range(n_types):
            allocations[t, 0] = x[t]
            allocations[t, 1] = x[n_types + t]
            payments[t] = x[2 * n_types + t]
        
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        if _verify_mechanism(types, probs, allocations, payments, eps=1e-8):
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            return None
            
    except Exception:
        return None

def _verify_mechanism(types, probs, allocations, payments, eps=1e-8):
    """Verify that the mechanism satisfies all constraints."""
    n_types = len(types)
    
    if np.any(allocations < -eps) or np.any(allocations > 1 + eps):
        return False
    if np.any(payments < -eps):
        return False
    
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_t = v_t @ allocations[t] - payments[t]
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            util_misreport = v_t @ allocations[t_prime] - payments[t_prime]
            if util_misreport > util_t + eps:
                return False
    
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_t = v_t @ allocations[t] - payments[t]
        if util_t < -eps:
            return False
    
    return True

def _baseline_result(types, probs):
    """Return a simple baseline mechanism that is always feasible."""
    n_types = len(types)
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
