"""Incentive-compatible finite auction design - optimal LP-based mechanism."""
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an optimal incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Distribution specification (list of 16 probabilities or similar)
        support: Support specification (16 types, 2 goods)
        constraints: Constraints dictionary
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 45.0  # Safety margin before 90s limit
    
    # Define the 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    
    # Parse distribution probabilities
    probs = _parse_distribution(distribution_spec, n_types)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return _fallback_result()
    
    # Use heuristic search for optimal mechanism
    result = _solve_heuristic_auction(probs, types, n_types, n_goods, start_time, DEADLINE)
    
    return result


def _parse_distribution(distribution_spec, n_types):
    """Parse distribution specification into probability vector."""
    probs = np.zeros(n_types)
    
    if distribution_spec is None:
        return probs / n_types  # Uniform fallback
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_types,):
            probs = arr
        elif arr.shape == (4, 4):
            probs = arr.flatten()
        elif arr.size == n_types:
            probs = arr.flatten()
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        # Try to parse from dict
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    
    # Normalize to ensure valid probabilities
    total = probs.sum()
    if total > 1e-12:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    # Clip small negative values
    probs = np.clip(probs, 0, 1)
    
    return probs


def _solve_heuristic_auction(probs, types, n_types, n_goods, start_time, deadline):
    """
    Solve the auction problem using a Linear Programming formulation.
    Variables: x_{t,g} (allocations) and u_t (utilities).
    Objective: Maximize expected revenue = sum_t probs[t] * (dot(v_t, x_t) - u_t)
    Constraints:
    1. u_t >= 0 (IR)
    2. u_t >= u_{t'} + dot(v_t - v_{t'}, x_{t'}) (IC)
    3. 0 <= x_{t,g} <= 1
    """
    
    # Variable ordering:
    # 0 .. 31: x_{t,g} for t in 0..15, g in 0..1
    # 32 .. 47: u_t for t in 0..15
    
    n_vars = n_types * n_goods + n_types
    x_start = 0
    u_start = n_types * n_goods
    
    # Objective coefficients: c_j such that we minimize -c^T z
    # Maximize sum_t pi_t (sum_g v_{t,g} x_{t,g} - u_t)
    # = sum_{t,g} pi_t v_{t,g} x_{t,g} - sum_t pi_t u_t
    # Minimize: sum_{t,g} -pi_t v_{t,g} x_{t,g} + sum_t pi_t u_t
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(n_goods):
            c[x_start + t * n_goods + g] = -probs[t] * types[t][g]
        c[u_start + t] = probs[t]
        
    # Inequality constraints: A_ub @ z <= b_ub
    # We have two types of constraints:
    # 1. u_t >= 0  =>  -u_t <= 0
    # 2. u_t - u_{t'} - sum_g (v_{t,g} - v_{t',g}) x_{t',g} >= 0 
    #    => -u_t + u_{t'} + sum_g (v_{t,g} - v_{t',g}) x_{t',g} <= 0
    
    A_rows = []
    b_rows = []
    
    # 1. IR constraints: -u_t <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
        
    # 2. IC constraints
    # For each pair (t, t'):
    # u_t >= u_{t'} + sum_g (v_{t,g} - v_{t',g}) x_{t',g}
    # -u_t + u_{t'} + sum_g (v_{t,g} - v_{t',g}) x_{t',g} <= 0
    
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            # Coefficient for u_t is -1
            row[u_start + t] = -1.0
            # Coefficient for u_{t'} is 1
            row[u_start + t_prime] = 1.0
            # Coefficients for x_{t',g}
            for g in range(n_goods):
                coeff = types[t][g] - types[t_prime][g]
                if coeff != 0:
                    row[x_start + t_prime * n_goods + g] = coeff
            A_rows.append(row)
            b_rows.append(0.0)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds for variables
    # x_{t,g} in [0, 1]
    # u_t in [0, inf) (bounded by IR constraint, but we can set upper bound to a large number or None)
    # Note: linprog requires bounds. We can set upper bound for u to something large, e.g., max_value * max_prob / min_prob? 
    # Or just None (infinity). SciPy linprog handles None as infinity.
    
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
        
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return _fallback_result()
            
        z = res.x
        x = z[x_start:x_start + n_types * n_goods].reshape(n_types, n_goods)
        u = z[u_start:u_start + n_types]
        
        # Calculate payments
        p = np.zeros(n_types)
        for t in range(n_types):
            p[t] = np.dot(x[t], types[t]) - u[t]
            
        # Verify constraints numerically to ensure no numerical errors
        # Check IR
        if np.any(u < -1e-8) or np.any(p < -1e-8):
            # If payments are slightly negative due to numerical error, clip them?
            # But if p < 0 significantly, it's invalid.
            # The LP should ensure u >= 0 and p = v.x - u. 
            # If u is large, p can be negative. 
            # Wait, IR is u >= 0. 
            # Individual Rationality usually means utility >= 0.
            # Non-negative payments is a separate constraint?
            # The task says: "require feasible probabilities, nonnegative payments, individual rationality"
            # So we need p >= 0 AND u >= 0.
            # My LP only enforced u >= 0.
            # I need to add p >= 0 constraints.
            # p_t = v_t . x_t - u_t >= 0  =>  u_t - v_t . x_t <= 0
            # This is another set of linear constraints.
            
            # Let's re-solve with p >= 0 constraints.
            return _solve_lp_with_nonneg_payments(probs, types, n_types, n_goods)
            
        return {
            'allocations': x,
            'payments': p
        }
    except Exception:
        return _fallback_result()

def _solve_lp_with_nonneg_payments(probs, types, n_types, n_goods):
    """
    Solve LP with explicit non-negative payment constraints.
    """
    n_vars = n_types * n_goods + n_types
    x_start = 0
    u_start = n_types * n_goods
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(n_goods):
            c[x_start + t * n_goods + g] = -probs[t] * types[t][g]
        c[u_start + t] = probs[t]
        
    A_rows = []
    b_rows = []
    
    # 1. IR: u_t >= 0 => -u_t <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
        
    # 2. IC: -u_t + u_{t'} + sum_g (v_{t,g} - v_{t',g}) x_{t',g} <= 0
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[u_start + t] = -1.0
            row[u_start + t_prime] = 1.0
            for g in range(n_goods):
                coeff = types[t][g] - types[t_prime][g]
                if coeff != 0:
                    row[x_start + t_prime * n_goods + g] = coeff
            A_rows.append(row)
            b_rows.append(0.0)
            
    # 3. Non-negative payments: p_t = v_t . x_t - u_t >= 0 => u_t - sum_g v_{t,g} x_{t,g} <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = 1.0
        for g in range(n_goods):
            row[x_start + t * n_goods + g] = -types[t][g]
        A_rows.append(row)
        b_rows.append(0.0)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
        
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return _fallback_result()
            
        z = res.x
        x = z[x_start:x_start + n_types * n_goods].reshape(n_types, n_goods)
        u = z[u_start:u_start + n_types]
        
        p = np.zeros(n_types)
        for t in range(n_types):
            p[t] = np.dot(x[t], types[t]) - u[t]
            
        # Final sanity check
        if np.any(p < -1e-8) or np.any(u < -1e-8):
             # Clip small negatives if any
             p = np.clip(p, 0, None)
             # But if we clipped, IR might be violated? 
             # If p was -1e-9, clipping to 0 increases utility, which is fine for IC/IR?
             # Actually, increasing p decreases utility. 
             # If p < 0, it means we are paying the bidder. 
             # If we set p=0, utility increases. 
             # IC: u_t >= u_{t'} + ...
             # If we increase u_t (by reducing p), it might violate IC for other types?
             # No, u_t is a variable. If the LP solution had p < 0, it means the constraint p>=0 was violated.
             # But we added p>=0 constraints. So this shouldn't happen unless numerical error.
             # If it happens, we return fallback to be safe.
             if np.min(p) < -1e-6:
                 return _fallback_result()
             p = np.clip(p, 0, None)
             
        return {
            'allocations': x,
            'payments': p
        }
    except Exception:
        return _fallback_result()


def _compute_min_payments(x, types):
    """
    Compute the minimal payments that satisfy IC and IR for a given allocation x.
    Returns (payments, valid).
    """
    n_types = len(types)
    
    # We need to find U(t) such that:
    # 1. U(t) >= 0
    # 2. U(t) >= U(t') + sum_g (t_g - t'_g) x(t', g)
    
    # This is a longest path problem on a graph where edge t' -> t has weight
    # w(t', t) = sum_g (t_g - t'_g) x(t', g)
    # We add a source node S with U(S) = 0 and edges S -> t with weight 0.
    
    # Since n_types is small (16), we can use Bellman-Ford for longest paths.
    # Note: Bellman-Ford finds shortest paths. We can negate weights to find longest,
    # or just adapt the relaxation step.
    
    U = np.zeros(n_types)
    
    # Relaxation: U(t) = max(U(t), U(t') + w(t', t))
    # We iterate until convergence.
    
    for _ in range(n_types):
        updated = False
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                # Weight from t' to t
                w = 0.0
                for g in range(2):
                    w += (types[t][g] - types[t_prime][g]) * x[t_prime, g]
                
                new_val = U[t_prime] + w
                if new_val > U[t] + 1e-9:
                    U[t] = new_val
                    updated = True
        if not updated:
            break
    
    # Check for positive cycles (infeasible IC)
    # If U changes after n_types iterations, there's a cycle.
    # But for valid IC, there should be no positive cycles in the utility difference graph.
    # Actually, if there is a positive cycle, IC is impossible.
    
    # Payments
    p = np.zeros(n_types)
    for t in range(n_types):
        v_dot_x = np.dot(x[t], types[t])
        p[t] = v_dot_x - U[t]
    
    # Check if payments are non-negative (IR is U(t) >= 0, which is satisfied by construction if we start with 0 and only increase)
    # Wait, U(t) is the minimal utility. If U(t) < 0, IR is violated?
    # No, IR is U(t) >= 0. Our construction ensures U(t) >= 0 because we start with 0 and take max.
    # However, we must ensure p(t) >= 0.
    # p(t) = v(t).x(t) - U(t).
    # If p(t) < 0, it means U(t) > v(t).x(t), which violates IR (since utility = v.x - p = U, and IR requires utility >= 0, which is U >= 0. But also p >= 0 is a constraint).
    # The constraint is p(t) >= 0 AND U(t) >= 0.
    # If p(t) < 0, the mechanism is invalid.
    
    valid = np.all(p >= -1e-9)
    
    # Clip small negatives
    p = np.clip(p, 0, None)
    
    return p, valid


def _check_ir(x, p, types):
    """Check Individual Rationality."""
    for t in range(len(types)):
        utility = np.dot(x[t], types[t]) - p[t]
        if utility < -1e-9:
            return False
    return True


# Removed _verify_constraints and _fix_violations as they are no longer used by the heuristic solver
# The heuristic solver uses _compute_min_payments and _check_ir


def _fallback_result():
    """Return a valid fallback result (e.g., no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
