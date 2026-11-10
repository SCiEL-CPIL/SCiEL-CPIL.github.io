import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an optimal incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Distribution specification
        support: Support specification
        constraints: Constraints dictionary
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    # Define the 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    V = np.array(types, dtype=np.float64)

    # Parse distribution probabilities
    probs = _parse_distribution(distribution_spec, n_types)

    if time.time() - start_time > DEADLINE:
        return _fallback_result()

    # Phase 1: Solve joint LP for allocations and utilities
    # This finds the global optimum for (x, u)
    result = _solve_joint_lp(probs, types, n_types, n_goods, start_time, DEADLINE)
    
    if result is None:
        # Fallback to a simple heuristic if LP fails
        x = np.ones((n_types, n_goods)) * 0.5
        p = np.zeros(n_types)
    else:
        x = result['allocations'].copy()
        p = result['payments'].copy()

    # Phase 2: Solve Payment LP with fixed allocations from Phase 1
    # This ensures payments are optimal for the given allocations
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10.0:
        p_new = _solve_payment_lp(x, probs, types, n_types, n_goods, start_time, DEADLINE)
        if p_new is not None:
            # Check if this payment vector is feasible
            if _check_feasibility(x, p_new, types, n_types, n_goods, tol=1e-9):
                p = p_new
            else:
                # If not feasible, keep old p or try to repair
                pass

    # Phase 3: Iterative Refinement (Alternating Optimization)
    # Try to improve x given p, then p given x
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        x, p = _refine_alternating(x, p, probs, types, n_types, n_goods, start_time, DEADLINE)

    # Phase 4: Final Verification and Repair
    x, p = _verify_and_repair(x, p, probs, types, n_types, n_goods)

    return {
        'allocations': x,
        'payments': p
    }

def _parse_distribution(distribution_spec, n_types):
    """Parse distribution specification into probability vector."""
    probs = np.zeros(n_types)

    if distribution_spec is None:
        return np.ones(n_types) / n_types

    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            d = distribution_spec['distribution']
            if isinstance(d, str) and d == 'uniform':
                return np.ones(n_types) / n_types
            elif isinstance(d, str) and d == 'geometric':
                types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
                base = 0.7
                probs = np.array([base ** types[i][0] * base ** types[i][1] for i in range(n_types)])
            else:
                probs = np.array(d, dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_types,):
            probs = arr
        elif arr.shape == (4, 4):
            probs = arr.flatten()
        elif arr.size == n_types:
            probs = arr.flatten()
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types

    probs = np.asarray(probs, dtype=np.float64).flatten()
    if probs.shape[0] != n_types:
        probs = np.ones(n_types) / n_types

    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total > 1e-15:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types

    return probs

def _solve_joint_lp(probs, types, n_types, n_goods, start_time, deadline):
    """
    Solve the joint LP for allocations x and utilities u.
    Maximize expected revenue = sum_t probs[t] * (dot(v_t, x_t) - u_t)
    Subject to:
      IR: u_t >= 0
      IC: u_t >= u_{t'} + dot(v_t - v_{t'}, x_{t'})
      Non-neg payments: p_t = dot(v_t, x_t) - u_t >= 0  => u_t <= dot(v_t, x_t)
      Bounds: 0 <= x_{t,g} <= 1
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

    # IR: -u_t <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # IC: u_t - u_{t'} - dot(v_t - v_{t'}, x_{t'}) >= 0
    # => -u_t + u_{t'} + dot(v_t - v_{t'}, x_{t'}) <= 0
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

    # Non-negative payments: u_t - dot(v_t, x_t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = 1.0
        for g in range(n_goods):
            row[x_start + t * n_goods + g] = -types[t][g]
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    bounds = [(0, 1)] * (n_types * n_goods) + [(0, None)] * n_types

    time_left = deadline - (time.time() - start_time) - 5.0
    if time_left < 1.0:
        return None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12,
                               'time_limit': max(1.0, time_left)})
        if not res.success or res.x is None:
            return None

        z = res.x
        x = z[x_start:x_start + n_types * n_goods].reshape(n_types, n_goods)
        u = z[u_start:u_start + n_types]

        x = np.clip(x, 0, 1)
        p = np.zeros(n_types)
        for t in range(n_types):
            p[t] = np.dot(x[t], types[t]) - u[t]
        p = np.maximum(p, 0.0)

        return {'allocations': x, 'payments': p}
    except Exception:
        return None

def _solve_payment_lp(x, probs, types, n_types, n_goods, start_time, deadline):
    """
    Given fixed allocations x, solve for maximum payments p.
    Maximize sum_t probs[t] * p_t
    Subject to:
      IC: p[t] - p[t'] <= v_t . (x_t - x_{t'})
      IR: p[t] <= v_t . x_t
      p[t] >= 0
    """
    n = n_types
    c = np.zeros(n)
    for t in range(n):
        c[t] = -probs[t]

    A_ub = []
    b_ub = []

    # IC constraints
    for t in range(n):
        for t_prime in range(n):
            if t == t_prime:
                continue
            row = np.zeros(n)
            row[t] = 1.0
            row[t_prime] = -1.0
            rhs = np.dot(types[t], x[t] - x[t_prime])
            A_ub.append(row)
            b_ub.append(rhs)

    # IR constraints
    for t in range(n):
        row = np.zeros(n)
        row[t] = 1.0
        A_ub.append(row)
        b_ub.append(np.dot(types[t], x[t]))

    # Non-negative payments
    for t in range(n):
        row = np.zeros(n)
        row[t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n

    time_left = deadline - (time.time() - start_time) - 8.0
    if time_left < 1.0:
        return None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-14,
                               'primal_feasibility_tolerance': 1e-14,
                               'time_limit': max(1.0, time_left)})
        if not res.success or res.x is None:
            return None

        p = np.clip(res.x, 0, None)
        return p
    except Exception:
        return None

def _solve_allocation_lp(p, probs, types, n_types, n_goods, start_time, deadline):
    """
    Given fixed payments p, solve for optimal allocations x.
    Maximize sum_t probs[t] * dot(v_t, x_t) - sum_t probs[t] * p_t
    Subject to:
      IC: p[t] - p[t'] <= v_t . (x_t - x_{t'})
      => -v_t . (x_t - x_{t'}) <= p[t] - p[t']
      => -sum_g v_{t,g} x_{t,g} + sum_g v_{t,g} x_{t',g} <= p[t] - p[t']
      IR: p[t] <= v_t . x_t  =>  -v_t . x_t <= -p[t]
      0 <= x_{t,g} <= 1
    """
    n_vars = n_types * n_goods
    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(n_goods):
            c[t * n_goods + g] = -probs[t] * types[t][g]
    
    # Constant term -sum probs[t]*p[t] is irrelevant for argmax
    
    A_ub = []
    b_ub = []
    
    # IC constraints: -v_t . x_t + v_t . x_{t'} <= p[t] - p[t']
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            # -v_t . x_t
            for g in range(n_goods):
                row[t * n_goods + g] -= types[t][g]
            # +v_t . x_{t'}
            for g in range(n_goods):
                row[t_prime * n_goods + g] += types[t][g]
            
            A_ub.append(row)
            b_ub.append(p[t] - p[t_prime])
            
    # IR constraints: -v_t . x_t <= -p[t]
    for t in range(n_types):
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[t * n_goods + g] -= types[t][g]
        A_ub.append(row)
        b_ub.append(-p[t])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, 1)] * n_vars
    
    time_left = deadline - (time.time() - start_time) - 5.0
    if time_left < 1.0:
        return None
        
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12,
                               'time_limit': max(1.0, time_left)})
        if not res.success or res.x is None:
            return None
            
        x = res.x.reshape(n_types, n_goods)
        x = np.clip(x, 0, 1)
        return x
    except Exception:
        return None

def _refine_alternating(x, p, probs, types, n_types, n_goods, start_time, deadline):
    """
    Alternate between optimizing x given p and p given x.
    """
    best_revenue = -np.inf
    best_x = x.copy()
    best_p = p.copy()
    
    for _ in range(10): # Max 10 iterations
        if time.time() - start_time > deadline - 2.0:
            break
            
        # 1. Optimize p given x
        p_new = _solve_payment_lp(x, probs, types, n_types, n_goods, start_time, deadline)
        if p_new is not None:
            p = p_new
        else:
            break
            
        # Check feasibility
        if not _check_feasibility(x, p, types, n_types, n_goods, tol=1e-9):
            break
            
        # Calculate revenue
        rev = np.sum(probs * p)
        if rev > best_revenue:
            best_revenue = rev
            best_x = x.copy()
            best_p = p.copy()
            
        # 2. Optimize x given p
        x_new = _solve_allocation_lp(p, probs, types, n_types, n_goods, start_time, deadline)
        if x_new is not None:
            # Check if x_new is feasible with current p
            if _check_feasibility(x_new, p, types, n_types, n_goods, tol=1e-9):
                x = x_new
            else:
                break
        else:
            break
            
    return best_x, best_p

def _check_feasibility(x, p, types, n_types, n_goods, tol=1e-10):
    """Check if (x, p) satisfies IR, IC, and non-negative payments."""
    V = np.array(types, dtype=np.float64)

    if np.any(p < -tol):
        return False

    for t in range(n_types):
        if p[t] > np.dot(V[t], x[t]) + tol:
            return False

    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            lhs = p[t] - p[t_prime]
            rhs = np.dot(V[t], x[t] - x[t_prime])
            if lhs > rhs + tol:
                return False

    return True

def _verify_and_repair(x, p, probs, types, n_types, n_goods):
    """
    Verify constraints and repair any numerical violations.
    """
    tol = 1e-9
    V = np.array(types, dtype=np.float64)

    x = np.clip(x, 0, 1)
    p = np.clip(p, 0, None)

    # If infeasible, try to reduce payments to satisfy IC/IR
    if not _check_feasibility(x, p, types, n_types, n_goods, tol=tol):
        # Simple repair: reduce p to satisfy IR
        for t in range(n_types):
            max_p = np.dot(V[t], x[t])
            if p[t] > max_p:
                p[t] = max_p
        
        # If still infeasible due to IC, we might need to reduce x
        # For now, just ensure IR and non-negativity
        if not _check_feasibility(x, p, types, n_types, n_goods, tol=tol):
            # Fallback to zero
            x = np.zeros((n_types, n_goods))
            p = np.zeros(n_types)

    return x, p

def _fallback_result():
    """Return a valid fallback result (e.g., no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
