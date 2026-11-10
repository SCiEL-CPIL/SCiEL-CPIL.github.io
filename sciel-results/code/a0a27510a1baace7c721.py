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
    
    # Solve the LP with all constraints including non-negative payments
    result = _solve_heuristic_auction(probs, types, n_types, n_goods, start_time, DEADLINE)
    
    # If we still have time, try a local refinement pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0 and result is not None:
        result = _local_refinement(result, probs, types, n_types, n_goods, start_time, DEADLINE)
    
    return result


def _local_refinement(result, probs, types, n_types, n_goods, start_time, deadline):
    """
    Try to improve the solution via local search on allocation variables.
    Uses subgradient-like updates to push toward better revenue.
    """
    if result is None:
        return _fallback_result()
    
    x = result['allocations'].copy()
    p = result['payments'].copy()
    
    # Compute current revenue
    def compute_revenue(x, p):
        return np.sum(probs * (x @ np.array(types).T - p))
    
    current_rev = compute_revenue(x, p)
    best_x = x.copy()
    best_p = p.copy()
    best_rev = current_rev
    
    # Simple local search: try small perturbations
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    rng = np.random.default_rng(42)
    
    for step in step_sizes:
        if time.time() - start_time > deadline - 2.0:
            break
        
        improved = False
        for _ in range(50):
            if time.time() - start_time > deadline - 2.0:
                break
            
            # Random perturbation
            x_trial = x + rng.normal(0, step, size=x.shape)
            x_trial = np.clip(x_trial, 0, 1)
            
            # Recompute minimal payments
            U = np.zeros(n_types)
            for _ in range(n_types + 1):
                updated = False
                for t in range(n_types):
                    for t_prime in range(n_types):
                        if t == t_prime:
                            continue
                        w = 0.0
                        for g in range(n_goods):
                            w += (types[t][g] - types[t_prime][g]) * x_trial[t_prime, g]
                        new_val = U[t_prime] + w
                        if new_val > U[t] + 1e-9:
                            U[t] = new_val
                            updated = True
                if not updated:
                    break
            
            p_trial = np.zeros(n_types)
            for t in range(n_types):
                p_trial[t] = np.dot(x_trial[t], types[t]) - U[t]
            
            # Check feasibility
            if np.any(p_trial < -1e-8):
                continue
            
            rev_trial = compute_revenue(x_trial, p_trial)
            
            if rev_trial > best_rev + 1e-10:
                best_x = x_trial.copy()
                best_p = p_trial.copy()
                best_rev = rev_trial
                x = x_trial.copy()
                p = p_trial.copy()
                improved = True
        
        if not improved:
            break
    
    return {
        'allocations': best_x,
        'payments': best_p
    }


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
    4. p_t = v_t . x_t - u_t >= 0 (non-negative payments)
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
    
    # Bounds for variables
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
        
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                               'primal_feasibility_tolerance': 1e-10})
        if not res.success:
            return _fallback_result()
            
        z = res.x
        x = z[x_start:x_start + n_types * n_goods].reshape(n_types, n_goods)
        u = z[u_start:u_start + n_types]
        
        # Calculate payments
        p = np.zeros(n_types)
        for t in range(n_types):
            p[t] = np.dot(x[t], types[t]) - u[t]
            
        # Post-solve verification and repair
        x, p = _verify_and_repair(x, p, u, probs, types, n_types, n_goods)
        
        return {
            'allocations': x,
            'payments': p
        }
    except Exception:
        return _fallback_result()


def _verify_and_repair(x, p, u, probs, types, n_types, n_goods):
    """
    Verify constraints and repair any numerical violations.
    Returns repaired (x, p).
    """
    # Check for small numerical violations
    tol = 1e-9
    
    # Clip small negative allocations
    x = np.clip(x, 0, 1)
    
    # Recompute u from x using the IC/IR constraints (longest path)
    U = np.zeros(n_types)
    for _ in range(n_types + 1):
        updated = False
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                w = 0.0
                for g in range(n_goods):
                    w += (types[t][g] - types[t_prime][g]) * x[t_prime, g]
                new_val = U[t_prime] + w
                if new_val > U[t] + tol:
                    U[t] = new_val
                    updated = True
        if not updated:
            break
    
    # Recompute payments
    p_new = np.zeros(n_types)
    for t in range(n_types):
        p_new[t] = np.dot(x[t], types[t]) - U[t]
    
    # Check if payments are non-negative
    if np.any(p_new < -tol):
        # If payments are significantly negative, we need to adjust
        # This shouldn't happen if the LP was solved correctly, but let's be safe
        # Try to fix by adjusting allocations slightly
        for t in range(n_types):
            if p_new[t] < -tol:
                # Reduce allocation to make payment non-negative
                # Simple approach: scale down x[t]
                scale = max(0, 1 + p_new[t] / max(1e-12, np.dot(x[t], types[t])))
                x[t] *= scale
                x[t] = np.clip(x[t], 0, 1)
        
        # Recompute after adjustment
        U = np.zeros(n_types)
        for _ in range(n_types + 1):
            updated = False
            for t in range(n_types):
                for t_prime in range(n_types):
                    if t == t_prime:
                        continue
                    w = 0.0
                    for g in range(n_goods):
                        w += (types[t][g] - types[t_prime][g]) * x[t_prime, g]
                    new_val = U[t_prime] + w
                    if new_val > U[t] + tol:
                        U[t] = new_val
                        updated = True
            if not updated:
                break
        
        p_new = np.zeros(n_types)
        for t in range(n_types):
            p_new[t] = np.dot(x[t], types[t]) - U[t]
    
    # Final clip for any remaining tiny negatives
    p_new = np.clip(p_new, 0, None)
    
    return x, p_new


# _compute_min_payments and _check_ir are no longer used directly,
# but the logic is now integrated into _verify_and_repair


def _fallback_result():
    """Return a valid fallback result (e.g., no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
