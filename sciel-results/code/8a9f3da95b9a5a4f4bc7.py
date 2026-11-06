import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 70.0
    
    # 1. Setup Report Space
    # Reports are (v1, v2) where v1, v2 in {0, 1, 2, 3}
    # Index i = v1 * 4 + v2
    n_types = 16
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    
    # 2. Parse Distribution
    probs = _get_probabilities(distribution_spec, n_types)
    
    # 3. Parse Constraints
    # Default regret bound is very tight
    # The task requires regret <= 1e-8
    
    # 4. Formulate LP
    # Decision Variables: x[i, g] for i in 0..15, g in 0..1
    # Total variables: 32
    n_vars = n_types * 2
    
    # Objective: Maximize Expected Revenue
    # Revenue E[R] = sum_{v} P(v) * (v1 * x_{v,1} + v2 * x_{v,2} - payment_v)
    # Payment Identity (IC):
    # p(v) = v1 * x_{v,1} + v2 * x_{v,2} - \sum_{u1 < v1} x_{(u1, v2), 1} - \sum_{u2 < v2} x_{(v1, u2), 2}
    # 
    # Substituting p(v) into Revenue:
    # E[R] = sum_{v} P(v) * [ v1 * x_{v,1} + v2 * x_{v,2} - \sum_{u1 < v1} x_{(u1, v2), 1} - \sum_{u2 < v2} x_{(v1, u2), 2} ]
    #
    # We can rewrite this by collecting coefficients for each x_{j, g}.
    # Let c_j be the coefficient of x_{j, g} in the objective function (to be maximized).
    # linprog minimizes -c.
    
    obj_coeffs = np.zeros(n_vars)
    
    # Iterate over all types v to determine their contribution to the coefficients of other types x_j
    for v in range(n_types):
        v1, v2 = values[v]
        p_v = probs[v]
        
        # Term: + p_v * v1 * x_{v, 1}
        idx_v1 = v * 2 + 0
        obj_coeffs[idx_v1] += p_v * v1
        
        # Term: + p_v * v2 * x_{v, 2}
        idx_v2 = v * 2 + 1
        obj_coeffs[idx_v2] += p_v * v2
        
        # Term: - p_v * \sum_{u1 < v1} x_{(u1, v2), 1}
        for u1 in range(int(v1)):
            idx_u1_v2 = u1 * 4 + int(v2)
            idx_alloc = idx_u1_v2 * 2 + 0
            obj_coeffs[idx_alloc] -= p_v
            
        # Term: - p_v * \sum_{u2 < v2} x_{(v1, u2), 2}
        for u2 in range(int(v2)):
            idx_v1_u2 = int(v1) * 4 + u2
            idx_alloc = idx_v1_u2 * 2 + 1
            obj_coeffs[idx_alloc] -= p_v
            
    c = -obj_coeffs
    
    # Constraints
    # 1. Bounds: 0 <= x <= 1
    bounds = [(0.0, 1.0)] * n_vars
    
    # 2. Monotonicity (for IC)
    # x_{v, g} must be non-decreasing in v_g.
    # For good 1 (index 0): x_{(v1, v2), 1} <= x_{(v1+1, v2), 1}
    # For good 2 (index 1): x_{(v1, v2), 2} <= x_{(v1, v2+1), 2}
    
    A_ub_rows = []
    b_ub_rows = []
    
    # Good 1 Monotonicity
    for v2 in range(4):
        for v1 in range(3):
            i_curr = v1 * 4 + v2
            i_next = (v1 + 1) * 4 + v2
            
            row = np.zeros(n_vars)
            row[i_curr * 2 + 0] = 1.0
            row[i_next * 2 + 0] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    # Good 2 Monotonicity
    for v1 in range(4):
        for v2 in range(3):
            i_curr = v1 * 4 + v2
            i_next = v1 * 4 + (v2 + 1)
            
            row = np.zeros(n_vars)
            row[i_curr * 2 + 1] = 1.0
            row[i_next * 2 + 1] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            x_flat = res.x
            x_alloc = x_flat.reshape(n_types, 2)
            x_alloc = np.clip(x_alloc, 0.0, 1.0)
            
            # Calculate Payments
            payments = np.zeros(n_types)
            for v in range(n_types):
                v1, v2 = values[v]
                p_v = v1 * x_alloc[v, 0] + v2 * x_alloc[v, 1]
                
                for u1 in range(int(v1)):
                    idx_u1_v2 = u1 * 4 + int(v2)
                    p_v -= x_alloc[idx_u1_v2, 0]
                    
                for u2 in range(int(v2)):
                    idx_v1_u2 = int(v1) * 4 + u2
                    p_v -= x_alloc[idx_v1_u2, 1]
                    
                payments[v] = p_v
            
            # Post-processing: Ensure IR and Non-negative payments
            # The LP formulation with monotonicity and bounds does not explicitly constrain p(v) >= 0 or IR.
            # However, for the specific revenue maximization problem, the optimal solution typically satisfies these.
            # If not, we must adjust.
            
            # Check IR: v1*x1 + v2*x2 - p >= 0
            # Check p >= 0
            
            if _check_feasibility(x_alloc, payments, values, n_types):
                 return {
                     'allocations': x_alloc,
                     'payments': payments
                 }
            else:
                 # If the LP solution is infeasible regarding IR/Non-negativity (which shouldn't happen for optimal revenue maximization 
                 # unless we have numerical issues or the problem allows negative payments which the task forbids),
                 # we fall back to a safe heuristic or repair.
                 # Given the "regret <= 1e-8" constraint, we need strict IC.
                 
                 # A common issue is p(v) < 0. We can set p(v) = max(0, p(v)) but that breaks IC.
                 # Instead, we can solve a secondary LP to minimize "violation" or just return the best feasible point.
                 # However, for this specific problem structure (2 goods, discrete), the LP solution is usually valid.
                 # Let's try a simple repair: Clip payments to 0, then check if IC holds. If not, shrink allocations.
                 
                 # Actually, a safer fallback is the "All-or-Nothing" or "Zero Payment" mechanism if the complex one fails validation.
                 # But let's try to repair the LP solution first.
                 
                 # Repair: Ensure p >= 0.
                 # If p < 0, it means the bidder is getting paid to participate.
                 # We can adjust the "base" payment or shrink allocations.
                 
                 # For now, return the best effort if it passes a looser check, otherwise fallback.
                 if _check_feasibility(x_alloc, np.clip(payments, 0, None), values, n_types):
                      return {
                          'allocations': x_alloc,
                          'payments': np.clip(payments, 0, None)
                      }
                 
        # Fallback if LP fails or validation fails
        return _fallback_mechanism()
        
    except Exception as e:
        return _fallback_mechanism()

def _get_probabilities(spec, n_types):
    """Parses various distribution formats into a probability vector."""
    if spec is None:
        return np.ones(n_types) / n_types
        
    if isinstance(spec, dict):
        probs = np.zeros(n_types)
        for key, val in spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = int(key[0]), int(key[1])
                if 0 <= v1 < 4 and 0 <= v2 < 4:
                    idx = v1 * 4 + v2
                    probs[idx] = float(val)
            elif isinstance(key, int):
                if 0 <= key < n_types:
                    probs[key] = float(val)
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.ones(n_types) / n_types
        
    if isinstance(spec, (list, tuple, np.ndarray)):
        arr = np.array(spec, dtype=float)
        if arr.shape == (n_types,):
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_types) / n_types
        # Handle 2D array of [v1, v2, p]
        if arr.ndim == 2 and arr.shape[1] >= 3:
            probs = np.zeros(n_types)
            for row in arr:
                v1, v2, p = int(row[0]), int(row[1]), float(row[2])
                if 0 <= v1 < 4 and 0 <= v2 < 4:
                    idx = v1 * 4 + v2
                    probs[idx] += p
            total = probs.sum()
            if total > 0:
                return probs / total
            return np.ones(n_types) / n_types
            
    return np.ones(n_types) / n_types

def _check_feasibility(allocs, pays, values, n_types):
    """Checks IR, IC, and Non-negativity."""
    eps = 1e-8
    
    # Check bounds
    if np.any(allocs < -eps) or np.any(allocs > 1 + eps):
        return False
    if np.any(pays < -eps):
        return False
        
    # Check IR and IC
    for j in range(n_types):
        v1_j, v2_j = values[j]
        u_true = v1_j * allocs[j, 0] + v2_j * allocs[j, 1] - pays[j]
        
        if u_true < -eps:
            return False
            
        for i in range(n_types):
            if i == j: continue
            u_mis = v1_j * allocs[i, 0] + v2_j * allocs[i, 1] - pays[i]
            if u_true - u_mis < -eps:
                return False
                
    return True

def _fallback_mechanism():
    """
    Returns a simple, valid mechanism.
    Allocates nothing, charges nothing.
    Revenue 0, but valid.
    """
    n_types = 16
    allocs = np.zeros((n_types, 2))
    pays = np.zeros(n_types)
    return {
        'allocations': allocs,
        'payments': pays
    }
