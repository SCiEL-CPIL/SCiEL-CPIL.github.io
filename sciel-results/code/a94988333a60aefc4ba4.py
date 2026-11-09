"""Incentive-compatible finite auction design via direct LP."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite-support auction mechanism design problem as a linear program.
    
    Returns a dict with 'allocations' (16,2) and 'payments' (16,).
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the problem structure
    # distribution_spec: describes the distributions
    # support: the type space
    # constraints: feasibility constraints
    
    # We need to figure out the structure from the inputs
    # Based on the task: 1 bidder, 2 goods, values in {0,1,2,3}, 3 distributions
    # Total reports: 4*4 = 16 (one per value combination for the single bidder)
    
    # Let's extract information from the inputs
    # The support should tell us the value space
    # distribution_spec tells us the distributions
    
    # Standard setup: 16 types (v1, v2) where v1, v2 in {0,1,2,3}
    n_types = 16
    n_goods = 2
    
    # Extract probabilities from distribution_spec
    # Try to parse the distribution_spec to get the probability vector
    try:
        if isinstance(distribution_spec, dict):
            # Could have 'probs' or similar key
            if 'probs' in distribution_spec:
                probs = np.array(distribution_spec['probs'], dtype=float)
            elif 'distribution' in distribution_spec:
                probs = np.array(distribution_spec['distribution'], dtype=float)
            elif 'weights' in distribution_spec:
                probs = np.array(distribution_spec['weights'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                # Try to find any list-like value
                for key in distribution_spec:
                    val = distribution_spec[key]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        arr = np.array(val, dtype=float)
                        if len(arr) == n_types:
                            probs = arr
                            break
                        elif len(arr) > 0:
                            probs = arr
                            break
                else:
                    probs = np.ones(n_types) / n_types
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
            if len(probs) != n_types:
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    except Exception:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = np.asarray(probs, dtype=float)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure we have 16 probabilities
    if len(probs) < n_types:
        # Pad or adjust
        new_probs = np.ones(n_types) / n_types
        for i in range(min(len(probs), n_types)):
            new_probs[i] = probs[i]
        s = new_probs.sum()
        if s > 0:
            new_probs /= s
        probs = new_probs
    elif len(probs) > n_types:
        probs = probs[:n_types]
        s = probs.sum()
        if s > 0:
            probs /= s
    
    # Extract values from support
    try:
        if isinstance(support, dict):
            if 'values' in support:
                vals = support['values']
            elif 'types' in support:
                vals = support['types']
            elif 'v' in support:
                vals = support['v']
            else:
                for key in support:
                    val = support[key]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        vals = val
                        break
                else:
                    vals = None
        elif isinstance(support, (list, tuple, np.ndarray)):
            vals = support
        else:
            vals = None
        
        if vals is not None:
            vals = np.array(vals)
            if vals.ndim == 2 and vals.shape == (n_types, n_goods):
                values = vals
            elif vals.ndim == 1 and len(vals) == n_types * n_goods:
                values = vals.reshape(n_types, n_goods)
            elif vals.ndim == 1 and len(vals) == n_types:
                # Maybe single value per type? Duplicate for 2 goods
                values = np.column_stack([vals, vals])
            else:
                values = None
        else:
            values = None
    except Exception:
        values = None
    
    if values is None:
        # Default: enumerate all combinations of {0,1,2,3}
        v_range = [0, 1, 2, 3]
        values = np.array([[v1, v2] for v1 in v_range for v2 in v_range], dtype=float)
    
    values = np.asarray(values, dtype=float)
    
    # Now set up the LP
    # Variables: x[i,g] = allocation probability of good g to bidder at type i (16*2 = 32 vars)
    #            p[i] = payment at type i (16 vars)
    # Total: 48 variables
    # 
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    #           In linprog form: minimize -sum_i probs[i] * p[i]
    #
    # Constraints:
    # 1. Feasibility: 0 <= x[i,g] <= 1 for all i,g
    #    -x[i,g] <= 0 (lower bound handled by bounds)
    #    x[i,g] <= 1 (upper bound handled by bounds)
    #
    # 2. Individual Rationality: p[i] <= sum_g x[i,g] * v[i,g] for all i
    #    => p[i] - sum_g x[i,g]*v[i,g] <= 0
    #
    # 3. Incentive Compatibility: For all i, j:
    #    p[i] - p[j] <= sum_g (x[i,g] - x[j,g]) * v[i,g]
    #    => p[i] - p[j] - sum_g x[i,g]*v[i,g] + sum_g x[j,g]*v[i,g] <= 0
    #
    #    This is: u(i, report_j) >= u(i, report_i) where
    #    u(i, report) = sum_g x[report,g]*v[i,g] - p[report]
    #    So: sum_g x[j,g]*v[i,g] - p[j] >= sum_g x[i,g]*v[i,g] - p[i]
    #    => p[i] - p[j] <= sum_g (x[i,g] - x[j,g]) * v[i,g]
    #    => p[i] - p[j] - sum_g x[i,g]*v[i,g] + sum_g x[j,g]*v[i,g] <= 0
    
    n_vars = n_types * n_goods + n_types  # 32 + 16 = 48
    x_offset = 0
    p_offset = n_types * n_goods  # 32
    
    # Objective: minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_offset + i] = -probs[i]
    
    # A_ub and b_ub for inequality constraints
    A_rows = []
    b_vals = []
    
    # IR constraints: p[i] - sum_g x[i,g]*v[i,g] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[p_offset + i] = 1.0
        for g in range(n_goods):
            row[x_offset + i * n_goods + g] = -values[i, g]
        A_rows.append(row)
        b_vals.append(0.0)
    
    # IC constraints: for all pairs (i, j), i != j
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # p[i] - p[j] - sum_g x[i,g]*v[i,g] + sum_g x[j,g]*v[i,g] <= 0
            row[p_offset + i] = 1.0
            row[p_offset + j] = -1.0
            for g in range(n_goods):
                row[x_offset + i * n_goods + g] = -values[i, g]
                row[x_offset + j * n_goods + g] = values[i, g]
            A_rows.append(row)
            b_vals.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((1, n_vars))
    b_ub = np.array(b_vals) if b_vals else np.zeros(1)
    
    # Bounds: 0 <= x[i,g] <= 1, p[i] >= 0
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Return a simple feasible solution
        allocs = np.zeros((n_types, n_goods))
        pays = np.zeros(n_types)
        return {'allocations': allocs, 'payments': pays}
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status == 0:  # Optimal
            x_opt = result.x
            allocations = x_opt[:n_types * n_goods].reshape(n_types, n_goods)
            payments = x_opt[n_types * n_goods:]
            
            # Ensure numerical precision
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)
            
            # Verify and fix any tiny violations
            # Check IC
            for i in range(n_types):
                for j in range(n_types):
                    if i == j:
                        continue
                    lhs = payments[i] - payments[j]
                    rhs = np.sum((allocations[i] - allocations[j]) * values[i])
                    if lhs - rhs > 1e-8:
                        # Violation - adjust payment
                        payments[i] = min(payments[i], payments[j] + rhs)
            
            # Check IR
            for i in range(n_types):
                max_pay = np.sum(allocations[i] * values[i])
                if payments[i] > max_pay + 1e-8:
                    payments[i] = max(0.0, max_pay)
            
            # Re-clip
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)
            
        else:
            # Fallback: use a simple feasible mechanism
            # Allocate good g to bidder with probability min(1, v[g]/max_val) or similar
            allocations = np.zeros((n_types, n_goods))
            for i in range(n_types):
                for g in range(n_goods):
                    if values[i, g] > 0:
                        allocations[i, g] = 1.0
            # Simple payment: p[i] = sum_g x[i,g] * v[i,g] (IR binding)
            payments = np.sum(allocations * values, axis=1)
            payments = np.maximum(payments, 0.0)
            
            # Actually let's try a simpler fallback: uniform allocation
            allocations = np.ones((n_types, n_goods)) * 0.5
            payments = np.sum(allocations * values, axis=1)
            payments = np.maximum(payments, 0.0)
            
    except Exception:
        # Fallback
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 1.0:
        pass  # Already close to deadline, just return
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
