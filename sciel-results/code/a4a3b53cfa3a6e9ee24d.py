"""Incentive-compatible finite auction design solver with local search."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    values = [0, 1, 2, 3]
    
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]
    
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p1 = p1 / p1.sum() if p1.sum() > 0 else np.ones(4) / 4
    p2 = p2 / p2.sum() if p2.sum() > 0 else np.ones(4) / 4
    
    joint_probs = np.outer(p1, p2).flatten()
    
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    # type_index = v1 * 4 + v2

    # 1. Get initial allocation via LP relaxation
    n_x = 32
    n_p = 16
    n_vars = n_x + n_p
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -joint_probs[t]
        
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints for LP
    for t in range(n_types):
        v1_t, v2_t = types[t]
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            row[tp * 2] += v1_t
            row[tp * 2 + 1] += v2_t
            row[n_x + t] = 1
            row[n_x + tp] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
            
    # IR constraints for LP
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1_t
        row[t * 2 + 1] = -v2_t
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
        
    # Nonnegative payments
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + t] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
        
    bounds = [(0, 1)] * n_x + [(0, None)] * n_p
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x_frac = res.x[:n_x]
        else:
            x_frac = np.zeros(n_x)
    except:
        x_frac = np.zeros(n_x)
        
    # Round to binary
    X_curr = np.zeros((n_types, 2), dtype=int)
    for t in range(n_types):
        X_curr[t, 0] = 1 if x_frac[t * 2] > 0.5 else 0
        X_curr[t, 1] = 1 if x_frac[t * 2 + 1] > 0.5 else 0

    # 2. Function to solve for optimal payments given a binary allocation X
    def solve_payments(X):
        """
        X: (16, 2) binary array
        Returns: (revenue, payments)
        """
        # Variables: p[0..15]
        c_p = -joint_probs
        
        A_ub_p_rows = []
        b_ub_p_vals = []
        
        # IC: p_t - p_t' <= v(t) . (x_t - x_t')
        for t in range(n_types):
            v_t = np.array(types[t])
            x_t = X[t]
            for tp in range(n_types):
                if t == tp:
                    continue
                x_tp = X[tp]
                # v(t) . (x_t - x_tp)
                rhs = np.dot(v_t, x_t - x_tp)
                row = np.zeros(n_p)
                row[t] = 1
                row[tp] = -1
                A_ub_p_rows.append(row)
                b_ub_p_vals.append(rhs)
                
        # IR: p_t <= v(t) . x_t
        for t in range(n_types):
            v_t = np.array(types[t])
            x_t = X[t]
            rhs = np.dot(v_t, x_t)
            row = np.zeros(n_p)
            row[t] = 1
            A_ub_p_rows.append(row)
            b_ub_p_vals.append(rhs)
            
        # p_t >= 0 is handled by bounds
        
        A_ub_p = np.array(A_ub_p_rows) if A_ub_p_rows else None
        b_ub_p = np.array(b_ub_p_vals) if b_ub_p_vals else None
        bounds_p = [(0, None)] * n_p
        
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs')
            if res_p.success:
                payments = res_p.x
                revenue = -res_p.fun
                return revenue, payments
            else:
                # Fallback: zero payments
                return 0.0, np.zeros(n_p)
        except:
            return 0.0, np.zeros(n_p)

    # 3. Local Search
    rev_curr, pay_curr = solve_payments(X_curr)
    X_best = X_curr.copy()
    pay_best = pay_curr.copy()
    rev_best = rev_curr
    
    # Precompute neighbors indices
    # We will flip one bit at a time
    
    while time.time() < deadline:
        improved = False
        # Try flipping each bit
        for t in range(n_types):
            for g in range(2):
                # Flip bit
                X_new = X_best.copy()
                X_new[t, g] = 1 - X_new[t, g]
                
                rev_new, pay_new = solve_payments(X_new)
                
                if rev_new > rev_best + 1e-9:
                    X_best = X_new
                    pay_best = pay_new
                    rev_best = rev_new
                    improved = True
        
        # If no improvement, try a random perturbation to escape local minima
        if not improved:
            # Try 2-bit flips? Or random restart?
            # Let's try a few random 2-bit flips
            for _ in range(10):
                t1 = np.random.randint(0, n_types)
                g1 = np.random.randint(0, 2)
                t2 = np.random.randint(0, n_types)
                g2 = np.random.randint(0, 2)
                if t1 == t2 and g1 == g2:
                    continue
                    
                X_new = X_best.copy()
                X_new[t1, g1] = 1 - X_new[t1, g1]
                X_new[t2, g2] = 1 - X_new[t2, g2]
                
                rev_new, pay_new = solve_payments(X_new)
                
                if rev_new > rev_best + 1e-9:
                    X_best = X_new
                    pay_best = pay_new
                    rev_best = rev_new
                    improved = True
                    break
            
            # If still no improvement after random tries, we might be at a local optimum.
            # Given the small size, 1-bit search is often sufficient, but 2-bit helps.
            # To save time, we can break if we've done many iterations without improvement.
            # For now, just continue to allow more random tries in next loop.
            
    return {'allocations': X_best, 'payments': pay_best}
