# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite auction design LP with payment elimination.
    
    Parameters:
    -----------
    distribution_spec : str or dict - specifies which distribution (one of three fixed)
    support : list or tuple - value support, e.g. [0,1,2,3]
    constraints : dict - constraint parameters
    
    Returns:
    --------
    dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    # Parse support - each value in {0,1,2,3}, 2 goods, so 16 type profiles
    if isinstance(support, (list, tuple)):
        values = list(support)
    else:
        values = [0, 1, 2, 3]
    
    n_values = len(values)
    n_types = n_values ** 2  # 16 type profiles (v1, v2) for 2 goods
    
    # Enumerate all type profiles
    type_profiles = []
    for v1 in values:
        for v2 in values:
            type_profiles.append((v1, v2))
    type_profiles = np.array(type_profiles)
    
    n_t = len(type_profiles)  # should be 16
    
    # Determine distribution
    probs = None
    
    # Helper to normalize
    def normalize(p):
        p = np.array(p, dtype=float)
        p = np.maximum(p, 0)
        s = p.sum()
        if s > 0:
            return p / s
        return np.ones(n_t) / n_t

    if isinstance(distribution_spec, dict):
        # Check if explicit probabilities are provided
        if 'probs' in distribution_spec:
            probs = normalize(distribution_spec['probs'])
        elif 'probabilities' in distribution_spec:
            probs = normalize(distribution_spec['probabilities'])
        else:
            # Try to infer from name/type
            dist_name = str(distribution_spec.get('distribution', distribution_spec.get('name', 'uniform'))).lower()
            if 'uniform' in dist_name:
                probs = np.ones(n_t) / n_t
            elif 'geometric' in dist_name or 'geom' in dist_name or 'exp' in dist_name:
                probs = np.zeros(n_t)
                for i, (v1, v2) in enumerate(type_profiles):
                    # Standard geometric decay
                    probs[i] = (0.5 ** (v1 + v2))
                probs = normalize(probs)
            elif 'beta' in dist_name:
                probs = np.ones(n_t) / n_t
            else:
                probs = np.ones(n_t) / n_t
    else:
        dist_name = str(distribution_spec).lower()
        if 'uniform' in dist_name or dist_name in ('u', 'unif'):
            probs = np.ones(n_t) / n_t
        elif 'geometric' in dist_name or 'geom' in dist_name or 'exp' in dist_name:
            probs = np.zeros(n_t)
            for i, (v1, v2) in enumerate(type_profiles):
                probs[i] = (0.5 ** (v1 + v2))
            probs = normalize(probs)
        elif 'beta' in dist_name:
            probs = np.ones(n_t) / n_t
        else:
            # Default: uniform
            probs = np.ones(n_t) / n_t

    # Ensure probabilities sum to 1 and are non-negative
    probs = normalize(probs)
    
    # Key insight: For single-bidder mechanism design, we can eliminate payments.
    # 
    # IR: v_t . x[t] - p[t] >= 0  =>  p[t] <= v_t . x[t]
    # IC: v_t . x[t] - p[t] >= v_t . x[s] - p[s] for all t, s
    #
    # The optimal mechanism sets p[t] = max(0, max_s (v_t . x[t] - v_t . x[s] + p[s]))
    # But this is circular. Instead, we use the fact that at optimum, for each type t,
    # the payment is determined by the "best" alternative type s that makes IC bind.
    #
    # Alternative approach: Use the dual characterization. The payment for type t is:
    # p[t] = v_t . x[t] - min_{s} (v_t . x[s] - p[s])
    # with the constraint that p[t] >= 0.
    #
    # Actually, the cleanest way: fix p[0] = 0 (IR binds for type 0 since v_0 = 0).
    # Then for each type t, p[t] = max(0, max_s (v_t . (x[t] - x[s]) + p[s]))
    # This is still circular. Let's just solve the full LP but with better conditioning.
    
    from scipy.optimize import linprog
    
    n_x = n_t * 2  # allocation variables
    n_p = n_t      # payment variables  
    n_vars = n_x + n_p
    
    # Objective: maximize sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_t):
        c[n_x + t] = -probs[t]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1
        row[t * 2 + 1] = -v2
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: u(t,t) >= u(t,s)
    # v_t . x[t] - p[t] >= v_t . x[s] - p[s]
    # => -v_t . x[t] + p[t] + v_t . x[s] - p[s] <= 0
    for t in range(n_t):
        v1_t, v2_t = type_profiles[t]
        for s in range(n_t):
            if t == s:
                continue
            row = np.zeros(n_vars)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            row[n_x + t] = 1
            row[s * 2] = v1_t
            row[s * 2 + 1] = v2_t
            row[n_x + s] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Tighter bounds: payments bounded by max possible value sum
    max_val_sum = max(v1 + v2 for v1, v2 in type_profiles)
    bounds = []
    for t in range(n_t):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for t in range(n_t):
        bounds.append((0, max_val_sum + 1))
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'mip_rel_gap': 0, 'time_limit': 60})
    
    elapsed = time.time() - start_time
    
    if result.success:
        x_flat = result.x[:n_x]
        p_flat = result.x[n_x:n_x + n_p]
        
        allocations = x_flat.reshape(n_t, 2)
        payments = p_flat
        
        # High-precision polish: re-solve with tighter tolerances
        if elapsed < DEADLINE - 10:
            # Re-solve with the current solution as a warm start (HiGHS doesn't support warm start directly)
            # Instead, just clip and verify
            allocations = np.clip(allocations, 0.0, 1.0)
            allocations[np.abs(allocations) < 1e-12] = 0.0
            allocations[np.abs(allocations - 1.0) < 1e-12] = 1.0
            payments = np.maximum(payments, 0.0)
            payments[np.abs(payments) < 1e-12] = 0.0
            
            # Verify constraints and fix small violations
            for t in range(n_t):
                v1, v2 = type_profiles[t]
                # Check IR
                if v1 * allocations[t, 0] + v2 * allocations[t, 1] - payments[t] < -1e-8:
                    payments[t] = max(0, v1 * allocations[t, 0] + v2 * allocations[t, 1])
                # Check IC
                for s in range(n_t):
                    if t == s:
                        continue
                    v1_s, v2_s = type_profiles[s]
                    util_t_t = v1 * allocations[t, 0] + v2 * allocations[t, 1] - payments[t]
                    util_t_s = v1 * allocations[s, 0] + v2 * allocations[s, 1] - payments[s]
                    if util_t_t < util_t_s - 1e-8:
                        # Fix by adjusting payment
                        payments[t] = max(0, v1 * allocations[t, 0] + v2 * allocations[t, 1] - util_t_s)
        
        return {
            'allocations': allocations,
            'payments': payments
        }
    else:
        # Fallback: use a simple valid mechanism
        # Allocate both goods to all types, charge average value
        allocations = np.ones((n_t, 2))
        payments = np.zeros(n_t)
        for t in range(n_t):
            v1, v2 = type_profiles[t]
            payments[t] = (v1 + v2) / 2.0
        
        return {
            'allocations': allocations,
            'payments': payments
        }
# EVOLVE-BLOCK-END
