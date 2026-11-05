"""Rewritten program using monotone LP reduction for speed and stability."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    # --- Parse inputs ---
    # distribution_spec: list of 4 probabilities for values {0,1,2,3}
    # support: set of allowed reports (ignored for output shape, but used for constraints? 
    # "Truthfulness holds only on the declared report/value support"
    # If support is smaller, we only enforce IC on support?
    # But output is 16x2. Let's assume we optimize for the full space to be safe, 
    # or only enforce IC on support if support is provided and smaller.
    # Given "three fixed distributions", and "Every allowed report is enumerated",
    # likely support is the full {0,1,2,3}^2.
    
    # Parse distribution
    p_vals = None
    try:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
            if len(probs) == 4:
                p_vals = np.outer(probs, probs)
            elif len(probs) == 16:
                p_vals = probs.reshape(4, 4)
        elif isinstance(distribution_spec, dict):
            # Handle dict format
            p_vals = np.zeros((4,4))
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    p_vals[k[0], k[1]] = v
            # Check if it's a marginal dict
            if all(isinstance(k, int) for k in distribution_spec.keys()):
                marg = np.array([distribution_spec.get(i, 0) for i in range(4)])
                p_vals = np.outer(marg, marg)
    except Exception:
        pass
    
    if p_vals is None:
        p_vals = np.ones((4,4)) / 16.0
    
    p_vals = p_vals / p_vals.sum()
    
    # Define full type space
    all_types = [(i, j) for i in range(4) for j in range(4)]
    type_index = {t: i for i, t in enumerate(all_types)}
    
    # Determine effective support for IC/IR constraints
    # If 'support' is None or empty, use all_types
    if support is None:
        effective_support = all_types
    else:
        if isinstance(support, (list, tuple, set)):
            effective_support = [tuple(r) for r in support]
            if not effective_support:
                effective_support = all_types
        else:
            effective_support = all_types
            
    # Ensure all types in effective_support are in type_index (they should be)
    
    # Variables:
    # x[0..31]: allocations for 16 types, 2 goods each
    # x[32..47]: payments for 16 types
    n_vars = 48
    c = np.zeros(n_vars)
    for i in range(16):
        t = all_types[i]
        c[32 + i] = -p_vals[t[0], t[1]]
        
    A_ub = []
    b_ub = []
    
    # 1. Monotonicity Constraints
    # x[t+e, g] >= x[t, g]  => -x[t+e, g] + x[t, g] <= 0
    # Only for types where t+e is in bounds
    for i in range(4):
        for j in range(4):
            t = (i, j)
            idx = type_index[t]
            # Check neighbor in dimension 0
            if i < 3:
                t_next = (i+1, j)
                idx_next = type_index[t_next]
                for g in range(2):
                    row = np.zeros(n_vars)
                    row[idx*2 + g] = 1
                    row[idx_next*2 + g] = -1
                    A_ub.append(row)
                    b_ub.append(0)
            # Check neighbor in dimension 1
            if j < 3:
                t_next = (i, j+1)
                idx_next = type_index[t_next]
                for g in range(2):
                    row = np.zeros(n_vars)
                    row[idx*2 + g] = 1
                    row[idx_next*2 + g] = -1
                    A_ub.append(row)
                    b_ub.append(0)
                    
    # 2. Individual Rationality (IR)
    # p[t] - sum x[t,g] * t[g] <= 0
    for t in effective_support:
        ti = type_index[t]
        row = np.zeros(n_vars)
        row[32 + ti] = 1
        row[ti*2 + 0] = -t[0]
        row[ti*2 + 1] = -t[1]
        A_ub.append(row)
        b_ub.append(0)
        
    # 3. Local Incentive Compatibility (IC)
    # For each adjacent pair (t, t'), we need:
    # U(t, t) >= U(t, t')  => p[t'] - p[t] <= x[t'] dot t - x[t] dot t
    # U(t', t') >= U(t', t) => p[t] - p[t'] <= x[t] dot t' - x[t'] dot t'
    
    # Generate adjacent pairs
    adjacent_pairs = set()
    for i in range(4):
        for j in range(4):
            t = (i, j)
            if i < 3:
                t_next = (i+1, j)
                adjacent_pairs.add((t, t_next))
                adjacent_pairs.add((t_next, t)) # Symmetric handling, or just one direction?
            if j < 3:
                t_next = (i, j+1)
                adjacent_pairs.add((t, t_next))
                adjacent_pairs.add((t_next, t))
                
    # We only need to check pairs where both types are in effective_support
    # And we only need one direction per edge?
    # Actually, IC is symmetric in the sense that if U(t,t)>=U(t,t') and U(t',t')>=U(t',t),
    # we cover both.
    
    for t1, t2 in adjacent_pairs:
        # Only enforce if both are in support
        if t1 not in effective_support or t2 not in effective_support:
            continue
            
        ti1 = type_index[t1]
        ti2 = type_index[t2]
        
        # Constraint 1: U(t1, t1) >= U(t1, t2)
        # p[t2] - p[t1] - (x[t2] dot t1 - x[t1] dot t1) <= 0
        row = np.zeros(n_vars)
        row[32 + ti2] = 1
        row[32 + ti1] = -1
        # - (x[t2] dot t1 - x[t1] dot t1) = x[t1] dot t1 - x[t2] dot t1
        for g in range(2):
            row[ti1*2 + g] = t1[g]
            row[ti2*2 + g] = -t1[g]
        A_ub.append(row)
        b_ub.append(0)
        
        # Constraint 2: U(t2, t2) >= U(t2, t1)
        # p[t1] - p[t2] - (x[t1] dot t2 - x[t2] dot t2) <= 0
        row = np.zeros(n_vars)
        row[32 + ti1] = 1
        row[32 + ti2] = -1
        # - (x[t1] dot t2 - x[t2] dot t2) = x[t2] dot t2 - x[t1] dot t2
        for g in range(2):
            row[ti2*2 + g] = t2[g]
            row[ti1*2 + g] = -t2[g]
        A_ub.append(row)
        b_ub.append(0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, 6.0))  # Max value is 3+3=6
        
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            alloc = res.x[:32].reshape(16, 2)
            pay = res.x[32:]
            # Clamp to bounds to handle numerical errors
            alloc = np.clip(alloc, 0, 1)
            pay = np.clip(pay, 0, 6)
            
            if time.time() > deadline:
                return {"allocations": alloc, "payments": pay}
                
            return {"allocations": alloc, "payments": pay}
        else:
            # Fallback
            pass
    except Exception:
        pass
        
    # Fallback mechanism: Myerson-style threshold
    # For independent distributions, optimal is threshold per good.
    # Calculate virtual values? 
    # Simple fallback: Allocate if value >= median.
    alloc = np.zeros((16, 2))
    pay = np.zeros(16)
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            # Heuristic: Allocate if value > 1.5
            if i > 1.5:
                alloc[idx, 0] = 1
                pay[idx] += 1.5
            if j > 1.5:
                alloc[idx, 1] = 1
                pay[idx] += 1.5
                
    return {"allocations": alloc, "payments": pay}

# EVOLVE-BLOCK-END
