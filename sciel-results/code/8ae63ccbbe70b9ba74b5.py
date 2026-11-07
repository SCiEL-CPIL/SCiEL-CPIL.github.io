import math
import time
import sys
import random

def construct():
    n = 256
    deadline = time.time() + 50  # Safety margin
    
    # 1. Generate edges (conflicts)
    conflicts = [0] * (n + 1) # 1-indexed, bitmask of conflicts
    weights = [0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = i
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) > 1:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Try SciPy MILP first
    try:
        import scipy.optimize as so
        import scipy.sparse as sp
        import numpy as np
        
        # Variables x_1 ... x_256
        # Minimize -sum(i * x_i)
        c = np.array([-i for i in range(1, n + 1)])
        
        # Constraints: x_i + x_j <= 1 for all conflicts
        # Number of constraints is number of edges
        edges = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if math.gcd(i, j) > 1:
                    edges.append((i, j))
        
        if not edges:
             # No conflicts, take all
             return list(range(1, n + 1))
        
        m_constraints = len(edges)
        rows = []
        cols = []
        vals = []
        
        for idx, (u, v) in enumerate(edges):
            # x_u + x_v <= 1
            # Variable index is 0-based in scipy: u-1, v-1
            rows.append(idx)
            cols.append(u - 1)
            vals.append(1)
            
            rows.append(idx)
            cols.append(v - 1)
            vals.append(1)
            
        A = sp.csr_matrix((vals, (rows, cols)), shape=(m_constraints, n))
        
        # Bounds
        lb = np.zeros(n)
        ub = np.ones(n)
        
        # Integrality
        integrality = np.ones(n) # 1 means integer
        
        res = so.milp(c, constraints=[so.LinearConstraint(A, ub=1)], 
                      integrality=integrality, bounds=so.Bounds(lb, ub))
        
        if res.status == 0: # Optimal
            x = res.x
            # x is float array, check if close to 0 or 1
            selected = []
            for i in range(n):
                if x[i] > 0.5:
                    selected.append(i + 1)
            return [int(v) for v in selected]
        else:
            # If MILP fails, fall back
            pass
            
    except Exception as e:
        pass
        
    # Fallback: Branch and Bound
    # Use bitmasks
    # Precompute weights array for quick access
    w = [0] + list(range(1, n + 1))
    
    best_sum = 0
    best_set = []
    
    # Initial greedy solution to set a good lower bound
    def greedy_initial(mask):
        curr_sum = 0
        curr_set = []
        m = mask
        while m:
            # Pick vertex with max weight
            max_w = -1
            max_v = -1
            mm = m
            while mm:
                # Get lowest set bit
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                if w[v] > max_w:
                    max_w = w[v]
                    max_v = v
                mm &= mm - 1
            if max_v == -1: break
            curr_sum += w[max_v]
            curr_set.append(max_v)
            m = m & ~conflicts[max_v] & ~(1 << max_v)
        return curr_sum, curr_set
        
    init_sum, init_set = greedy_initial((1 << (n + 1)) - 2) # Mask for 1..256
    # Note: 1 << 256 is large, Python handles big ints.
    # Mask bits 1 to 256. Bit 0 is unused.
    full_mask = (1 << (n + 1)) - 2
    
    best_sum = init_sum
    best_set = init_set
    
    def solve_bnb(mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        # Upper bound: sum of weights of all remaining vertices
        # Calculate sum of weights in mask
        rem_sum = 0
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            rem_sum += w[v]
            mm &= mm - 1
            
        if current_sum + rem_sum <= best_sum:
            return
            
        # Heuristic: Pick vertex with highest weight in mask
        max_w = -1
        max_v = -1
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            if w[v] > max_w:
                max_w = w[v]
                max_v = v
            mm &= mm - 1
            
        if max_v == -1:
            return
            
        # Branch 1: Include max_v
        # Check if it's actually in mask (it is)
        new_mask_include = mask & ~conflicts[max_v] & ~(1 << max_v)
        solve_bnb(new_mask_include, current_sum + w[max_v], current_set + [max_v])
        
        # Branch 2: Exclude max_v
        new_mask_exclude = mask & ~(1 << max_v)
        solve_bnb(new_mask_exclude, current_sum, current_set)
        
    solve_bnb(full_mask, 0, [])
    
    return [int(x) for x in best_set]
