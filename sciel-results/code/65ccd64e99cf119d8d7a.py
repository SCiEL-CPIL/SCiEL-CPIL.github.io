import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    # 1. Generate conflicting triples
    # Triple (a, b, c) with a < b <= c <= n and a | (b+c)
    # Constraint: x_a + x_b + x_c <= 2
    
    constraints_A_rows = []
    
    for a in range(1, n + 1):
        min_sum = 2 * a + 2
        max_sum = 2 * n
        
        first_k = (min_sum + a - 1) // a
        last_k = max_sum // a
        
        for k in range(first_k, last_k + 1):
            s = k * a
            b_min = max(a + 1, s - n)
            b_max = min(s // 2, n)
            
            if b_min > b_max:
                continue
                
            for b in range(b_min, b_max + 1):
                c = s - b
                if c > n:
                    continue
                if c < b:
                    continue
                # Add constraint row for (a, b, c)
                constraints_A_rows.append((a - 1, b - 1, c - 1))
                
                if time.time() > deadline - 15:
                    return greedy_construct(n, deadline)

    M = len(constraints_A_rows)
    
    if M == 0:
        return list(range(1, n + 1))

    # Build A matrix (sparse)
    rows = []
    cols = []
    vals = []
    
    for idx, (i, j, k) in enumerate(constraints_A_rows):
        rows.extend([idx, idx, idx])
        cols.extend([i, j, k])
        vals.extend([1, 1, 1])
        
    A = coo_matrix((vals, (rows, cols)), shape=(M, n))
    
    # Objective: maximize sum(x)
    c = -np.ones(n)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: A x <= 2
    lin_cons = LinearConstraint(A, lb=-np.inf, ub=2)
    
    # Integrality: 1 means integer
    integrality = np.ones(n)
    
    try:
        # Provide an initial guess to help the solver
        # Initial guess: top third [171, 256]
        x0 = np.zeros(n)
        for i in range(170, n): # 0-indexed, so 170 is 171
            x0[i] = 1.0
            
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 60, "presolve": "on"},
                   x0=x0)
        
        if res.status == 0 or res.status == 1: # Optimal or Local Optimal
            S = set()
            for i in range(n):
                if res.x[i] > 0.5:
                    S.add(i + 1)
            
            if is_valid_set(S, n):
                return list(S)
            else:
                return greedy_construct(n, deadline)
        else:
            return greedy_construct(n, deadline)
            
    except Exception as e:
        return greedy_construct(n, deadline)

def is_valid_set(S, n):
    if not S:
        return False
    lst = sorted(S)
    m = len(lst)
    for i in range(m):
        a = lst[i]
        for j in range(i + 1, m):
            b = lst[j]
            for k in range(j, m):
                c = lst[k]
                if a < b <= c:
                    if (b + c) % a == 0:
                        return False
    return True

def greedy_construct(n, deadline):
    # Greedy descending: add x if it doesn't conflict with larger elements in S
    S = set()
    for x in range(n, 0, -1):
        if time.time() > deadline - 5:
            break
        valid = True
        # Check if x divides any sum of two larger elements in S
        # x < b <= c
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(x)
            
    return list(S)
