import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    # 1. Generate conflicting triples
    # Triple (a, b, c) with a < b <= c <= n and a | (b+c)
    # We want to forbid x_a + x_b + x_c = 3
    # So constraint: x_a + x_b + x_c <= 2
    
    constraints_A_rows = []
    
    # Optimization: Precompute valid (b,c) pairs for each sum?
    # Or just iterate a, then multiples, then b.
    
    for a in range(1, n + 1):
        # b > a, c >= b, c <= n
        # Min sum = (a+1) + (a+1) = 2a + 2
        # Max sum = n + n = 2n
        min_sum = 2 * a + 2
        max_sum = 2 * n
        
        # Find multiples of a in [min_sum, max_sum]
        first_k = (min_sum + a - 1) // a
        last_k = max_sum // a
        
        for k in range(first_k, last_k + 1):
            s = k * a
            # b + c = s
            # a < b <= c <= n
            # b >= a + 1
            # c = s - b
            # c >= b => s - b >= b => 2b <= s => b <= s // 2
            # c <= n => s - b <= n => b >= s - n
            # b <= n
            
            b_min = max(a + 1, s - n)
            b_max = min(s // 2, n)
            
            if b_min > b_max:
                continue
                
            for b in range(b_min, b_max + 1):
                c = s - b
                # We already ensured b <= c and c <= n via b_max and b_min logic mostly
                # Check c <= n is covered by b >= s - n
                # Check c >= b is covered by b <= s/2
                # Check b > a is covered by b_min >= a+1
                
                # Add constraint row for (a, b, c)
                # Indices are 0-based in ILP
                # We can store as tuple (a-1, b-1, c-1)
                constraints_A_rows.append((a - 1, b - 1, c - 1))
                
                if time.time() > deadline - 10:
                    # Fallback to greedy if generation takes too long
                    return greedy_construct(n, deadline)

    M = len(constraints_A_rows)
    
    # If no constraints, take all? (Unlikely, but possible for small n)
    if M == 0:
        return list(range(1, n + 1))

    # Build A matrix (sparse)
    # scipy milp accepts sparse matrices
    from scipy.sparse import coo_matrix
    
    rows = []
    cols = []
    vals = []
    
    for idx, (i, j, k) in enumerate(constraints_A_rows):
        rows.extend([idx, idx, idx])
        cols.extend([i, j, k])
        vals.extend([1, 1, 1])
        
    A = coo_matrix((vals, (rows, cols)), shape=(M, n))
    
    # Objective: maximize sum(x)
    # milp minimizes c^T x, so c = -1
    c = -np.ones(n)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: A x <= 2
    # LinearConstraint(A, lb=-np.inf, ub=2)
    lin_cons = LinearConstraint(A, lb=-np.inf, ub=2)
    
    # Integrality: 0 means continuous, 1 means integer
    integrality = np.ones(n)
    
    try:
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 70})
        
        if res.status == 0 or res.status == 1: # Optimal or Local Optimal
            # res.x is the solution
            # Filter valid integers
            S = set()
            for i in range(n):
                if res.x[i] > 0.5:
                    S.add(i + 1)
            
            # Verify solution validity
            if is_valid_set(S, n):
                return list(S)
            else:
                # If invalid, fallback
                return greedy_construct(n, deadline)
        else:
            # Solver failed or timeout
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
    # Original greedy logic as fallback
    K = 171
    S = set(range(K, n + 1))
    
    for x in range(K - 1, 0, -1):
        if time.time() > deadline - 5:
            break
        valid = True
        lst = list(S)
        m = len(lst)
        for i in range(m):
            b = lst[i]
            for j in range(i, m):
                c = lst[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(x)
            
    return list(S)
