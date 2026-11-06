import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    limit = 55.0 # Safety margin for 60s limit

    n = 256
    
    # Weights: 1/i
    weights = np.array([1.0 / i for i in range(1, n + 1)])
    
    # Generate all 3-term arithmetic progressions
    constraints = []
    for a in range(1, n + 1):
        max_d = (n - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            constraints.append((a - 1, b - 1, c - 1))
            
    num_constraints = len(constraints)
    
    # Build sparse constraint matrix A
    rows = []
    cols = []
    data = []
    
    for i, (a, b, c) in enumerate(constraints):
        rows.extend([i, i, i])
        cols.extend([a, b, c])
        data.extend([1, 1, 1])
        
    A = csr_matrix((data, (rows, cols)), shape=(num_constraints, n))
    b = np.full(num_constraints, 2.0)
    
    # Objective: Maximize sum(weights * x) -> Minimize -sum(weights * x)
    c = -weights
    
    integrality = np.ones(n)
    
    lin_cons = LinearConstraint(A, lb=None, ub=b)
    var_bounds = Bounds(lb=0, ub=1)
    
    try:
        # Set time limit to ensure we return before hard kill
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=var_bounds, 
                   options={'time_limit': 50})
        
        if res.success and res.x is not None:
            solution_mask = (res.x > 0.5).astype(int)
            selected_indices = [i + 1 for i in range(n) if solution_mask[i] == 1]
            
            if selected_indices:
                return selected_indices
                
    except Exception:
        pass
        
    # Fallback Greedy Solution
    # Simple greedy: iterate 1 to n, add if no conflict with previously added
    # This is O(N^2) which is fast for N=256
    S = set()
    for x in range(1, n + 1):
        # Check if x forms a 3-AP with any two elements in S
        # Or just check if x conflicts with any single element? 
        # No, conflict is pairwise in the sense that if we add x, we must ensure no existing pair in S forms an AP with x.
        # Actually, the condition is "no 3-AP in S".
        # If we add x, we need to ensure there are no a, b in S such that {a, b, x} is an AP.
        # This is equivalent to checking if x is in the "conflict" of any pair?
        # Easier: For each x, check if adding it creates a 3-AP.
        # A 3-AP involving x can be:
        # 1. x, y, z (x is start): y, z in S, y-x = z-y
        # 2. y, x, z (x is middle): y, z in S, x-y = z-x
        # 3. y, z, x (x is end): y, z in S, z-y = x-z
        
        # To make it O(N^2) total, we can precompute or just check.
        # Since N=256, checking all pairs in S for each x is O(|S|^2) per x, total O(N^3) worst case ~ 16M ops. Fine.
        
        valid = True
        S_list = list(S)
        # Check if x is middle: y, x, z => z = 2x - y
        for y in S_list:
            z = 2 * x - y
            if z in S:
                valid = False
                break
        if not valid:
            continue
            
        # Check if x is end (start or end of AP with two in S)
        # If x is start: x, y, z. y, z in S. y = x + d, z = x + 2d.
        # If x is end: y, z, x. y, z in S. z = x - d, y = x - 2d.
        # Combined: Check if there exist y, z in S such that {x, y, z} is AP.
        # This is covered by:
        # 1. x is middle: checked above.
        # 2. x is not middle: then one of the other two is middle.
        #    If y is middle: x, y, z. z = 2y - x. Check if z in S.
        #    If z is middle: x, z, y. y = 2z - x. Check if y in S.
        
        for y in S_list:
            # y is middle
            z = 2 * y - x
            if z in S:
                valid = False
                break
            # y is start (x is end) -> z is middle? No.
            # If y is start, x is end, middle is (x+y)/2.
            # Let's just check all pairs (y, z) in S? No, that's slow.
            # The check "x is middle" covers APs where x is the average.
            # The check "y is middle" covers APs where y is the average.
            # What if z is the average? Then x and y are ends.
            # z = (x+y)/2.
            # So we need to check if (x+y) is even and (x+y)//2 is in S.
            
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z in S:
                    valid = False
                    break
        if valid:
            S.add(x)
            
    return list(S)
