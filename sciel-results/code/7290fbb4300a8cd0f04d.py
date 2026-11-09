import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import time

def construct():
    n = 256
    
    # 1. Try MILP
    try:
        # Generate constraints
        row_indices = []
        col_indices = []
        data = []
        
        # For each a, b, c such that a < b <= c and a | (b+c)
        # Add constraint x_a + x_b + x_c <= 2
        for a in range(1, n + 1):
            # b+c is multiple of a
            # min b = a+1, min c = a+1 => min sum 2a+2
            # max sum 2n
            for k in range(3, 2 * n // a + 1):
                S = k * a
                # b + c = S
                # a < b <= c <= n
                # b >= max(a+1, S-n)
                # b <= S//2
                min_b = max(a + 1, S - n)
                max_b = S // 2
                
                for b in range(min_b, max_b + 1):
                    c = S - b
                    if c > n:
                        continue
                    if c < b:
                        continue
                    # Constraint: x[a-1] + x[b-1] + x[c-1] <= 2
                    row_idx = len(data) // 3 # Current row in A
                    # We are building sparse matrix, so we just append indices
                    row_indices.extend([row_idx, row_idx, row_idx])
                    col_indices.extend([a - 1, b - 1, c - 1])
                    data.extend([1, 1, 1])
                    
        if not data:
            return list(range(1, n + 1))
            
        num_constraints = len(data) // 3
        A = csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, n))
        
        c_obj = -np.ones(n)
        b_ub = np.full(num_constraints, 2)
        integrality = np.ones(n)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(c=c_obj, 
                   constraints=LinearConstraint(A, ub=b_ub), 
                   integrality=integrality, 
                   bounds=bounds, 
                   options={"time_limit": 50, "mip_rel_gap": 0.0})
                   
        if res.status == 0:
            x = res.x
            selected = [i + 1 for i in range(n) if x[i] > 0.5]
            if selected:
                return selected
        # If status is not 0, or selected is empty, fall through to greedy
    except Exception:
        pass

    # 2. Fallback: Greedy Construction
    # We want to maximize size.
    # Small numbers are more likely to divide sums, so they are more "dangerous" as 'a'.
    # Large numbers are less likely to be 'a' for any pair in the set?
    # Actually, if we pick large numbers, the sums are large, and divisors are likely small (not in set).
    # So a set of large numbers is likely valid.
    # Let's try picking from n down to 1.
    
    S = []
    S_set = set()
    
    for i in range(n, 0, -1):
        # Check if i can be added to S
        valid = True
        # i can be a, b, or c in the triple (a, b, c) with a < b <= c
        
        # Case 1: i is the largest element c.
        # Then we need a, b in S such that a < b < i and a | (b + i)
        # Since b < i, b is in S.
        # We iterate b in S where b < i
        for b in S:
            if b >= i:
                continue
            # Now find a in S such that a < b and a | (b + i)
            # To optimize, we can precompute divisors of (b+i)
            # But S is small initially.
            # Check all a in S where a < b
            for a in S:
                if a < b:
                    if (b + i) % a == 0:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Case 2: i is the middle element b.
        # Then a < i < c. a in S, c in S.
        # a | (i + c)
        for a in S:
            if a >= i:
                continue
            # c must be in S, c > i
            # We can check if any c in S with c > i satisfies a | (i+c)
            # This is harder to check quickly without iterating c.
            # But S is sorted? No, we add in decreasing order, so S is decreasing.
            # Let's keep S as a list.
            for c in S:
                if c > i:
                    if (i + c) % a == 0:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Case 3: i is the smallest element a.
        # Then i < b <= c. b, c in S.
        # i | (b + c)
        for b in S:
            if b <= i:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % i == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(i)
            
    if S:
        return sorted(S)
        
    # Final fallback
    return list(range(171, n + 1))
