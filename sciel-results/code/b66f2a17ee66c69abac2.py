def construct():
    import math
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np

    n = 256
    max_sum = 2 * n
    squares = []
    i = 1
    while i * i <= max_sum:
        if i * i >= 2: # 1 is a square but min sum is 1+1=2, so 1 is irrelevant for sums, but 4 is the first relevant one. Actually 1+? no, min a,b is 1. 1+1=2. 1+2=3. 1+3=4. So 4 is the first square sum.
            squares.append(i * i)
        i += 1
    
    # Variables x[0] ... x[255] correspond to numbers 1 ... 256
    c = -np.ones(n) # Minimize negative sum to maximize sum
    
    # Build constraints
    # We need A_ub @ x <= b_ub
    # Constraint type: x_a + x_b <= 1 for a < b, 2*x_a <= 1 for a == b
    # In scipy milp, we use LinearConstraint(A, lb, ub)
    # Or simpler: A_ub, b_ub if using older interface, but milp uses LinearConstraint.
    # Let's build rows for A.
    
    rows = []
    rhs = []
    
    for S in squares:
        # Pairs (a, b) such that a + b = S, 1 <= a <= b <= 256
        # a ranges from max(1, S-256) to min(256, S//2)
        a_min = max(1, S - 256)
        a_max = min(256, S // 2)
        
        for a in range(a_min, a_max + 1):
            b = S - a
            # a <= b is guaranteed by a <= S//2
            if a == b:
                # 2 * x_{a-1} <= 1  => x_{a-1} <= 0.5
                # In binary, this means x_{a-1} must be 0.
                # We can add constraint: 2 * x_{a-1} <= 1
                row = np.zeros(n)
                row[a-1] = 2
                rows.append(row)
                rhs.append(1)
            else:
                # x_{a-1} + x_{b-1} <= 1
                row = np.zeros(n)
                row[a-1] = 1
                row[b-1] = 1
                rows.append(row)
                rhs.append(1)
                
    if not rows:
        # Fallback if no constraints (should not happen)
        return list(range(1, n+1))

    A = np.array(rows)
    b_ub = np.array(rhs)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are binary
    integrality = np.ones(n)
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A, ub=b_ub), bounds=bounds, integrality=integrality)
        
        if res.status == 0:
            # res.x contains the optimal solution
            selected = []
            for i in range(n):
                if res.x[i] > 0.5:
                    selected.append(i + 1)
            
            # Verify
            import math
            for i, a in enumerate(selected):
                for b in selected[i:]:
                    s = a + b
                    root = math.isqrt(s)
                    if root * root == s:
                         # If verification fails, return a valid greedy set as fallback
                         break
                else:
                    continue
                break
            else:
                return selected
            
        # If status != 0 or verification failed, fallback to greedy
    except Exception:
        pass

    # Fallback: Simple Greedy
    # This ensures we always return a valid non-empty set even if ILP fails
    all_candidates = [a for a in range(1, n + 1) if 2 * a not in set(squares)]
    conflicts = {a: set() for a in all_candidates}
    for idx, a in enumerate(all_candidates):
        for b in all_candidates[idx+1:]:
            if (a + b) in set(squares):
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    def try_add_fast(s, v):
        return len(conflicts[v] & s) == 0
        
    current_set = set()
    for v in all_candidates:
        if try_add_fast(current_set, v):
            current_set.add(v)
            
    return sorted(current_set)
