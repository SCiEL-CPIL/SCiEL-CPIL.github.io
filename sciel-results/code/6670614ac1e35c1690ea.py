def construct():
    import numpy as np
    from math import gcd
    
    n = 256
    nums = list(range(2, n + 1))
    m = len(nums)
    idx_map = {num: i for i, num in enumerate(nums)}
    
    # Build conflict list
    conflicts = []
    conflict_set = {a: set() for a in nums}
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            g = gcd(a, b)
            l = a * b // g
            if l <= n:
                conflicts.append((i, j)) # Wait, j is not defined here in this loop structure if I just append
                conflict_set[a].add(b)
                conflict_set[b].add(a)
                
    # Re-do conflict building correctly for ILP indices
    # We need indices i, j
    edges = []
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if i >= j:
                continue
            g = gcd(a, b)
            l = a * b // g
            if l <= n:
                edges.append((i, j))
                
    # Try ILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import coo_matrix
        
        c = np.array([-1.0 / num for num in nums])
        
        if not edges:
            return nums
            
        rows = []
        cols = []
        vals = []
        for i, j in edges:
            rows.append(len(rows))
            cols.append(i)
            vals.append(1)
            cols.append(j)
            vals.append(1)
            
        A = coo_matrix((vals, (rows, cols)), shape=(len(rows), m))
        lb = np.zeros(len(rows))
        ub = np.ones(len(rows))
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(m)
        bounds = Bounds(0, 1)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            selected = [nums[i] for i in range(m) if res.x[i] > 0.5]
            return selected
    except Exception:
        pass
        
    # Fallback: Improved Greedy + Local Search
    # Use the logic from the current program but ensure it's self-contained
    
    weight = {a: 1.0 / a for a in nums}
    
    def greedy(order):
        selected = set()
        for a in order:
            if not conflict_set[a] & selected:
                selected.add(a)
        return selected

    # Ordering 1: by a ascending
    order1 = sorted(nums, key=lambda x: x)
    s1 = greedy(order1)
    
    # Ordering 2: by degree ascending
    order2 = sorted(nums, key=lambda x: (len(conflict_set[x]), x))
    s2 = greedy(order2)
    
    # Ordering 3: by weight/degree
    order3 = sorted(nums, key=lambda x: (-weight[x] / max(1, len(conflict_set[x])), x))
    s3 = greedy(order3)
    
    candidates = [(s1, sum(weight[a] for a in s1)), 
                  (s2, sum(weight[a] for a in s2)), 
                  (s3, sum(weight[a] for a in s3))]
    
    best_set, best_sum = max(candidates, key=lambda x: x[1])
    
    # Local search
    improved = True
    while improved:
        improved = False
        # Try adding
        for a in nums:
            if a not in best_set and not conflict_set[a] & best_set:
                best_set.add(a)
                best_sum += weight[a]
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap
        for a in list(best_set):
            for b in nums:
                if b in best_set or b == a:
                    continue
                new_set = (best_set - {a}) | {b}
                if conflict_set[b] & (best_set - {a}):
                    continue
                new_sum = best_sum - weight[a] + weight[b]
                if new_sum > best_sum:
                    best_set = new_set
                    best_sum = new_sum
                    improved = True
                    break
            if improved:
                break
                
    return list(best_set)
