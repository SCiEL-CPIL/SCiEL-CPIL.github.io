import numpy as np
from math import lcm

def construct():
    n = 128
    
    # Step 1: Identify all "bad" triples
    # A triple (a, b, c) with a < b < c is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    
    # Precompute LCM table for speed
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    
    # If no bad triples, return all
    if not bad_triples:
        return list(range(1, n + 1))

    # Step 2: Solve using Integer Linear Programming (ILP)
    # Maximize sum(x_i) subject to x_a + x_b + x_c <= 2 for all bad triples
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = -np.ones(n)
        
        A = np.zeros((len(bad_triples), n))
        for idx, (a, b, c_val) in enumerate(bad_triples):
            A[idx, a-1] = 1.0
            A[idx, b-1] = 1.0
            A[idx, c_val-1] = 1.0
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A, 0, 2),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0 or res.status == 1:
            selected = [i + 1 for i in range(n) if res.x[i] > 0.5]
            if selected:
                return selected
                
    except Exception:
        pass

    # Fallback: Greedy heuristic if ILP fails
    import random
    
    def greedy_select(order):
        selected = []
        for x in order:
            ok = True
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i+1, len(selected)):
                    b = selected[j]
                    if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
        return selected

    best_set = []
    for seed_val in range(20):
        random.seed(seed_val)
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = greedy_select(order)
        if len(s) > len(best_set):
            best_set = s
            
    if not best_set:
        best_set = [1]
        
    return best_set
