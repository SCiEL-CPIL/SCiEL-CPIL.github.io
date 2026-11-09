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
    # Minimize hitting set (vertices to remove) such that every bad triple is hit.
    # y_i = 1 if i is removed. Minimize sum(y_i) s.t. y_a + y_b + y_c >= 1.
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = np.ones(n)
        
        A = np.zeros((len(bad_triples), n))
        for idx, (a, b, c_val) in enumerate(bad_triples):
            A[idx, a-1] = 1.0
            A[idx, b-1] = 1.0
            A[idx, c_val-1] = 1.0
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A, 1, np.inf),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0 or res.status == 1:
            removed = [i for i in range(n) if res.x[i] > 0.5]
            selected = [i + 1 for i in range(n) if i not in removed]
            if selected:
                return selected
                
    except Exception:
        pass

    # Fallback: Greedy Hitting Set heuristic if ILP fails
    # Find a small set of vertices to remove such that all bad triples are hit.
    def greedy_hitting_set():
        # Map element to set of triple indices
        elem_to_triples = {i: [] for i in range(1, n + 1)}
        for idx, (a, b, c_val) in enumerate(bad_triples):
            elem_to_triples[a].append(idx)
            elem_to_triples[b].append(idx)
            elem_to_triples[c_val].append(idx)
            
        active_triples = set(range(len(bad_triples)))
        cover = []
        
        while active_triples:
            # Find element with max active triples
            best_elem = -1
            max_count = -1
            for e in range(1, n + 1):
                # Count active triples containing e
                count = 0
                for t in elem_to_triples[e]:
                    if t in active_triples:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_elem = e
                    
            if best_elem == -1:
                break
                
            cover.append(best_elem)
            # Deactivate all triples containing best_elem
            for t in elem_to_triples[best_elem]:
                if t in active_triples:
                    active_triples.remove(t)
                    
        return cover

    cover = greedy_hitting_set()
    selected = [i for i in range(1, n + 1) if i not in cover]
    
    if not selected:
        selected = [1]
        
    return selected
