import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    N = 180
    
    # Precompute conflicts and triples
    # triples contains (a, b, c) such that 1/a = 1/b + 1/c with a < b < c
    triples = []
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find factor pairs d, e of a^2 such that d < e
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N:
                triples.append((a, b, c))
                
    # Precompute for each vertex, the list of triples it participates in
    vertex_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples):
        vertex_triples[a].append(idx)
        vertex_triples[b].append(idx)
        vertex_triples[c].append(idx)
        
    # Attempt to solve using MILP if available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c_obj = -np.ones(N)
        
        # Constraints: for each triple, x_a + x_b + x_c <= 2
        # A_ub is the matrix for A_ub @ x <= b_ub
        A_ub = np.zeros((len(triples), N))
        b_ub = np.full(len(triples), 2)
        
        for idx, (a, b, c) in enumerate(triples):
            A_ub[idx, a-1] = 1
            A_ub[idx, b-1] = 1
            A_ub[idx, c-1] = 1
            
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        
        # Integrality: all variables are integer
        integrality = np.ones(N)
        
        # Solve
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            bounds=bounds,
            integrality=integrality
        )
        
        S = set()
        if res.status == 0:
            for i in range(N):
                if res.x[i] > 0.5:
                    S.add(i+1)
            
            # Verify validity
            valid = True
            for a, b, c in triples:
                if a in S and b in S and c in S:
                    valid = False
                    break
            
            if valid:
                # Local search polish: try to add more elements
                # Try adding each element not in S
                for v in range(1, N + 1):
                    if v in S:
                        continue
                    if time.time() > deadline - 5:
                        break
                    
                    # Check if adding v creates a conflict
                    can_add = True
                    for tidx in vertex_triples[v]:
                        a, b, c = triples[tidx]
                        # v is one of a, b, c; check if the other two are in S
                        others = [x for x in (a, b, c) if x != v]
                        if others[0] in S and others[1] in S:
                            can_add = False
                            break
                    
                    if can_add:
                        S.add(v)
                        
                return sorted(S)
                
    except Exception:
        pass
        
    # Fallback: Greedy construction with precomputed vertex_triples
    S = set()
    for a in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        can_add = True
        # Check if adding a creates a conflict using precomputed vertex_triples
        for tidx in vertex_triples[a]:
            tri = triples[tidx]
            # a is one of the elements; check if the other two are in S
            others = [x for x in tri if x != a]
            if others[0] in S and others[1] in S:
                can_add = False
                break
        
        if can_add:
            S.add(a)
            
    return sorted(S)
