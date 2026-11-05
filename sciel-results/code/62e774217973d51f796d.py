import numpy as np
from itertools import combinations
from math import lcm
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples
    forbidden_triples = []
    for a, b, c in combinations(numbers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            forbidden_triples.append((a-1, b-1, c-1))
            
    if not forbidden_triples:
        return numbers
        
    # 2. Formulate ILP
    c = -np.ones(N)
    
    M = len(forbidden_triples)
    if M == 0:
        return numbers

    rows = []
    cols = []
    data = []
    for idx, t in enumerate(forbidden_triples):
        rows.extend([idx, idx, idx])
        cols.extend(t)
        data.extend([1, 1, 1])
        
    A_ub = csr_matrix((data, (rows, cols)), shape=(M, N))
    b_ub = np.full(M, 2)
    
    constraints = LinearConstraint(A_ub, 0, b_ub)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c, constraints=constraints, integrality=np.ones(N), bounds=bounds, time_limit=55)
        if res.status == 0:
            sol = [i+1 for i in range(N) if res.x[i] > 0.5]
            if len(sol) > 0:
                return sol
    except Exception:
        pass
        
    # Fallback valid set
    return [1, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
