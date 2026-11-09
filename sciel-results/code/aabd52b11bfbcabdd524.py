import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 180
    
    # 1. Generate all forbidden triples {a, b, c} with a < b, c and 1/a = 1/b + 1/c
    # This implies (b-a)(c-a) = a^2
    # Let a be the smallest element.
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        for d in range(1, int(np.sqrt(a2)) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b == c:
                    continue
                if b <= n and c <= n:
                    triples.append((a, b, c))
    
    if not triples:
        return list(range(1, n + 1))

    # 2. Set up ILP
    # Variables: x_1, ..., x_n
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c_obj = -np.ones(n)
    
    # Constraints: x_a + x_b + x_c <= 2 for each triple
    # A_ub * x <= b_ub
    A_ub = np.zeros((len(triples), n))
    b_ub = np.full(len(triples), 2)
    
    for i, (a, b, c) in enumerate(triples):
        A_ub[i, a-1] = 1
        A_ub[i, b-1] = 1
        A_ub[i, c-1] = 1
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer
    integrality = np.ones(n)
    
    try:
        res = milp(
            c=c_obj,
            constraints=[LinearConstraint(A_ub, -np.inf, b_ub)],
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            # res.x contains the solution
            # Due to floating point, threshold at 0.5
            solution = [i + 1 for i in range(n) if res.x[i] > 0.5]
            if len(solution) > 0:
                return sorted(solution)
        else:
            # Fallback to greedy if ILP fails
            pass
            
    except Exception:
        pass

    # Fallback to greedy if ILP fails or is not available
    S = set()
    conflict_pairs = {}
    for k in range(1, n + 1):
        k2 = k * k
        pairs = set()
        for d in range(1, k2 + 1):
            if k2 % d == 0:
                e = k2 // d
                a = k + d
                b = k + e
                if a == b:
                    continue
                if 1 <= a <= n and 1 <= b <= n:
                    pairs.add((min(a, b), max(a, b)))
        conflict_pairs[k] = pairs

    for k in range(n, 0, -1):
        conflict = False
        for (a, b) in conflict_pairs[k]:
            if a in S and b in S:
                conflict = True
                break
        if not conflict:
            S.add(k)
            
    return sorted(S)
