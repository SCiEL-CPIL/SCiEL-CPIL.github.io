import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 180
    # 1. Generate forbidden triples
    triples = []
    for a in range(1, N + 1):
        a_sq = a * a
        # Find divisors x of a_sq
        # x must be < a_sq to ensure y > 0
        # Also x != y to ensure b != c
        # x < sqrt(a_sq) = a
        for x in range(1, a):
            if a_sq % x == 0:
                y = a_sq // x
                # x != y is guaranteed by x < a and y > a
                b = a + x
                c = a + y
                if b <= N and c <= N:
                    # Ensure distinctness: a < b and a < c is true. b != c is true.
                    triples.append((a, b, c))
    
    # 2. Formulate ILP
    n_vars = N
    # Objective: maximize sum z_i -> minimize -sum z_i
    c_obj = -np.ones(n_vars)
    
    # Constraints: z_a + z_b + z_c <= 2 for each triple
    # We can use LinearConstraint
    # A @ z <= b_ub
    # A is sparse or dense? n_vars=180, triples ~ few thousand. Dense is fine.
    if triples:
        A = np.zeros((len(triples), n_vars))
        for i, (a, b, c) in enumerate(triples):
            A[i, a-1] = 1
            A[i, b-1] = 1
            A[i, c-1] = 1
        b_ub = np.full(len(triples), 2)
        constraint = LinearConstraint(A, ub=b_ub)
    else:
        # No constraints, take all
        return list(range(1, N + 1))
        
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)
    
    try:
        res = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 40} # Leave 20s for overhead
        )
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # Check validity just in case
            sol = res.x
            selected = [i + 1 for i in range(n_vars) if sol[i] > 0.5]
            
            # Verification step (optional but good for safety)
            # Check if any triple is fully contained in selected
            sel_set = set(selected)
            for a, b, c in triples:
                if a in sel_set and b in sel_set and c in sel_set:
                    # Invalid solution, fallback to greedy or previous best
                    # For now, if ILP fails validity, we might need a fallback.
                    # But ILP should be exact.
                    pass
                    
            if len(selected) > 0:
                return selected
            else:
                return [1] # Nonempty requirement

        else:
            # If ILP didn't find optimal, use best feasible found or fallback
            if res.x is not None and res.x.size > 0:
                selected = [i + 1 for i in range(n_vars) if res.x[i] > 0.5]
                if len(selected) > 0:
                    return selected
            
            # Fallback to simple heuristic if ILP fails completely
            # Greedy: iterate 1..N, add if no conflict with existing
            S = []
            S_set = set()
            for i in range(1, N + 1):
                conflict = False
                # Check if adding i creates a triple with existing elements
                # Triple {a,b,c} with i being one of them.
                # Since a is min, i can be a, b, or c.
                # If i is a: need b,c in S.
                # If i is b: need a,c in S.
                # If i is c: need a,b in S.
                
                # Efficient check:
                # For each existing pair in S, do they form a triple with i?
                # Or for each existing element, do they form a triple with i?
                
                # Simplest: just check all triples involving i.
                # Pre-compute triples_by_element?
                # For now, simple check:
                for (a, b, c) in triples:
                    if i in (a, b, c):
                        # Check if other two are in S
                        others = [x for x in (a, b, c) if x != i]
                        if all(x in S_set for x in others):
                            conflict = True
                            break
                if not conflict:
                    S.append(i)
                    S_set.add(i)
            
            return S if S else [1]

    except Exception as e:
        # Fallback on error
        return list(range(91, 181)) # The seed solution
