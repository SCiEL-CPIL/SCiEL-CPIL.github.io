import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute constraints for MILP and Heuristic
    constraints_data = [] # List of lists of triple indices for each K(2,2,2)
    
    for six in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            indices = [triple_index[t] for t in transversal]
            constraints_data.append(indices)
            
    num_constraints = len(constraints_data)
    
    # Strategy 1: Try MILP if we have time
    if time.time() < deadline - 10:
        try:
            rows = []
            cols = []
            data = []
            
            for i, c_indices in enumerate(constraints_data):
                for j in c_indices:
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
                    
            A = csr_matrix((data, (rows, cols)), shape=(num_constraints, num_triples))
            
            c = -np.ones(num_triples)
            cons = LinearConstraint(A, lb=-np.inf, ub=7)
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(num_triples)
            
            sol = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            
            if sol.status in [0, 1]: # Optimal or Feasible
                x = sol.x
                selected_indices = [i for i in range(num_triples) if x[i] > 0.5]
                if selected_indices:
                    result = [list(all_triples[i]) for i in selected_indices]
                    return sorted(result)
        except Exception:
            pass # Fallback to heuristic
            
    # Strategy 2: Heuristic Fallback (Original Code)
    # ... (Insert original heuristic code here) ...
