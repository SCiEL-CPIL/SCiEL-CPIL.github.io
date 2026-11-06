import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    N = 180
    
    # Generate forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2
    # Let k = b-a. Then c-a = a^2/k. So k must divide a^2.
    # b = a + k, c = a + a^2/k.
    # Conditions: a < b < c <= N
    # a+k > a => k > 0
    # a + a^2/k > a + k => a^2/k > k => a^2 > k^2 => k < a
    # c <= N => a + a^2/k <= N
    
    triples = []
    for a in range(1, N + 1):
        # Find divisors k of a^2 such that k < a
        a2 = a * a
        for k in range(1, a):
            if a2 % k == 0:
                b = a + k
                c = a + a2 // k
                if b < c and c <= N:
                    triples.append((a, b, c))
    
    # If no triples, return all numbers
    if not triples:
        return list(range(1, N + 1))
        
    num_vars = N
    
    # Objective: maximize sum(x_i) => minimize -sum(x_i)
    c_obj = np.full(num_vars, -1.0)
    
    # Constraints: x_a + x_b + x_c <= 2 for each triple
    # We build the constraint matrix A_ub
    A_rows = []
    A_cols = []
    A_vals = []
    
    for idx, (a, b, c) in enumerate(triples):
        # Indices are 0-based in the array, values are 1-based
        A_rows.extend([idx, idx, idx])
        A_cols.extend([a-1, b-1, c-1])
        A_vals.extend([1, 1, 1])
        
    if A_rows:
        A_sparse = lil_matrix((len(triples), num_vars), dtype=np.float64)
        for r, c_idx, v in zip(A_rows, A_cols, A_vals):
            A_sparse[r, c_idx] = v
        A = A_sparse.tocsr()
        
        b_ub = np.full(len(triples), 2.0)
        
        constraints = LinearConstraint(A, ub=b_ub)
    else:
        # No constraints other than bounds
        constraints = LinearConstraint(np.zeros((1, num_vars)), ub=np.inf)
        
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(
            c=c_obj,
            constraints=constraints,
            integrality=np.ones(num_vars),
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # res.x[i] is close to 0 or 1
            solution = [i+1 for i in range(num_vars) if res.x[i] > 0.5]
            
            # Check time
            if time.time() - start_time > timeout:
                # Fallback to seed if somehow took too long (unlikely)
                return list(range(91, 181))
                
            return solution
        else:
            # If solver fails, fallback to a greedy heuristic or seed
            # Greedy: start with all, remove elements involved in most violations
            # Or just return seed
            return list(range(91, 181))
            
    except Exception:
        # Fallback to seed if any error occurs
        return list(range(91, 181))
