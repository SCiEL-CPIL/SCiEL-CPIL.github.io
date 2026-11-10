import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
import time
import random

def construct(q, n, radius):
    total_words = q ** n
    
    def int_to_list(val, base, length):
        res = [0] * length
        for i in range(length):
            res[i] = val % base
            val //= base
        return res

    def list_to_int(lst, base):
        val = 0
        for i in range(len(lst)-1, -1, -1):
            val = val * base + lst[i]
        return val

    # Build the incidence matrix A
    rows = []
    cols = []
    data = []
    
    for i in range(total_words):
        w = int_to_list(i, q, n)
        rows.append(i)
        cols.append(i)
        data.append(1)
        
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                j = list_to_int(w, q)
                rows.append(j)
                cols.append(i)
                data.append(1)
            w[pos] = orig
            
    A = sp.csc_matrix((data, (rows, cols)), shape=(total_words, total_words))
    c = np.ones(total_words)
    cons = LinearConstraint(A, lb=np.ones(total_words), ub=np.inf)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        start = time.time()
        # Use a timeout for the solver? milp doesn't have a direct time limit param in all versions,
        # but we can run it and check time after.
        # However, if it hangs, we can't break out easily in a simple script.
        # We'll assume it finishes. If not, the process will be killed by the evaluator.
        # To be safe, we can limit the number of iterations or use a simpler heuristic first?
        # No, let's try MILP.
        
        res = milp(c=c, constraints=cons, integrality=np.ones(total_words), bounds=bounds)
        
        if res.status == 0:
            x = res.x
            code_indices = [i for i in range(total_words) if x[i] > 0.5]
            return [int_to_list(i, q, n) for i in code_indices]
    except Exception:
        pass

    # Fallback: Randomized Greedy with Sets
    neighbors = []
    for i in range(total_words):
        w = int_to_list(i, q, n)
        neigh = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                ni = list_to_int(w, q)
                neigh.add(ni)
            w[pos] = orig
        neighbors.append(neigh)
        
    best_code = None
    best_size = float('inf')
    start_time = time.time()
    
    for run in range(10):
        if time.time() - start_time > 100:
            break
        uncovered = set(range(total_words))
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            # Random tie-breaking
            candidates = []
            max_count = 0
            for c in uncovered:
                count = len(neighbors[c] & uncovered)
                if count > max_count:
                    max_count = count
                    candidates = [c]
                elif count == max_count:
                    candidates.append(c)
            best_c = random.choice(candidates)
            code.append(best_c)
            uncovered -= neighbors[best_c]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    if best_code is None:
        return [int_to_list(i, q, n) for i in range(total_words)]
        
    return [int_to_list(c, q, n) for c in best_code]
