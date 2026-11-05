import time
import random
import itertools
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try to find a valid coloring for n=34 using ILP
    # If that fails, try smaller n
    
    for n in [34, 33, 32, 30]:
        if time.time() > deadline - 5.0:
            break
            
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        rows = []
        cols = []
        data = []
        b_lows = []
        b_ups = []
        
        # Red K4 constraints: sum(x) >= 1
        for combo in itertools.combinations(range(n), 4):
            t1 = (combo[0], combo[1], combo[2])
            t2 = (combo[0], combo[1], combo[3])
            t3 = (combo[0], combo[2], combo[3])
            t4 = (combo[1], combo[2], combo[3])
            
            i1 = triple_idx[t1]
            i2 = triple_idx[t2]
            i3 = triple_idx[t3]
            i4 = triple_idx[t4]
            
            row_idx = len(b_lows)
            rows.extend([row_idx] * 4)
            cols.extend([i1, i2, i3, i4])
            data.extend([1.0, 1.0, 1.0, 1.0])
            
            b_lows.append(1.0)
            b_ups.append(np.inf)
            
        # Blue K5 constraints: sum(x) <= 9
        for combo in itertools.combinations(range(n), 5):
            row_idx = len(b_lows)
            for t in itertools.combinations(combo, 3):
                i = triple_idx[t]
                rows.append(row_idx)
                cols.append(i)
                data.append(1.0)
            
            b_lows.append(-np.inf)
            b_ups.append(9.0)
        
        if not rows:
            continue
            
        A = csr_matrix((data, (rows, cols)), shape=(len(b_lows), num_triples))
        lb = np.array(b_lows)
        ub = np.array(b_ups)
        
        integrality = np.ones(num_triples)
        bounds = Bounds(lb=np.zeros(num_triples), ub=np.ones(num_triples))
        
        # Objective: Minimize 0
        c = np.zeros(num_triples)
        
        try:
            # HiGHS is the default solver. It's very fast.
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0: # Optimal
                x = res.x
                colors = [int(round(val)) for val in x]
                return {'n': n, 'colors': colors}
            else:
                continue
                
        except Exception as e:
            continue

    # Fallback: Return a small valid coloring
    # n=6 is small enough that we can brute force or use a simple heuristic.
    # But even n=1 is valid.
    # Let's try a quick heuristic for n=20 if time permits?
    # No, keep it simple and safe.
    return {'n': 1, 'colors': [0]}
