import time
import numpy as np
from math import lcm, gcd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def construct():
    start_time = time.time()
    # Hard deadline to avoid timeout kill (90s limit, use 60s margin)
    deadline = 60.0
    
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Greedy fallback function
    def greedy_solve():
        selected = []
        # Sort by value ascending to prioritize larger reciprocals (1/x)
        for v in sorted(candidates):
            conflict = False
            for y in selected:
                if lcm(v, y) <= 256:
                    conflict = True
                    break
            if not conflict:
                selected.append(v)
        return selected

    try:
        # Build conflict graph
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            vi = candidates[i]
            for j in range(i + 1, n):
                vj = candidates[j]
                g = gcd(vi, vj)
                l = vi * vj // g
                if l <= 256:
                    row_idx = len(rows)
                    rows.append(row_idx)
                    cols.append(i)
                    data.append(1)
                    cols.append(j)
                    data.append(1)
        
        m = len(rows)
        
        if m > 0:
            A = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
        else:
            A = sparse.csr_matrix((0, n))
        
        c = np.array([-1.0 / v for v in candidates])
        
        constraints = LinearConstraint(A, ub=np.ones(m))
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        
        # Solve MIP
        # time_limit in options is in seconds
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 45})
        
        if res.success and res.x is not None:
            x_sol = res.x
            selected = [candidates[i] for i in range(n) if x_sol[i] > 0.5]
            
            # Verify
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if lcm(selected[i], selected[j]) <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(selected) > 0:
                return selected
        
        # If MIP failed or returned invalid/empty, use greedy
        return greedy_solve()
        
    except Exception:
        return greedy_solve()
