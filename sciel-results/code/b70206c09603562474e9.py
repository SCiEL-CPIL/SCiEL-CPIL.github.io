import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
import sys

def construct():
    n = 24
    conflicts = []
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    for a in range(1, 25):
        target = recips[a]
        # J must be subset of {a+1, ..., 24}
        candidates = list(range(a + 1, 25))
        
        if not candidates:
            continue
            
        # Precompute suffix sums for pruning
        # candidates are sorted ascending, so 1/j are descending
        suffix_sums = [0] * (len(candidates) + 1)
        for i in range(len(candidates) - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + recips[candidates[i]]
            
        def dfs(start_idx, remaining, current_J):
            if remaining == 0:
                if current_J:
                    conflicts.append((a, tuple(current_J)))
                return
            if remaining < 0:
                return
            if suffix_sums[start_idx] < remaining:
                return
                
            for i in range(start_idx, len(candidates)):
                j = candidates[i]
                term = recips[j]
                if term > remaining:
                    continue
                current_J.append(j)
                dfs(i + 1, remaining - term, current_J)
                current_J.pop()
                
        dfs(0, target, [])
        
    unique_constraints = set()
    for a, J in conflicts:
        coeffs = [0] * n
        coeffs[a-1] = 1
        for j in J:
            coeffs[j-1] += 1
        unique_constraints.add(tuple(coeffs))
        
    if not unique_constraints:
        return list(range(1, 25))
        
    A_ub_list = []
    b_ub_list = []
    for coeffs in unique_constraints:
        A_ub_list.append(coeffs)
        b_ub_list.append(sum(coeffs) - 1)
        
    A_ub = np.array(A_ub_list, dtype=float)
    b_ub = np.array(b_ub_list, dtype=float)
    
    c = np.ones(n) * -1
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        selected = [i+1 for i in range(n) if x[i] > 0.5]
        if selected:
            return selected
        else:
            return [1]
    else:
        return list(range(13, 25))
