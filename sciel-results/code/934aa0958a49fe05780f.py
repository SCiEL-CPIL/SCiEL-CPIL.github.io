import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 55  # 5 seconds safety margin
    
    # We suspect the optimal set is in the upper range.
    # Let's try L=40.
    L = 40
    R = 100
    elements = list(range(L, R + 1))
    n = len(elements)
    idx_map = {val: i for i, val in enumerate(elements)}
    
    # Generate constraints: subsets summing to 200
    # Max size k_max such that min sum <= 200
    # min sum of k elements is k*L + k*(k-1)/2
    max_k = 1
    while max_k < n:
        min_sum = max_k * L + max_k * (max_k - 1) // 2
        if min_sum > 200:
            max_k -= 1
            break
        max_k += 1
    # max_k is the largest k with min_sum <= 200
    # We need to check k from 3 to max_k (since pairs can't sum to 200)
    
    constraints = []
    
    # Generate all combinations of size 3 to max_k
    for k in range(3, max_k + 1):
        for comb in itertools.combinations(elements, k):
            if sum(comb) == 200:
                # Create constraint row
                row = [0] * n
                for val in comb:
                    row[idx_map[val]] = 1
                constraints.append(row)
        
        # Time check
        if time.time() - start_time > timeout / 2:
            break
            
    if time.time() - start_time > timeout:
        # Fallback to greedy or previous best if generation takes too long
        # But generation for L=40 should be fast
        pass

    # If no constraints found (unlikely), return all
    if not constraints:
        return elements

    # Build ILP
    # Maximize sum(x)
    # c = -1 for all (minimize -sum(x))
    c = -np.ones(n)
    
    A_ub = np.array(constraints)
    b_ub = np.ones(len(constraints)) * 2  # Sum <= 2
    
    # x_i in {0, 1}
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    try:
        res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # res.x is the solution
            selected_indices = np.where(res.x > 0.5)[0]
            selected_values = [elements[i] for i in selected_indices]
            return selected_values
        else:
            # Fallback to greedy if ILP fails
            selected, reachable = [], {0}
            for x in range(100, 0, -1):
                new = {s + x for s in reachable if s + x <= 200}
                if 200 not in new:
                    selected.append(x)
                    reachable |= new
            return selected
    except Exception:
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
