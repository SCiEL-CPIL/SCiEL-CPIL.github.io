"""Construct the maximum pairwise noncoprime subset of [1,255] containing 255."""

import time
from math import gcd

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Candidate set: numbers in [1,255] divisible by at least one prime factor of 255
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    idx255 = candidates.index(255)
    
    # Build incompatibility list: pairs (i,j) where gcd(candidates[i], candidates[j]) == 1
    # We need to exclude at least one from each incompatible pair
    incompatible = []
    for i in range(n):
        for j in range(i+1, n):
            if gcd(candidates[i], candidates[j]) == 1:
                incompatible.append((i, j))
    
    # Fallback: multiples of 3 (size 85)
    fallback = [x for x in range(1, 256) if x % 3 == 0]
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Variables: x_i in {0,1} for each candidate
        # Objective: maximize sum(x_i)
        c = -np.ones(n)  # minimize -sum(x_i)
        
        # Constraints:
        # 1. x_idx255 = 1 (must include 255)
        # 2. For each incompatible pair (i,j): x_i + x_j <= 1
        
        A = []
        b_lower = []
        b_upper = []
        
        # Constraint: x_255 >= 1 and x_255 <= 1
        row = np.zeros(n)
        row[idx255] = 1.0
        A.append(row)
        b_lower.append(1.0)
        b_upper.append(1.0)
        
        # Constraints: x_i + x_j <= 1 for incompatible pairs
        for (i, j) in incompatible:
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A.append(row)
            b_lower.append(0.0)
            b_upper.append(1.0)
        
        A = np.array(A)
        b_lower = np.array(b_lower)
        b_upper = np.array(b_upper)
        
        integrality = np.ones(n)  # all binary
        bounds = Bounds(lb=0.0, ub=1.0)
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            selected = [candidates[i] for i in range(n) if result.x[i] > 0.5]
            if 255 in selected and len(selected) > len(fallback):
                # Verify validity
                valid = True
                for i in range(len(selected)):
                    for j in range(i+1, len(selected)):
                        if gcd(selected[i], selected[j]) == 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return selected
        
        # If MILP didn't improve, try greedy heuristic
        # Start with all candidates, remove conflicts
        # Actually, let's try a simple local search: start from multiples of 3, try adding elements
        current_set = set(fallback)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in candidates:
                if x in current_set:
                    continue
                # Check if x can be added
                can_add = True
                for y in current_set:
                    if gcd(x, y) == 1:
                        can_add = False
                        break
                if can_add:
                    current_set.add(x)
                    improved = True
        
        if len(current_set) > len(fallback):
            return list(current_set)
        
    except Exception:
        pass
    
    return fallback
