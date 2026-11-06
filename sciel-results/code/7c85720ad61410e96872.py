import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
import sys

def construct():
    n = 24
    elements = list(range(2, 25))
    # Precompute reciprocals as Fractions
    recips = {i: Fraction(1, i) for i in elements}
    
    constraints_data = []
    seen = set()
    
    # For each target a in 1..24
    for a in range(1, 25):
        target = Fraction(1, a)
        # Available elements are 2..24 excluding a
        # If a is in 2..24, exclude it. If a=1, exclude nothing (but 1 is not in elements list anyway)
        available = [j for j in elements if j != a]
        
        # Sort available elements to help pruning? 
        # Smaller elements have larger reciprocals. 
        # We want to find subsets summing to target.
        # DFS with index
        def dfs(idx, current_sum, current_set):
            if current_sum == target:
                # Found a subset
                # Create constraint
                coeffs = [0] * n
                coeffs[a-1] = 1
                for j in current_set:
                    coeffs[j-1] += 1
                key = tuple(coeffs)
                if key not in seen:
                    seen.add(key)
                    constraints_data.append((coeffs, len(current_set)))
                return
            
            # Pruning: if current_sum > target, return
            if current_sum > target:
                return
            
            # If we've considered all elements, return
            if idx == len(available):
                return
            
            # Optimization: Calculate max possible sum from remaining elements
            # If current_sum + sum(recips[j] for j in available[idx:]) < target, return
            # This requires precomputing suffix sums of reciprocals for the available list.
            # But available list changes with 'a'. 
            # Given small n, simple pruning might suffice.
            
            # Option 1: Include available[idx]
            j = available[idx]
            new_sum = current_sum + recips[j]
            if new_sum <= target:
                dfs(idx + 1, new_sum, current_set + [j])
            
            # Option 2: Exclude available[idx]
            dfs(idx + 1, current_sum, current_set)
            
        dfs(0, Fraction(0, 1), [])

    if not constraints_data:
        # If no constraints, we can pick all? 
        # But 1/2 = 1/3 + 1/6 is a constraint. So there should be constraints.
        return list(range(1, 25))
    
    # Build ILP
    c = np.ones(n) * -1 # Minimize negative sum -> Maximize sum
    A = np.array([c for c, _ in constraints_data], dtype=float)
    b_ub = np.array([s for _, s in constraints_data], dtype=float)
    
    constraints = [LinearConstraint(A, -np.inf, b_ub)]
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        selected = [i+1 for i in range(n) if x[i] > 0.5]
        if selected:
            return selected
        else:
            return [1]
    else:
        # Fallback
        return list(range(13, 25))
