import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
import sys

def construct():
    n = 24
    conflicts = []
    seen = set()
    
    # For each a in 1..23, find subsets J of {a+1..24} such that sum(1/j for j in J) == 1/a
    for a in range(1, 24):
        target = Fraction(1, a)
        elements = list(range(a + 1, 25))
        
        # Helper to find all subsets with exact sum
        # We use DFS with pruning
        # Max size estimate: if we pick smallest elements, sum grows.
        # If sum of smallest k > target, stop.
        # If sum of largest k < target, we need more elements (but limited by range).
        
        found_subsets = []
        
        def dfs(start, current_sum, current_set):
            # Pruning: if current_sum > target, return
            if current_sum > target:
                return
            
            # If current_sum == target, record
            if current_sum == target and current_set:
                found_subsets.append(tuple(current_set))
                # We can continue to find larger sets?
                # Yes, e.g. 1/2 = 1/3+1/6 = 1/4+1/5+1/20?
                # 1/4+1/5+1/20 = 5/20+4/20+1/20 = 10/20 = 1/2.
                # So multiple subsets can sum to same target.
                # We need ALL of them to create constraints.
            
            # Try adding more elements
            for i in range(start, len(elements)):
                j = elements[i]
                new_sum = current_sum + Fraction(1, j)
                # Pruning: even if we add all remaining smallest, can we reach target?
                # Actually, just adding j and recursing is fine for small N.
                # But we can prune if new_sum > target.
                if new_sum > target:
                    break # Since elements are sorted, further elements are larger, sum will be even larger
                # Also, if current_sum + sum of all remaining from i+1 < target, we might need to go deeper
                # But simple recursion is okay for N=24.
                
                dfs(i + 1, new_sum, current_set + [j])
        
        dfs(0, Fraction(0, 1), [])
        
        # Add constraints for each found subset
        for J in found_subsets:
            # Constraint: x_a + sum(x_j for j in J) <= len(J)
            coeffs = [0] * 24
            coeffs[a - 1] = 1
            for j in J:
                coeffs[j - 1] += 1
            
            key = tuple(coeffs)
            if key not in seen:
                seen.add(key)
                conflicts.append((coeffs, len(J)))
                
    if not conflicts:
        return list(range(1, 25))
        
    # Build ILP
    c = np.ones(n) * -1
    A_ub = np.array([c for c, _ in conflicts], dtype=float)
    b_ub = np.array([s for _, s in conflicts], dtype=float)
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        selected = [i + 1 for i in range(n) if x[i] > 0.5]
        if selected:
            return selected
        else:
            return [1]
    else:
        # Fallback: Greedy selection of largest numbers
        # As analyzed, large numbers have fewer conflicts.
        # Let's try to pick as many large numbers as possible.
        # Start with all, remove conflicts?
        # Or just return a known good set.
        # The set {13..24} has size 12.
        # Can we do better?
        # If a>=2 has no conflicts with >a, then {10..24} might be valid?
        # Check a=10. J subset {11..24}. Sum 1/10.
        # 1/11+1/12+1/13+1/14+1/15+1/16+1/17+1/18+1/19+1/20+1/21+1/22+1/23+1/24
        # This sum is approx H_24 - H_10 approx 3.77 - 2.93 = 0.84.
        # We need sum 0.1.
        # 1/11+1/12 = 0.15 > 0.1.
        # 1/11+1/12+... is too big.
        # We need small sum.
        # 1/20+1/21+... small.
        # Is there a subset of {11..24} summing to 1/10?
        # 1/11+1/12+1/13+1/14+1/15+1/16+1/17+1/18+1/19+1/20 = 0.15+...
        # Let's rely on the ILP. If it fails, return range(13,25).
        return list(range(13, 25))
