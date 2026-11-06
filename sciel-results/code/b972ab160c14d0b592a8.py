import time
from math import gcd
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)
    
    # Number-theoretic reduction:
    # If we include x, we must exclude all multiples kx (k>=2) since lcm(x,kx)=kx<=256
    # This means: for each x, either include x OR include some of its multiples, but not both
    # Key insight: small numbers have high weight but many conflicts
    # 
    # Strategy: Identify "forced" choices
    # For prime p where p > 128: p is only multiple of itself, so no conflict with multiples
    # For composite numbers, we need to decide carefully
    #
    # Better approach: Use the fact that if x divides y, they conflict
    # So we can think in terms of "antichain" in divisibility poset
    #
    # Let's build a reduced problem:
    # 1. For each number, determine if it's "dominated" by a smaller number
    # 2. A number y is dominated by x if x|y and 1/x > 1/y (always true if x<y)
    #    But we can't just take all minimal elements because they might conflict with each other
    #
    # Actually, the constraint is lcm(a,b) > 256, not a|b
    # Two numbers conflict if lcm(a,b) <= 256
    #
    # Let's identify which numbers are "safe" to include together
    # Numbers > 128: any two distinct numbers > 128 have lcm > 256? 
    # No: 129 and 258? But 258 > 256, so only numbers in [2,256]
    # 130 and 260? 260 > 256
    # Actually, for a,b > 128: lcm(a,b) >= max(a,b) > 128, but could be <= 256
    # Example: 130 and 130? No, distinct
    # 130 and 260? 260 > 256
    # 144 and 192: lcm(144,192) = 576 > 256
    # 128 and 256: lcm = 256, so they conflict!
    # 128 and 192: lcm(128,192) = 768 > 256
    #
    # Let me compute conflicts more carefully
    #
    # Build conflict graph
    conflict = [[False]*n for _ in range(n)]
    for i in range(n):
        a = nodes[i]
        for j in range(i+1, n):
            b = nodes[j]
            g = gcd(a, b)
            lcm_ab = a * b // g
            if lcm_ab <= N:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Now use branch and bound with good pruning
    # Upper bound: sum of weights of all nodes that don't conflict with current selection
    # 
    # Better: Use the fact that the problem is equivalent to finding a maximum weight independent set
    # in the conflict graph
    #
    # Let's try a smarter MILP formulation with reduced variables
    # 
    # Observation: If x | y and x < y, then x and y conflict
    # So for each "chain" of divisibility, we can pick at most one
    #
    # Let's group numbers by their "primitive part" or use a different approach
    #
    # Simple approach: Solve the MILP but with better initial solution and tighter bounds
    #
    # First, get a good initial solution using greedy
    weights = np.array([1.0 / x for x in nodes])
    
    # Greedy: pick highest weight, remove conflicts, repeat
    remaining = set(range(n))
    greedy_solution = []
    greedy_value = 0.0
    
    while remaining:
        # Pick node with max weight in remaining
        best_idx = max(remaining, key=lambda i: weights[i])
        greedy_solution.append(best_idx)
        greedy_value += weights[best_idx]
        
        # Remove best_idx and all its conflicts from remaining
        to_remove = {best_idx}
        for j in remaining:
            if conflict[best_idx][j]:
                to_remove.add(j)
        remaining -= to_remove
    
    # Now solve MILP with this as initial solution
    # But scipy milp doesn't accept initial solution directly
    # So let's just solve the MILP
    #
    # To make it faster, let's reduce the problem
    # 
    # Key insight: Numbers that are multiples of 2 conflict with 2
    # Numbers that are multiples of 3 conflict with 3
    # etc.
    #
    # Let's identify which numbers are "free" (don't conflict with small numbers)
    # 
    # Actually, let's just solve the MILP with time limit and hope it works
    # But first, let's try to reduce the number of variables
    #
    # Reduction: If a number x is such that all its multiples are also in the set,
    # and x has higher weight than any of its multiples, then including x is always better
    # than including any of its multiples
    #
    # But this is not quite right because multiples might not conflict with each other
    #
    # Let's try a different reduction:
    # For each odd number x, its even multiples 2x, 4x, etc. all conflict with x
    # But 2x and 4x also conflict with each other (since 2x | 4x)
    #
    # I think the best approach is to just solve the MILP with good parameters
    #
    # Build constraint matrix
    A_rows = []
    b_vals = []
    
    for i in range(n):
        for j in range(i+1, n):
            if conflict[i][j]:
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_vals.append(1.0)
    
    if not A_rows:
        return nodes
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    lin_cons = LinearConstraint(A, ub=b)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    c = -weights
    
    try:
        res = milp(
            c=c,
            constraints=lin_cons,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 45.0, "mip_rel_gap": 1e-10}
        )
        
        if res.status == 0 and res.x is not None:
            x = res.x
            selected_indices = np.where(x > 0.5)[0]
            if len(selected_indices) == 0:
                return [nodes[i] for i in greedy_solution]
            
            result = [nodes[i] for i in selected_indices]
            
            # Verify
            valid = True
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    a, b = result[i], result[j]
                    g = gcd(a, b)
                    if a * b // g <= N:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check if better than greedy
                result_value = sum(1.0/x for x in result)
                if result_value >= greedy_value - 1e-10:
                    return result
            
            return [nodes[i] for i in greedy_solution]
        else:
            return [nodes[i] for i in greedy_solution]
            
    except Exception as e:
        return [nodes[i] for i in greedy_solution]

if __name__ == "__main__":
    construct()
