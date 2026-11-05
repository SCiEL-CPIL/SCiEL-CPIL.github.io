import time
from math import lcm, gcd
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Weights: 1/x
    weights = np.array([1.0 / x for x in candidates])
    
    # Find conflicts
    # A conflict exists if lcm(a, b) <= 256
    # We store conflicts as pairs of indices
    conflicts = []
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            # Optimization: if a > 256, no conflict (but a <= 256 here)
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                conflicts.append((i, j))
                
    # Build MILP
    # Maximize sum w_i x_i
    # Subject to x_i + x_j <= 1 for all conflicts
    # x_i in {0, 1}
    
    # Number of constraints = len(conflicts)
    m = len(conflicts)
    
    if m == 0:
        return candidates # No conflicts, take all? No, weights are positive, take all.
        
    # Objective function c (minimize -c)
    c = -weights
    
    # Constraint matrix A
    # A is m x n
    # For each conflict (i, j), row k has 1 at i and 1 at j
    rows = []
    cols = []
    data = []
    for k, (i, j) in enumerate(conflicts):
        rows.append(k)
        cols.append(i)
        data.append(1)
        
        rows.append(k)
        cols.append(j)
        data.append(1)
        
    A = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    
    # Bounds for variables: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all binary
    integrality = np.ones(n)
    
    # Linear constraints: A x <= m
    # milp uses (lb, ub) for each row
    # Here we want A x <= 1 for all rows
    # So lb = -inf, ub = 1
    lin_cons = LinearConstraint(A, lb=-np.inf, ub=np.ones(m))
    
    # Check time
    if time.time() > deadline:
        # Fallback to heuristic
        return heuristic_fallback(candidates, conflicts, weights)

    try:
        res = milp(c, constraints=lin_cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            x = res.x
            selected = [candidates[i] for i in range(n) if x[i] > 0.5]
            return selected
        elif res.status == 1: # Feasible but not optimal (maybe time limit?)
            x = res.x
            selected = [candidates[i] for i in range(n) if x[i] > 0.5]
            # Ensure validity
            if is_valid(selected, candidates):
                return selected
            else:
                return heuristic_fallback(candidates, conflicts, weights)
        else:
            return heuristic_fallback(candidates, conflicts, weights)
            
    except Exception as e:
        return heuristic_fallback(candidates, conflicts, weights)

def is_valid(selected, candidates):
    # Check if selected is a valid independent set
    # selected is a list of values
    # We need to check pairs
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            a = selected[i]
            b = selected[j]
            g = gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                return False
    return True

def heuristic_fallback(candidates, conflicts, weights):
    # Greedy + Local Search
    # This mimics the 1.071 score strategy
    # 1. Greedy init
    # Sort by weight desc (1/x desc => x asc? No, 1/2 > 1/3, so smaller x has higher weight)
    # Wait, maximize sum 1/x. So we want small numbers.
    # Sort candidates by 1/x descending => x ascending.
    sorted_cands = sorted(candidates, key=lambda x: -1.0/x)
    
    selected = []
    # Map value to index in original list? Or just work with values.
    # Let's work with indices in 'candidates' list for speed, or just values.
    # Using values is easier for the final return.
    
    # Greedy
    for val in sorted_cands:
        compatible = True
        for other in selected:
            g = gcd(val, other)
            l = (val // g) * other
            if l <= 256:
                compatible = False
                break
        if compatible:
            selected.append(val)
            
    # Local Search: 1-swap, 2-swap
    # We need to efficiently check compatibility.
    # Precompute conflict set?
    # conflicts is list of (idx_i, idx_j)
    # Let's create a set of conflicting pairs of values for O(1) lookup?
    # Or a dict value -> set of conflicting values.
    conflict_map = {}
    for i, j in conflicts:
        vi = candidates[i]
        vj = candidates[j]
        if vi not in conflict_map: conflict_map[vi] = set()
        if vj not in conflict_map: conflict_map[vj] = set()
        conflict_map[vi].add(vj)
        conflict_map[vj].add(vi)
        
    def is_compatible(val, S):
        for other in S:
            if other in conflict_map.get(val, set()):
                return False
        return True
        
    improved = True
    while improved:
        improved = False
        current_sum = sum(1.0/v for v in selected)
        
        # 1. Try adding any unselected number
        for val in candidates:
            if val in selected: continue
            if is_compatible(val, selected):
                # Check if adding increases sum (it always does, weight > 0)
                # But wait, if we add, we might block future swaps?
                # In independent set, adding a node is always good if compatible.
                # However, if we add a low weight node, it might block a high weight node later?
                # No, we are doing local search on the current set.
                # If we can add it, we should.
                selected.append(val)
                improved = True
                break # Restart loop to check for more additions
                
        if improved: continue
        
        # 2. Try 1-swap: remove u, add v
        for u in list(selected):
            S_without_u = [x for x in selected if x != u]
            for v in candidates:
                if v in selected: continue
                if is_compatible(v, S_without_u):
                    # Compare sum
                    # Old: sum(S)
                    # New: sum(S) - 1/u + 1/v
                    if (1.0/v) > (1.0/u):
                        selected.remove(u)
                        selected.append(v)
                        improved = True
                        break
            if improved: break
            
        if improved: continue
        
        # 3. Try 2-swap: remove u, add v, w
        # This is more expensive O(N^3) roughly
        # Optimize: Only try swaps that might improve sum.
        for u in list(selected):
            S_without_u = [x for x in selected if x != u]
            # Find all v compatible with S_without_u
            candidates_v = []
            for v in candidates:
                if v in selected: continue
                if is_compatible(v, S_without_u):
                    candidates_v.append(v)
            
            # Try pairs v, w from candidates_v
            for i in range(len(candidates_v)):
                v = candidates_v[i]
                for j in range(i+1, len(candidates_v)):
                    w = candidates_v[j]
                    # Check if v and w are compatible with each other
                    if w in conflict_map.get(v, set()):
                        continue
                    
                    # Check improvement
                    # Old: sum(S)
                    # New: sum(S) - 1/u + 1/v + 1/w
                    if (1.0/v + 1.0/w) > (1.0/u):
                        selected.remove(u)
                        selected.append(v)
                        selected.append(w)
                        improved = True
                        break
                if improved: break
            if improved: break
            
        # 4. Try 1-remove, 2-add is covered. What about 2-remove, 3-add?
        # Given time constraints, 2-swap is usually sufficient for this size.
        
    return selected
