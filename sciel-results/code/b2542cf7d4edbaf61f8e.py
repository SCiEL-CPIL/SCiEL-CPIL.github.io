"""Strategy: Maximum Independent Set via ILP (scipy.optimize.milp) on a 3-uniform hypergraph."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted([a, b, c]))
                    triples.add(triple)
    
    triples_list = list(triples)
    num_triples = len(triples_list)
    
    if time.time() > deadline:
        return numbers # Fallback, though shouldn't happen this fast
    
    # 2. Setup ILP
    # Variables x_i for i in 1..180. Index 0 corresponds to 1, etc.
    # Minimize -sum(x_i)
    c = -np.ones(N)
    
    # Constraints: sum(x_a + x_b + x_c) <= 2 for each triple
    # A is num_triples x N
    A = np.zeros((num_triples, N))
    b_ub = np.full(num_triples, 2)
    
    for i, t in enumerate(triples_list):
        # t contains 1-based indices
        A[i, t[0]-1] = 1
        A[i, t[1]-1] = 1
        A[i, t[2]-1] = 1
    
    constraints = LinearConstraint(A, ub=b_ub)
    integrality = np.ones(N) # All variables are integers
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 45})
        
        if res.status == 0 or res.status == 3: # 0: optimal, 3: time limit reached with incumbent
            x = res.x
            # Round to nearest integer (should be 0 or 1)
            x_int = np.round(x).astype(int)
            
            # Construct set from 1-based indices
            selected = set()
            for i in range(N):
                if x_int[i] == 1:
                    selected.add(i + 1)
            
            # Verify validity
            valid = True
            for t in triples_list:
                if t[0] in selected and t[1] in selected and t[2] in selected:
                    valid = False
                    break
            
            if valid and len(selected) > 0:
                return list(selected)
                
    except Exception as e:
        pass # Fall through to heuristic if ILP fails
    
    # Fallback Heuristic: Greedy Removal + Local Search
    # This is similar to the original but slightly improved with randomization
    current_set = set(numbers)
    
    # Precompute degrees for removal heuristic
    degree = {x: 0 for x in numbers}
    for t in triples_list:
        for x in t:
            degree[x] += 1
            
    # Greedy removal
    changed = True
    while changed:
        changed = False
        for t in triples_list:
            if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                candidates = [t[0], t[1], t[2]]
                candidates.sort(key=lambda x: degree[x], reverse=True)
                to_remove = candidates[0]
                current_set.remove(to_remove)
                changed = True
                break
                
    best_set = current_set.copy()
    removed = set(numbers) - best_set
    
    # Local Search: Try to add back elements
    # Simple pass
    for x in list(removed):
        conflict = False
        for t in triples_list:
            if x in t:
                others = [v for v in t if v != x]
                if others[0] in best_set and others[1] in best_set:
                    conflict = True
                    break
        if not conflict:
            best_set.add(x)
            removed.remove(x)
            
    return list(best_set)
