"""Strategy: Maximum Independent Set via ILP (scipy.optimize.milp) with variable reduction and heuristic fallback."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (evaluator kills at 90s, but we want to be safe)
    
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
    
    # 2. Identify safe elements (not in any triple)
    involved = set()
    for t in triples_list:
        involved.update(t)
    
    safe_elements = [i for i in range(1, N + 1) if i not in involved]
    
    # 3. Setup ILP on the reduced variable set (only involved elements)
    v_list = sorted(involved)
    n_vars = len(v_list)
    v_index = {v: i for i, v in enumerate(v_list)}
    
    if n_vars == 0:
        return safe_elements # All are safe
    
    # Objective: maximize sum x_i -> minimize -sum x_i
    c_obj = -np.ones(n_vars)
    
    # Constraints: sum(x_a + x_b + x_c) <= 2 for each triple
    A = np.zeros((num_triples, n_vars))
    b_ub = np.full(num_triples, 2)
    
    for i, t in enumerate(triples_list):
        # t contains 1-based indices
        for val in t:
            A[i, v_index[val]] = 1
    
    constraints = LinearConstraint(A, ub=b_ub)
    integrality = np.ones(n_vars) # All variables are integers
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    best_solution = None
    
    try:
        # Run MILP with a time limit
        # We use a slightly tighter time limit to leave room for verification
        time_limit = 45.0
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": time_limit})
        
        if res.status == 0 or res.status == 3: # 0: optimal, 3: time limit reached with incumbent
            x = res.x
            # Round to nearest integer (should be 0 or 1)
            x_int = np.round(x).astype(int)
            
            # Construct set from involved elements
            selected_involved = set()
            for i in range(n_vars):
                if x_int[i] == 1:
                    selected_involved.add(v_list[i])
            
            # Combine with safe elements
            candidate_set = set(safe_elements) | selected_involved
            
            # Verify validity
            valid = True
            for t in triples_list:
                if t[0] in candidate_set and t[1] in candidate_set and t[2] in candidate_set:
                    valid = False
                    break
            
            if valid and len(candidate_set) > 0:
                best_solution = candidate_set
                
    except Exception as e:
        pass # Fall through to heuristic if ILP fails
    
    # If ILP didn't provide a valid solution, use heuristic
    if best_solution is None:
        current_set = set(numbers)
        
        # Precompute degrees for removal heuristic
        degree = {x: 0 for x in numbers}
        for t in triples_list:
            for x in t:
                degree[x] += 1
                
        # Greedy removal
        changed = True
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            for t in triples_list:
                if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                    candidates = [t[0], t[1], t[2]]
                    candidates.sort(key=lambda x: degree[x], reverse=True)
                    to_remove = candidates[0]
                    current_set.remove(to_remove)
                    changed = True
                    break
                    
        best_solution = current_set.copy()
        removed = set(numbers) - best_solution
        
        # Local Search: Try to add back elements
        # Simple pass
        for x in list(removed):
            if time.time() > deadline - 2.0:
                break
            conflict = False
            for t in triples_list:
                if x in t:
                    others = [v for v in t if v != x]
                    if others[0] in best_solution and others[1] in best_solution:
                        conflict = True
                        break
            if not conflict:
                best_solution.add(x)
                removed.remove(x)
                
    # Final safety check
    if not best_solution:
        return safe_elements if safe_elements else [1]
        
    return list(best_solution)
