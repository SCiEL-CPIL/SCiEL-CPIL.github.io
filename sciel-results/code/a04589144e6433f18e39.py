"""Strategy: Maximum Independent Set via ILP (scipy.optimize.milp) on a 3-uniform hypergraph, with greedy fallback."""
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  bc = a(b+c)  =>  a = bc / (b+c)
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
    
    # 2. Greedy removal to get an initial feasible solution
    # Precompute degree of each element in the hypergraph
    degree = {x: 0 for x in numbers}
    for t in triples_list:
        for x in t:
            degree[x] += 1
    
    current_set = set(numbers)
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
    
    # Try to add back elements
    removed = set(numbers) - current_set
    best_set = current_set.copy()
    
    # Local search: try to add back elements
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        current = best_set.copy()
        removed_list = list(removed)
        random.shuffle(removed_list)
        
        for x in removed_list:
            if time.time() > deadline - 5:
                break
            conflict = False
            for t in triples_list:
                if x in t:
                    others = [v for v in t if v != x]
                    if others[0] in current and others[1] in current:
                        conflict = True
                        break
            if not conflict:
                current.add(x)
        
        if len(current) > len(best_set):
            best_set = current
            removed = set(numbers) - best_set
    
    best_size = len(best_set)
    
    # 3. Try ILP to find the optimal or near-optimal solution
    # Variables: x_i for i in 1..180 (index i-1)
    # Maximize sum(x_i)
    # Subject to: for each triple (a,b,c), x_a + x_b + x_c <= 2
    #             x_i in {0, 1}
    
    try:
        # c vector: objective coefficients (all 1s for maximization)
        c = np.ones(N)
        
        # Constraints: for each triple, x_a + x_b + x_c <= 2
        # A matrix of shape (num_triples, N)
        A = np.zeros((num_triples, N))
        for idx, t in enumerate(triples_list):
            A[idx, t[0] - 1] = 1
            A[idx, t[1] - 1] = 1
            A[idx, t[2] - 1] = 1
        
        # Bounds: x_i in [0, 1]
        integrality = np.ones(N)  # all variables are integer
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        
        # Constraint: A @ x <= 2 for each row
        # LinearConstraint(A, lb=-np.inf, ub=2)
        lin_cons = LinearConstraint(A, lb=-np.inf, ub=2)
        
        # Run MILP
        milp_result = milp(
            c=c,
            constraints=[lin_cons],
            integrality=integrality,
            bounds=bounds
        )
        
        if milp_result.success and milp_result.x is not None:
            # Extract solution
            x = milp_result.x
            # Round to nearest integer (should already be 0 or 1)
            x_rounded = np.round(x).astype(int)
            
            # Build the set
            ilp_set = set()
            for i in range(N):
                if x_rounded[i] == 1:
                    ilp_set.add(i + 1)
            
            # Verify no forbidden triples
            valid = True
            for t in triples_list:
                if t[0] in ilp_set and t[1] in ilp_set and t[2] in ilp_set:
                    valid = False
                    break
            
            if valid and len(ilp_set) >= best_size:
                best_set = ilp_set
                best_size = len(ilp_set)
        
        # If ILP didn't find a better solution, use the greedy result
    except Exception as e:
        # Fall back to greedy result
        pass
    
    # Final verification
    for t in triples_list:
        if t[0] in best_set and t[1] in best_set and t[2] in best_set:
            # Should not happen, but just in case, remove one
            best_set.remove(t[0])
            break
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
