import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import random

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  a = bc / (b+c)
    # We iterate over pairs (b, c) with b < c to generate potential a.
    # If a is an integer, distinct from b and c, and in [1, N], it forms a triple.
    
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            denom = b + c
            num = b * c
            if num % denom == 0:
                a = num // denom
                if 1 <= a <= N and a != b and a != c:
                    # Store sorted tuple to avoid duplicates
                    triple = tuple(sorted((a, b, c)))
                    triples.add(triple)
    
    # If no triples, return all numbers
    if not triples:
        return numbers

    # 2. Formulate ILP
    n_vars = N
    # Objective: maximize sum z_i -> minimize -sum z_i
    c_obj = -np.ones(n_vars)
    
    # Constraints: z_a + z_b + z_c <= 2 for each triple
    # A is sparse or dense? n_vars=180, triples ~ few thousand. Dense is fine.
    A = np.zeros((len(triples), n_vars))
    for i, (a, b, c) in enumerate(triples):
        A[i, a-1] = 1
        A[i, b-1] = 1
        A[i, c-1] = 1
    b_ub = np.full(len(triples), 2)
    constraint = LinearConstraint(A, ub=b_ub)
        
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)
    
    best_set = []
    
    try:
        # Use a time limit to ensure we return something
        res = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 45} # Leave 10s for overhead and fallback
        )
        
        if res.status == 0 or res.status == 1: # Optimal or feasible found
            if res.x is not None and res.x.size > 0:
                selected = [i + 1 for i in range(n_vars) if res.x[i] > 0.5]
                if len(selected) > 0:
                    best_set = selected
        else:
            # If ILP failed or timed out without a good solution, use fallback
            pass
            
    except Exception:
        pass

    # 3. Fallback Greedy Heuristic (from current program)
    # This ensures we always return a valid set, even if ILP fails.
    
    # Pre-compute conflicts for greedy
    conflicts = {i: set() for i in range(1, N + 1)}
    for t in triples:
        a, b, c = t
        conflicts[a].add((b, c))
        conflicts[b].add((a, c))
        conflicts[c].add((a, b))
    
    def greedy_construct():
        S = set()
        # Sort by degree (number of conflicts) ascending (pick safe nodes first)
        sorted_nodes = sorted(range(1, N + 1), key=lambda x: len(conflicts[x]))
        
        for v in sorted_nodes:
            # Check if v conflicts with any existing pair in S
            is_safe = True
            for u, w in conflicts[v]:
                if u in S and w in S:
                    is_safe = False
                    break
            if is_safe:
                S.add(v)
        return S
    
    # Run greedy to get a baseline
    greedy_set = greedy_construct()
    
    # If ILP didn't find anything, or found something smaller than greedy, use greedy
    if not best_set:
        best_set = list(greedy_set)
    else:
        # Verify ILP solution
        best_set_set = set(best_set)
        valid = True
        for a, b, c in triples:
            if a in best_set_set and b in best_set_set and c in best_set_set:
                valid = False
                break
        
        if not valid:
            best_set = list(greedy_set)
        elif len(greedy_set) > len(best_set):
            best_set = list(greedy_set)

    # Ensure non-empty
    if not best_set:
        best_set = [1]

    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    S = set(res)
    valid = True
    N = 180
    for a in S:
        for b in S:
            if a == b: continue
            for c in S:
                if a == c or b == c: continue
                if 1.0/a == 1.0/b + 1.0/c:
                    valid = False
                    print(f"Found conflict: {a}, {b}, {c}")
    print("Valid:", valid)
