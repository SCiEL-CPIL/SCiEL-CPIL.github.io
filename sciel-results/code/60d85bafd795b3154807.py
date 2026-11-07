import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    N = 128
    
    # Generate all constraints (hyperedges)
    constraints = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            max_c = N - b
            if max_c <= b:
                continue
            for c in range(b + 1, max_c + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                vals = {a, b, c, s1, s2, s3}
                constraints.append(frozenset(vals))
                
    unique_constraints = list(set(constraints))
    
    # Precompute static weights
    elem_to_cons = {i: [] for i in range(1, N + 1)}
    for con in unique_constraints:
        for v in con:
            elem_to_cons[v].append(con)
            
    static_weight = {i: len(elem_to_cons[i]) for i in range(1, N + 1)}
    
    # Initialize S with all numbers
    S = set(range(1, N + 1))
    
    # Initial violated constraints
    # Since S contains all elements, all constraints are violated.
    violated_cons = list(unique_constraints)
    
    # Greedy Removal
    while violated_cons:
        if time.time() > deadline - 15:
            break
            
        # Pick a random violated constraint to avoid bias
        con = random.choice(violated_cons)
        
        # Pick element with highest static weight in this constraint
        best_elem = max(con, key=lambda x: static_weight[x])
        
        S.remove(best_elem)
        
        # Update violated_cons: remove all constraints containing best_elem
        # This is the expensive part, but O(V)
        violated_cons = [c for c in violated_cons if best_elem not in c]
        
    # Local Search: Add back
    def can_add(x, S):
        # Check if adding x to S creates a violation
        # Only need to check constraints containing x
        for con in elem_to_cons[x]:
            # If all other elements are in S, adding x violates it
            # Note: x is not in S yet
            if all(v in S for v in con if v != x):
                return False
        return True

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to add any element
        for x in range(1, N + 1):
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
                improved = True
        
        # If no simple addition, try swaps
        if not improved:
            S_list = list(S)
            not_S_list = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(S_list)
            random.shuffle(not_S_list)
            
            for x in not_S_list:
                if time.time() > deadline - 2:
                    break
                    
                # Find a constraint blocking x
                blocking_con = None
                for con in elem_to_cons[x]:
                    if all(v in S for v in con if v != x):
                        blocking_con = con
                        break
                        
                if blocking_con is None:
                    S.add(x)
                    improved = True
                    continue
                    
                # Try removing y from blocking_con
                for y in blocking_con:
                    if y == x:
                        continue
                        
                    S_temp = S - {y}
                    if can_add(x, S_temp):
                        S = S_temp
                        S.add(x)
                        improved = True
                        break
                
                if improved:
                    break # Restart outer loop
                    
    # Final pass
    for x in range(1, N + 1):
        if x not in S:
            if can_add(x, S):
                S.add(x)
                
    return [int(x) for x in S]
