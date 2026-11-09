"""Combined program: fast greedy initialization + efficient local search with deadline guard."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s limit
    
    n = 256
    
    # Precompute all 3-APs for fast lookup
    # ap_conflicts[x] = set of pairs (a,b) such that {a,x,b} or {x,a,b} or {a,b,x} forms a 3-AP
    # Better: for each x, precompute which other elements conflict with x
    # conflict[x] = set of y such that x and y are in some 3-AP together (with some third element)
    
    # Actually, let's precompute: for each pair (a, b), what is the third element c that would form a 3-AP?
    # If a < b: c = 2b - a (if c > b and c <= n) or c = a - (b-a) = 2a-b (if c < a and c >= 1)
    # So for each pair (a,b) with a < b, the potential 3-APs are:
    #   (a, b, 2b-a) if 2b-a <= n
    #   (2a-b, a, b) if 2a-b >= 1
    
    # Precompute: for each x, the set of "forbidden partners" - elements y such that 
    # adding both x and y to an empty set would create a 3-AP. But that's not quite right.
    # 
    # Better approach: precompute for each x, and for each potential third element z,
    # what pairs (y1, y2) would form a 3-AP with z? This is getting complex.
    #
    # Simplest fast approach: maintain a list of elements in S, and for can_add,
    # iterate over that list (size K ~ 50-80) instead of range(1, n+1) (size 256).
    
    def can_add_fast(membership, elements_list, x):
        """Check if adding x to set (elements_list) maintains AP-free property.
        elements_list is a list of elements currently in S.
        membership is a boolean array for O(1) lookup of the third element."""
        for y in elements_list:
            # Case 1: x is the middle. y and z=2x-y form AP with x as middle.
            z = 2*x - y
            if z != y and z != x and 1 <= z <= n and membership[z]:
                return False
            # Case 2: x is an end, y is the middle. Other end = 2y-x
            z = 2*y - x
            if z != y and z != x and 1 <= z <= n and membership[z]:
                return False
        return True
    
    # Greedy initialization: add smallest elements first (maximizes reciprocal sum)
    S = set()
    membership = np.zeros(n+1, dtype=bool)
    elements_list = []
    for x in range(1, n + 1):
        if can_add_fast(membership, elements_list, x):
            S.add(x)
            membership[x] = True
            elements_list.append(x)
    
    best_S = set(S)
    best_val = sum(1.0 / x for x in S)
    
    # Local search with fast can_add
    while time.time() < deadline:
        improved = False
        excluded = set(range(1, n + 1)) - S
        
        # Phase 1: Try adding each excluded element (prioritize small x for max reciprocal)
        for x in sorted(excluded, key=lambda v: -1.0 / v):
            if time.time() > deadline:
                break
            if can_add_fast(membership, elements_list, x):
                new_val = best_val + 1.0 / x
                if new_val > best_val:
                    S.add(x)
                    membership[x] = True
                    elements_list.append(x)
                    best_S = set(S)
                    best_val = new_val
                    improved = True
                    excluded = set(range(1, n + 1)) - S
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where v < u for improvement)
        if not improved:
            S_list = sorted(S, reverse=True)  # Try removing largest first
            excluded_list = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                # Temporarily remove u
                membership[u] = False
                elements_list.remove(u)
                temp_val = best_val - 1.0 / u
                
                found_swap = False
                for v in excluded_list:
                    if v == u:
                        continue
                    if v >= u:
                        break  # We want v < u for improvement
                    if can_add_fast(membership, elements_list, v):
                        new_val = temp_val + 1.0 / v
                        if new_val > best_val + 1e-15:
                            S.remove(u)
                            S.add(v)
                            membership[u] = False
                            membership[v] = True
                            elements_list.append(v)
                            best_S = set(S)
                            best_val = new_val
                            improved = True
                            found_swap = True
                            break  # Restart swap search with new S
                
                if not found_swap:
                    # Restore u
                    membership[u] = True
                    elements_list.append(u)
                
                if improved:
                    break
        
        if not improved:
            # Phase 3: Try pair swaps (remove u1, u2, add v1, v2)
            if time.time() > deadline - 5.0:
                break
            
            S_list = sorted(S, reverse=True)
            excluded_list = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            found_improvement = False
            if len(S_list) >= 2:
                u1, u2 = S_list[0], S_list[1]
                # Temporarily remove u1, u2
                membership[u1] = False
                membership[u2] = False
                elements_list.remove(u1)
                elements_list.remove(u2)
                temp_val = best_val - 1.0 / u1 - 1.0 / u2
                
                # Try adding pairs of small excluded elements
                for i, v1 in enumerate(excluded_list[:30]):
                    if v1 == u1 or v1 == u2:
                        continue
                    if can_add_fast(membership, elements_list, v1):
                        membership[v1] = True
                        elements_list.append(v1)
                        for v2 in excluded_list[i+1:30]:
                            if v2 == u1 or v2 == u2 or v2 == v1:
                                continue
                            if can_add_fast(membership, elements_list, v2):
                                new_val = temp_val + 1.0 / v1 + 1.0 / v2
                                if new_val > best_val + 1e-15:
                                    S.discard(u1)
                                    S.discard(u2)
                                    S.add(v1)
                                    S.add(v2)
                                    membership[v1] = True
                                    membership[v2] = True
                                    elements_list.append(v2)
                                    best_S = set(S)
                                    best_val = new_val
                                    found_improvement = True
                                    break
                        # Restore v1 if no improvement found with it
                        if not found_improvement:
                            membership[v1] = False
                            elements_list.remove(v1)
                    if found_improvement:
                        break
                
                if not found_improvement:
                    # Restore u1, u2
                    membership[u1] = True
                    membership[u2] = True
                    elements_list.append(u1)
                    elements_list.append(u2)
            
            if found_improvement:
                continue
            else:
                break
    
    return list(best_S)
