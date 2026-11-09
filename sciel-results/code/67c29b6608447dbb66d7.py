"""Combined program: fast greedy initialization + efficient local search with deadline guard."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s limit
    
    n = 256
    
    # Bitset-based AP-free check: represent set as a bitset for fast membership
    # can_add_bitset: check if x can be added to set represented by bitset
    def can_add_bitset(bitset, x):
        """Check if adding x to set (bitset) maintains AP-free property."""
        # Case 1: x is the middle term: need a, b in S with a + b = 2x, a != b
        # For each a in S, check if 2x - a is in S and != a
        # Equivalent: check if there exists a in S such that (2x - a) in S and a != (2x - a)
        # i.e., a != x (since a + (2x-a) = 2x, and we need a != 2x-a => a != x)
        # So: for each a in S with a != x, check if 2x - a is in S
        # But we can do this more efficiently with bitset operations
        
        # Case 2: x is an end term: need b, c in S with x + c = 2b or a + x = 2b
        # i.e., for each b in S, check if 2b - x is in S and 2b - x != b and 2b - x != x
        
        # Let's use direct iteration but with numpy for speed
        # Actually for n=256, let's use a different approach:
        # Precompute forbidden values
        
        # For x to be added, no 3-AP involving x should exist.
        # A 3-AP involving x: {x, y, z} where x+y=2z or x+z=2y or y+z=2x
        # Case 1: y+z=2x (x is middle). Need y,z in S, y!=z.
        # Case 2: x+z=2y (x is end, y is middle). Need y,z in S, y!=x, z!=x.
        # Case 3: x+y=2z (x is end, z is middle). Need y,z in S, y!=x, z!=x.
        
        # Case 1: x is middle
        # For each y in S, check if 2x-y is in S and 2x-y != y
        for y in range(1, n+1):
            if bitset[y]:
                z = 2*x - y
                if z != y and 1 <= z <= n and bitset[z]:
                    return False
        
        # Case 2 & 3: x is an end
        # For each y in S (as middle), check if 2y-x is in S and 2y-x != y and 2y-x != x
        for y in range(1, n+1):
            if bitset[y]:
                z = 2*y - x
                if z != y and z != x and 1 <= z <= n and bitset[z]:
                    return False
        
        return True
    
    # Greedy initialization: add smallest elements first (maximizes reciprocal sum)
    S = set()
    bitset = np.zeros(n+1, dtype=bool)
    for x in range(1, n + 1):
        if can_add_bitset(bitset, x):
            S.add(x)
            bitset[x] = True
    
    best_S = set(S)
    best_val = sum(1.0 / x for x in S)
    
    # Local search with bitset-based can_add
    # We'll use a temperature-based approach for more diverse exploration
    
    while time.time() < deadline:
        improved = False
        S_list = sorted(S)
        excluded = set(range(1, n + 1)) - S
        
        # Phase 1: Try adding each excluded element (prioritize small x)
        for x in sorted(excluded, key=lambda v: -1.0 / v):
            if time.time() > deadline:
                break
            if can_add_bitset(bitset, x):
                new_S = S | {x}
                new_val = best_val + 1.0 / x
                if new_val > best_val:
                    S = new_S
                    bitset[x] = True
                    best_S = set(S)
                    best_val = new_val
                    improved = True
                    excluded = set(range(1, n + 1)) - S
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where v < u)
        if not improved:
            S_list = sorted(S, reverse=True)  # Try removing largest first
            excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                S_temp = S - {u}
                bitset_temp = bitset.copy()
                bitset_temp[u] = False
                temp_val = best_val - 1.0 / u
                
                for v in excluded:
                    if v == u:
                        continue
                    if v >= u:
                        break  # We want v < u for improvement
                    if can_add_bitset(bitset_temp, v):
                        new_val = temp_val + 1.0 / v
                        if new_val > best_val + 1e-15:
                            S = S_temp | {v}
                            bitset = bitset_temp.copy()
                            bitset[v] = True
                            best_S = set(S)
                            best_val = new_val
                            improved = True
                            break  # Restart swap search with new S
                
                if improved:
                    break
        
        if not improved:
            # Phase 3: Try pair swaps (remove u1, u2, add v1, v2)
            # Only do this if we have time
            if time.time() > deadline - 5.0:
                break
            
            S_list = sorted(S, reverse=True)
            excluded = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            found_improvement = False
            # Try removing the two largest elements
            if len(S_list) >= 2:
                u1, u2 = S_list[0], S_list[1]
                S_temp = S - {u1} - {u2}
                bitset_temp = bitset.copy()
                bitset_temp[u1] = False
                bitset_temp[u2] = False
                temp_val = best_val - 1.0 / u1 - 1.0 / u2
                
                # Try adding pairs of small excluded elements
                for i, v1 in enumerate(excluded[:20]):  # Only try top 20 candidates
                    if v1 == u1 or v1 == u2:
                        continue
                    if can_add_bitset(bitset_temp, v1):
                        bitset_temp2 = bitset_temp.copy()
                        bitset_temp2[v1] = True
                        for v2 in excluded[i+1:20]:
                            if v2 == u1 or v2 == u2:
                                continue
                            if can_add_bitset(bitset_temp2, v2):
                                new_val = temp_val + 1.0 / v1 + 1.0 / v2
                                if new_val > best_val + 1e-15:
                                    S = S_temp | {v1, v2}
                                    bitset = bitset_temp2.copy()
                                    bitset[v2] = True
                                    best_S = set(S)
                                    best_val = new_val
                                    found_improvement = True
                                    break
                    if found_improvement:
                        break
            
            if found_improvement:
                continue  # Loop back to try more improvements
            else:
                break  # No improvement possible
    
    return list(best_S)
