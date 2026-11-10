"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute strict subset relation for all pairs? No, too big (4096x4096).
    # We compute d(x, y) on the fly using bit operations.
    # d(x, y) = 1 if x is strict subset of y or y is strict subset of x.
    # x < y (subset) iff (x & y) == x and x != y.
    
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    def is_comparable(a, b):
        if a == b: return 0
        return 1 if (is_strict_subset(a, b) or is_strict_subset(b, a)) else 0
    
    # 1. Initialization: Greedy construction
    # Start with empty set? Or a chain?
    # A chain of length 12: {0}, {0,1}, ..., {0..11}
    # Then add sets that are comparable to the most existing sets.
    
    current = []
    current_set = set()
    
    # Add the chain
    # Chain: 0, 1, 3, 7, ..., 2^12-1 (using masks for prefixes? No, just any chain)
    # Let's use the chain of adding bit 0, then 1, etc.
    # mask: 1, 3, 7, 15, ...
    chain_mask = 0
    for i in range(n):
        chain_mask |= (1 << i)
        current.append(chain_mask)
        current_set.add(chain_mask)
        
    # Greedily add remaining m - n sets
    # We need to pick x not in current_set to maximize sum_{y in current} is_comparable(x, y)
    
    # To speed up greedy addition, we can just iterate all masks and pick best.
    # O(total_masks * len(current)) per addition.
    # 4096 * 40 * 28 additions ~ 4.5M ops. Fast enough.
    
    while len(current) < m:
        if time.time() > deadline - 10: break # Reserve time for local search
        
        best_x = -1
        best_score = -1
        for x in range(total_masks):
            if x in current_set: continue
            score = 0
            for y in current:
                score += is_comparable(x, y)
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x != -1:
            current.append(best_x)
            current_set.add(best_x)
        else:
            break
            
    # Convert to list of lists for return, but keep masks for search
    # We will do local search on masks, then convert at the end.
    
    # 2. Local Search
    # Maintain deg_C[y] = sum_{z in current} is_comparable(y, z) for all y in range(total_masks)
    
    deg = [0] * total_masks
    
    # Initialize deg
    for y in range(total_masks):
        s = 0
        for z in current:
            s += is_comparable(y, z)
        deg[y] = s
        
    # Calculate initial score
    current_score = 0
    for i in range(len(current)):
        for j in range(i+1, len(current)):
            current_score += is_comparable(current[i], current[j])
            
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Find best swap
        # For each i in current, try swapping with x not in current
        # Delta = deg[x] - is_comparable(x, current[i]) - deg[current[i]]
        # Note: deg[current[i]] includes is_comparable(current[i], current[i]) which is 0.
        # But deg[x] includes is_comparable(x, current[i])? No, x is not in current.
        # Wait, deg[x] = sum_{z in current} is_comparable(x, z).
        # The term is_comparable(x, current[i]) is part of deg[x].
        # When we remove current[i], the contribution of current[i] to deg[x] is is_comparable(x, current[i]).
        # So sum_{z in current \ {current[i]}} is_comparable(x, z) = deg[x] - is_comparable(x, current[i]).
        
        # The contribution of current[i] to the rest is sum_{z in current \ {current[i]}} is_comparable(current[i], z) = deg[current[i]].
        
        # So Delta = (deg[x] - is_comparable(x, current[i])) - deg[current[i]]
        
        best_delta = 0
        best_i = -1
        best_x = -1
        
        # Optimization: Only consider x that might improve.
        # We iterate all x not in current.
        # To make this faster, we can just check all x.
        
        current_set = set(current)
        
        for i in range(len(current)):
            c_i = current[i]
            deg_c_i = deg[c_i]
            
            # If deg_c_i is very low, swapping might help?
            # We want to maximize Delta.
            # Max possible Delta for this i is max_x(deg[x] - is_comparable(x, c_i)) - deg_c_i
            # max_x(deg[x]) is bounded by len(current).
            
            for x in range(total_masks):
                if x in current_set: continue
                
                # Calculate delta
                # is_comparable(x, c_i)
                if x == c_i:
                    comp_x_c = 0
                else:
                    comp_x_c = 1 if ((x & c_i) == x or (x & c_i) == c_i) else 0
                
                delta = (deg[x] - comp_x_c) - deg_c_i
                
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_x = x
                    
        if best_i != -1 and best_delta > 0:
            # Perform swap
            old_c = current[best_i]
            current[best_i] = best_x
            current_set.remove(old_c)
            current_set.add(best_x)
            
            # Update deg
            # deg[y] += is_comparable(y, best_x) - is_comparable(y, old_c)
            # This loop is O(total_masks)
            for y in range(total_masks):
                # Optimization: Only update if comparable status changes?
                # is_comparable(y, best_x) and is_comparable(y, old_c)
                
                comp_y_new = 0
                if y != best_x:
                    if (y & best_x) == y or (y & best_x) == best_x:
                        comp_y_new = 1
                        
                comp_y_old = 0
                if y != old_c:
                    if (y & old_c) == y or (y & old_c) == old_c:
                        comp_y_old = 1
                        
                diff = comp_y_new - comp_y_old
                if diff != 0:
                    deg[y] += diff
                    
            current_score += best_delta
            improved = True
            
    # Convert masks to lists of integers
    result = []
    for mask in current:
        # Ensure types are Python int
        lst = []
        for b in range(n):
            if (mask >> b) & 1:
                lst.append(b)
        result.append(lst)
        
    return result
# EVOLVE-BLOCK-END
