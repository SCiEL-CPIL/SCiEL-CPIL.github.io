import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute conflicts for each element x
    # conflicts[x] is a list of bitmasks. Each mask has bits set for y and z 
    # such that (x, y, z) is a bad triple (gcd(x,y)==gcd(y,z)==gcd(z,x)).
    # We assume x < y < z for the triple to avoid duplicates in generation,
    # but for checking, we just need to know if y and z are in S.
    # Actually, for a given x, y and z can be any distinct elements != x.
    # It's easier to generate all bad triples (a,b,c) with a<b<c, and for each,
    # add the mask for (b,c) to conflicts[a], (a,c) to conflicts[b], and (a,b) to conflicts[c].
    
    conflicts = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcds[a][b]
            for c in range(b + 1, N + 1):
                gbc = gcds[b][c]
                if gab == gbc:
                    gca = gcds[c][a]
                    if gca == gab:
                        # Triple (a, b, c) is bad
                        mask_bc = (1 << b) | (1 << c)
                        mask_ac = (1 << a) | (1 << c)
                        mask_ab = (1 << a) | (1 << b)
                        
                        conflicts[a].append(mask_bc)
                        conflicts[b].append(mask_ac)
                        conflicts[c].append(mask_ab)

    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    best_set_mask = 0
    best_size = 0
    
    def can_add(S_mask, x):
        """Check if x can be added to S_mask without creating a bad triple."""
        # x is 1-indexed. Bit x-1 represents x? 
        # Let's use bit index = value - 1 to keep numbers small? 
        # N=128, so 128 bits fits in a Python int easily.
        # Let's stick to bit index = value. So bit 0 is unused, bit 1 is 1, ..., bit 128 is 128.
        # Python ints are arbitrary precision, so this is fine.
        
        for mask in conflicts[x]:
            # Check if both bits corresponding to the other two elements are set in S_mask
            if (S_mask & mask) == mask:
                return False
        return True

    def local_search(S_mask):
        """Improve S_mask by adding/removing elements."""
        S = S_mask
        current_size = bin(S).count("1")
        
        # Phase 1: Try to add elements
        # Iterate through elements not in S
        not_in_S = [x for x in range(1, N + 1) if not (S & (1 << x))]
        random.shuffle(not_in_S)
        
        for x in not_in_S:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S |= (1 << x)
                current_size += 1
                
        # Phase 2: Try to remove elements to allow adding more (simple swap heuristic)
        # This is a basic local search. We can do a few iterations of:
        # 1. Pick a random element y in S to remove.
        # 2. Try to add any element x not in S.
        # 3. If we added x, keep the change. If not, revert.
        
        for _ in range(50): # Limit iterations per local search
            if time.time() > deadline:
                break
            
            # Pick a random element to potentially remove
            # Get list of elements in S
            elems_in_S = [x for x in range(1, N + 1) if (S & (1 << x))]
            if not elems_in_S:
                break
            
            y = random.choice(elems_in_S)
            S_temp = S & ~(1 << y)
            
            # Try to add something
            not_in_S_temp = [x for x in range(1, N + 1) if not (S_temp & (1 << x))]
            random.shuffle(not_in_S_temp)
            
            added = False
            for x in not_in_S_temp:
                if can_add(S_temp, x):
                    S_temp |= (1 << x)
                    added = True
                    break
            
            if added:
                S = S_temp
                current_size = bin(S).count("1")
            # If not added, we just wasted time removing y. 
            # To be more efficient, we should only remove if it enables an addition.
            # The above logic does that: if no x was added, S remains unchanged (we didn't assign S_temp to S).
            # Wait, the logic above: if added is False, S is not updated. So y was effectively "not removed".
            # This is correct.

        return S, current_size

    # Run multiple restarts with different initial sets
    # Strategy 1: Start with empty set and greedily add
    # Strategy 2: Start with a random subset and repair/optimize
    
    for run in range(100):
        if time.time() > deadline - 5:
            break
            
        if run % 2 == 0:
            # Greedy construction
            S_mask = 0
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S_mask, x):
                    S_mask |= (1 << x)
        else:
            # Random start: pick ~50 random elements, then repair by removing conflicts
            S_mask = 0
            for _ in range(50):
                x = random.randint(1, N)
                S_mask |= (1 << x)
            
            # Repair: Remove elements involved in conflicts
            # We need to find a valid subset of this random set.
            # Simple approach: iterate through elements in S_mask, if it's part of a conflict, remove it?
            # Or just use local search which handles additions, but we need to handle removals first.
            # Let's do a greedy repair:
            S_repaired = 0
            elems = [x for x in range(1, N + 1) if (S_mask & (1 << x))]
            random.shuffle(elems)
            for x in elems:
                if can_add(S_repaired, x):
                    S_repaired |= (1 << x)
            S_mask = S_repaired

        S_mask, size = local_search(S_mask)
        
        if size > best_size:
            best_size = size
            best_set_mask = S_mask

    # Convert best_set_mask to list
    result = [x for x in range(1, N + 1) if (best_set_mask & (1 << x))]
    
    # Final verification just in case
    # If result is empty, return [1] as fallback (though 1 is usually safe? 
    # gcd(1, a) = 1, gcd(1, b) = 1, gcd(a, b) might not be 1. 
    # If gcd(a,b)=1, then (1,a,b) is a bad triple. 
    # So [1] is valid only if no other elements are present. 
    # Since we maximize cardinality, we assume best_size > 0.)
    
    if not result:
        return [1]
        
    return result
