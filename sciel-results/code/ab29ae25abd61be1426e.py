import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # For each element x, precompute which pairs (a,b) conflict with x
    # conflict_mask[x] is a bitmask where bit (a*128+b) is set if (a,b,x) is a bad triple
    # Actually, let's use: for each x, store a list of (a,b) pairs that conflict
    # But for O(1) check, we can precompute for each pair (a,b), which x's conflict
    # 
    # Better approach: For each x, precompute a set of "forbidden pairs" 
    # Then to check if x can be added to S, check if any forbidden pair is entirely in S
    # 
    # Even better: Use bitmask representation
    # Represent S as a bitmask (128 bits)
    # For each x, precompute a list of pairs (a,b) such that (x,a,b) is bad
    # To check validity: for each (a,b) in conflicts[x], check if both a and b are in S
    # This is O(conflicts[x]) which is fast
    
    # Precompute conflicts: conflicts[x] = list of (a,b) where a<b and (x,a,b) is bad triple
    conflicts = {x: [] for x in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(1, N + 1):
                if c != a and c != b:
                    if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                        # (a, b, c) is a bad triple
                        # Add (a,b) to conflicts[c], (a,c) to conflicts[b], (b,c) to conflicts[a]
                        pass
    
    # More efficient: iterate over all triples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    # (a, b, c) is a bad triple
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
    
    # Sort conflicts for each x by the pair for faster lookup
    for x in range(1, N + 1):
        conflicts[x].sort()
    
    # Bitmask helpers
    def bit_set(mask, x):
        return (mask >> (x - 1)) & 1
    
    def set_bit(mask, x):
        return mask | (1 << (x - 1))
    
    def clear_bit(mask, x):
        return mask & ~(1 << (x - 1))
    
    def can_add(mask, x):
        """Check if x can be added to the set represented by mask"""
        for (a, b) in conflicts[x]:
            if bit_set(mask, a) and bit_set(mask, b):
                return False
        return True
    
    def is_valid(mask):
        """Check if the set represented by mask is valid"""
        for x in range(1, N + 1):
            if bit_set(mask, x):
                # Check if x has any conflict pair entirely in mask
                for (a, b) in conflicts[x]:
                    if a < x and b < x and bit_set(mask, a) and bit_set(mask, b):
                        return False
        return True
    
    # Greedy construction
    def greedy_construct():
        mask = 0
        candidates = numbers[:]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(mask, x):
                mask = set_bit(mask, x)
        return mask
    
    # Exhaustive local search: try all 1-add, then all 1-remove-1-add, then all 1-remove-2-add
    def local_search(mask):
        current_mask = mask
        current_size = bin(current_mask).count('1')
        
        # Phase 1: Try to add any single element
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for x in range(1, N + 1):
                if not bit_set(current_mask, x) and can_add(current_mask, x):
                    current_mask = set_bit(current_mask, x)
                    current_size += 1
                    improved = True
                    break
        
        # Phase 2: Try 1-remove-1-add swaps
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try all possible swaps
            for u in range(1, N + 1):
                if not bit_set(current_mask, u):
                    continue
                for v in range(1, N + 1):
                    if bit_set(current_mask, v):
                        continue
                    # Try removing u and adding v
                    new_mask = clear_bit(current_mask, u)
                    new_mask = set_bit(new_mask, v)
                    # Check if v can be added to (current_mask - {u})
                    # We need to check conflicts of v against new_mask
                    valid = True
                    for (a, b) in conflicts[v]:
                        if bit_set(new_mask, a) and bit_set(new_mask, b):
                            valid = False
                            break
                    if valid:
                        current_mask = new_mask
                        improved = True
                        break
                if improved:
                    break
        
        # Phase 3: Try 1-remove-2-add moves
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try all possible 1-remove-2-add
            for u in range(1, N + 1):
                if not bit_set(current_mask, u):
                    continue
                for v in range(1, N + 1):
                    if bit_set(current_mask, v):
                        continue
                    for w in range(v + 1, N + 1):
                        if bit_set(current_mask, w):
                            continue
                        # Try removing u and adding v, w
                        new_mask = clear_bit(current_mask, u)
                        new_mask = set_bit(new_mask, v)
                        new_mask = set_bit(new_mask, w)
                        # Check validity: v and w must not create conflicts with new_mask
                        valid = True
                        for (a, b) in conflicts[v]:
                            if bit_set(new_mask, a) and bit_set(new_mask, b):
                                valid = False
                                break
                        if valid:
                            for (a, b) in conflicts[w]:
                                if bit_set(new_mask, a) and bit_set(new_mask, b):
                                    valid = False
                                    break
                        if valid:
                            current_mask = new_mask
                            current_size += 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current_mask
    
    best_mask = 0
    best_size = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        # Greedy
        mask = greedy_construct()
        # Local Search
        mask = local_search(mask)
        
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Convert best_mask to list
    result = []
    for x in range(1, N + 1):
        if bit_set(best_mask, x):
            result.append(x)
    
    return result if result else [1]

# EVOLVE-BLOCK-END
