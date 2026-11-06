import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85  # 5s buffer before 90s kill

    # Pre-compute all 4-APs and for each element, the "conflict triples"
    # A conflict triple for element x is a set of 3 elements {a,b,c} such that
    # if all of {a,b,c} are in the set, adding x would create a 4-AP.
    # Actually, it's simpler: for each 4-AP (a,b,c,d), if we add x to the set,
    # it creates a 4-AP iff the other 3 elements of that AP are already in the set.
    # So for each element x, we want a list of "masks" or "triples".
    
    # Let's use a different representation for speed.
    # in_set is a Python integer (bitset). Bit i is 1 if i is in the set.
    # For each element x, precompute a list of bitmasks M such that
    # (in_set & M) == M means adding x is forbidden.
    # M is the bitmask of the other 3 elements in the 4-AP.
    
    conflict_masks = [[] for _ in range(N)]
    
    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(0, max_a):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap = [a, b, c, e]
            # For each element in the AP, the conflict is the other 3
            for i in range(4):
                x = ap[i]
                others = [ap[j] for j in range(4) if j != i]
                mask = 0
                for y in others:
                    mask |= (1 << y)
                conflict_masks[x].append(mask)

    def can_add(x, bitset):
        for mask in conflict_masks[x]:
            if (bitset & mask) == mask:
                return False
        return True

    def add_element(x, bitset):
        bitset |= (1 << x)
        return bitset

    def remove_element(x, bitset):
        bitset &= ~(1 << x)
        return bitset

    def get_size(bitset):
        return bin(bitset).count('1')

    # 1. Initial Greedy Passes
    # We want to find a large initial set.
    # Greedy: iterate through elements in random order, add if possible.
    
    best_bitset = 0
    best_size = 0
    
    # Run many greedy restarts
    # Time budget for greedy: ~10s
    greedy_deadline = start_time + 10
    
    while time.time() < greedy_deadline:
        bitset = 0
        candidates = list(range(N))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, bitset):
                bitset |= (1 << x)
        
        size = get_size(bitset)
        if size > best_size:
            best_size = size
            best_bitset = bitset

    # 2. Local Search (Hill Climbing with Kicks)
    current_bitset = best_bitset
    current_size = best_size
    
    # Helper to get list of elements in bitset
    def get_elements(bitset):
        return [i for i in range(N) if (bitset >> i) & 1]

    # Helper to get list of elements NOT in bitset
    def get_complement(bitset):
        return [i for i in range(N) if not ((bitset >> i) & 1)]

    kick_sizes = [1, 2, 3, 4]
    
    # Time budget for local search: remaining time
    while time.time() < deadline:
        # Try to improve current set
        # 1. Greedy fill: try to add any available element
        improved = True
        while improved:
            improved = False
            # Random order for diversity
            candidates = get_complement(current_bitset)
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, current_bitset):
                    current_bitset |= (1 << x)
                    current_size += 1
                    improved = True
        
        # If we improved, update best
        if current_size > best_size:
            best_size = current_size
            best_bitset = current_bitset
            
        # 2. Kick: remove k random elements
        k = random.choice(kick_sizes)
        current_elements = get_elements(current_bitset)
        
        if len(current_elements) < k:
            continue
            
        removed = random.sample(current_elements, k)
        
        # Save state
        old_bitset = current_bitset
        old_size = current_size
        
        # Remove
        for x in removed:
            current_bitset &= ~(1 << x)
            current_size -= 1
            
        # Try to add back / fill
        # Greedy fill again
        improved = True
        while improved:
            improved = False
            candidates = get_complement(current_bitset)
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, current_bitset):
                    current_bitset |= (1 << x)
                    current_size += 1
                    improved = True
        
        # Accept if better or equal?
        # If better, keep. If equal, keep (to explore). If worse, revert.
        # Actually, if we revert on equal, we might get stuck.
        # Let's accept if current_size >= old_size.
        if current_size < old_size:
            current_bitset = old_bitset
            current_size = old_size
        else:
            # Update best if improved
            if current_size > best_size:
                best_size = current_size
                best_bitset = current_bitset

    # Final polish on the best bitset found
    final_bitset = best_bitset
    improved = True
    while improved:
        improved = False
        candidates = get_complement(final_bitset)
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, final_bitset):
                final_bitset |= (1 << x)
                improved = True

    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if (final_bitset >> i) & 1]
    return sorted(result)
