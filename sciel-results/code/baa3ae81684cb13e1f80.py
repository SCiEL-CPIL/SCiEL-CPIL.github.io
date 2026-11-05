"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 4, use Local Search / Simulated Annealing on the permutation order
    # 1. Generate standard recursive superpermutation
    def standard_construction(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    std_word = standard_construction(n)
    
    # 2. Extract the sequence of permutations in the order they first appear
    all_perms = set(permutations(range(1, n+1)))
    perm_list = []
    seen = set()
    for i in range(len(std_word) - n + 1):
        p = tuple(std_word[i:i+n])
        if p in all_perms and p not in seen:
            seen.add(p)
            perm_list.append(p)
            
    # It's possible we missed some if the standard word logic was weird, 
    # but it should contain all.
    if len(perm_list) < len(all_perms):
        # Fallback: just use the standard word
        return "".join(map(str, std_word))
        
    # 3. Define overlap function
    def get_overlap(a, b):
        # a, b are tuples of length n
        # Max overlap is n-1
        for k in range(n-1, -1, -1):
            if k == 0:
                return 0
            if a[n-k:] == b[:k]:
                return k
        return 0

    # 4. Calculate total length for a given order
    def calc_length(order):
        if not order:
            return 0
        length = n
        for i in range(len(order) - 1):
            length += n - get_overlap(order[i], order[i+1])
        return length

    current_order = perm_list
    current_len = calc_length(current_order)
    best_order = current_order[:]
    best_len = current_len
    
    # 5. Local Search / Simulated Annealing
    # Precompute overlaps? 
    # Overlap depends on the pair. 
    # We can cache overlaps for pairs that appear frequently? 
    # Or just compute on the fly. N=5040, O(n) per overlap is fast enough.
    
    # Optimization: Precompute a map from permutation to index?
    # Not strictly needed for the swap logic, but useful for reconstruction.
    
    # Simulated Annealing Parameters
    # We have ~110 seconds.
    # Each step is O(1) overlap calcs (3 pairs involved in swap).
    # We can do ~10^6 to 10^7 steps.
    
    T = 1.0
    cooling_rate = 0.9999
    min_T = 0.001
    
    # To make it faster, we can pre-compute overlaps for adjacent pairs in the current order
    # and update them incrementally.
    
    # Let's use a simpler Hill Climbing first, then add SA if time permits.
    # Actually, SA is better for escaping local minima.
    
    # Incremental update of length:
    # Let L be the current total length.
    # L = n + sum_{i=0}^{N-2} (n - ov[i])
    # where ov[i] = overlap(order[i], order[i+1])
    
    ov = [0] * (len(current_order) - 1)
    for i in range(len(ov)):
        ov[i] = get_overlap(current_order[i], current_order[i+1])
    
    current_len = n
    for o in ov:
        current_len += n - o
        
    # Verify current_len matches calc_length
    # assert current_len == calc_length(current_order)
    
    best_len = current_len
    best_order = current_order[:]
    
    steps = 0
    while time.time() < deadline - 5:
        # Pick a random adjacent pair to swap
        i = random.randint(0, len(current_order) - 2)
        
        # Calculate delta for swapping i and i+1
        # Affected overlaps: 
        # Left: (i-1, i) and (i, i+1) and (i+1, i+2)
        # New:   (i-1, i+1) and (i+1, i) and (i, i+2)
        
        # Old costs:
        old_cost = 0
        if i > 0:
            old_cost += n - ov[i-1] # overlap(i-1, i)
        old_cost += n - ov[i]       # overlap(i, i+1)
        if i < len(ov) - 1:
            old_cost += n - ov[i+1] # overlap(i+1, i+2)
            
        # New costs:
        new_cost = 0
        if i > 0:
            new_cost += n - get_overlap(current_order[i-1], current_order[i+1])
        new_cost += n - get_overlap(current_order[i+1], current_order[i])
        if i < len(ov) - 1:
            new_cost += n - get_overlap(current_order[i], current_order[i+2])
            
        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            # Swap in order
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            
            # Update overlaps
            if i > 0:
                ov[i-1] = get_overlap(current_order[i-1], current_order[i])
            ov[i] = get_overlap(current_order[i], current_order[i+1])
            if i < len(ov) - 1:
                ov[i+1] = get_overlap(current_order[i+1], current_order[i+2])
                
            current_len += delta
            
            if current_len < best_len:
                best_len = current_len
                best_order = current_order[:]
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
        steps += 1

    # Reconstruct word from best_order
    final_word = list(best_order[0])
    for i in range(1, len(best_order)):
        p = best_order[i]
        # Find max overlap with suffix of final_word
        # Since we know the overlap from the search, we could store it, 
        # but it's safer to recompute for the final reconstruction.
        # Or we can just use the max overlap function.
        max_ov = 0
        for k in range(n-1, -1, -1):
            if k == 0: break
            if final_word[n-k:] == list(p[:k]):
                max_ov = k
                break
        final_word.extend(p[max_ov:])
        
    return "".join(map(str, final_word))
# EVOLVE-BLOCK-END
