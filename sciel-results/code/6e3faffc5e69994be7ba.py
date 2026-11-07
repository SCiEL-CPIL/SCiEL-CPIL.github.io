import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leave 5s margin
    
    best_n = 0
    best_set = []
    
    def try_greedy(fam):
        nonlocal best_n, best_set
        if time.time() > deadline:
            return
        
        current_set = []
        # bits[k] is a bitmask where bit 'sum' is 1 if 'sum' is achievable with exactly k elements
        bits = [0] * 61  # Max size 60, so cardinality 0..60
        bits[0] = 1  # Sum 0 is achievable with 0 elements
        all_bits = 1  # Union of all bits
        
        used = set()
        
        for x in fam:
            if time.time() > deadline:
                break
            if x in used:
                continue
            
            # Pre-compute the "others" mask: all bits NOT in the target cardinality k+1
            # For each k, we need to check if (bits[k] << x) intersects with sums in cardinalities != k+1
            # We can compute: others_for_k = all_bits & ~bits[k+1]
            # But we need to check for ALL k simultaneously
            
            # Optimized: For each k, new_sums = bits[k] << x
            # Collision if new_sums & (all_bits & ~bits[k+1]) != 0
            # Equivalently: new_sums & all_bits & ~bits[k+1] != 0
            # Which is: (bits[k] << x) & all_bits & ~bits[k+1] != 0
            
            valid = True
            for k in range(len(current_set), -1, -1):
                if bits[k] == 0:
                    continue
                new_sums = bits[k] << x
                others = all_bits & ~bits[k + 1]
                if new_sums & others:
                    valid = False
                    break
            
            if valid:
                # Add x to the set
                new_all_bits = all_bits
                for k in range(len(current_set), -1, -1):
                    if bits[k]:
                        shifted = bits[k] << x
                        bits[k + 1] |= shifted
                        new_all_bits |= shifted
                all_bits = new_all_bits
                current_set.append(x)
                used.add(x)
                
                if len(current_set) > best_n:
                    best_n = len(current_set)
                    best_set = current_set[:]
                
                if best_n >= 60:
                    break
    
    # Try descending order first (largest numbers give better separation)
    fam_desc = list(range(256, 0, -1))
    try_greedy(fam_desc)
    
    # Try ascending order
    if best_n < 60 and time.time() < deadline:
        fam_asc = list(range(1, 257))
        try_greedy(fam_asc)
    
    # Try various structured orderings
    if best_n < 60 and time.time() < deadline:
        # Even then odd
        fam = list(range(2, 257, 2)) + list(range(1, 257, 2))
        try_greedy(fam)
    
    if best_n < 60 and time.time() < deadline:
        # Odd then even
        fam = list(range(1, 257, 2)) + list(range(2, 257, 2))
        try_greedy(fam)
    
    # Try random shuffles
    if best_n < 60 and time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline:
                break
            fam = list(range(1, 257))
            random.shuffle(fam)
            try_greedy(fam)
    
    # Local search: try to replace elements with larger unused ones
    if best_n > 0 and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current = best_set[:]
            used = set(current)
            
            # Rebuild bits for current set
            bits = [0] * (best_n + 1)
            bits[0] = 1
            all_bits = 1
            for x in current:
                for k in range(best_n - 1, -1, -1):
                    if bits[k]:
                        bits[k + 1] |= bits[k] << x
                        all_bits |= bits[k] << x
            
            # Try replacing each element with a larger unused one
            for i in range(best_n):
                if time.time() > deadline - 2:
                    break
                old_x = current[i]
                # Try replacing with larger values
                for new_x in range(old_x + 1, 257):
                    if new_x in used:
                        continue
                    
                    # Remove old_x, try adding new_x
                    # This is complex; instead, try building a new set from scratch
                    # with all elements except old_x, plus new_x
                    candidate_set = [x for x in current if x != old_x] + [new_x]
                    candidate_set.sort(reverse=True)
                    
                    # Validate
                    bits2 = [0] * (best_n + 1)
                    bits2[0] = 1
                    all_bits2 = 1
                    valid = True
                    for x in candidate_set:
                        for k in range(best_n - 1, -1, -1):
                            if bits2[k]:
                                shifted = bits2[k] << x
                                others = all_bits2 & ~bits2[k + 1]
                                if shifted & others:
                                    valid = False
                                    break
                                bits2[k + 1] |= shifted
                                all_bits2 |= shifted
                        if not valid:
                            break
                    
                    if valid:
                        best_set = candidate_set
                        improved = True
                        break
            
            if improved:
                # Update best_set and continue
                pass
    
    # Ensure we return a valid list of ints
    result = [int(x) for x in best_set]
    return result
