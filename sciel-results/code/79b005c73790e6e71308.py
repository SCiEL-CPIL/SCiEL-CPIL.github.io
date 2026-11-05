import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Check Paley Tournaments for small primes
    # Primes p % 4 == 3
    paley_candidates = [31, 43, 59]
    
    for p in paley_candidates:
        if time.time() > deadline:
            break
        # Construct Paley
        squares = set((x * x) % p for x in range(1, p))
        in_masks = [0] * p
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    # i -> j
                    in_masks[j] |= (1 << i)
                    orientation.append(0)
                else:
                    # j -> i
                    in_masks[i] |= (1 << j)
                    orientation.append(1)
        
        # Validate
        valid = True
        # Optimization: Check all 4-sets
        # For p=31, C(31,4) is small
        for quad in itertools.combinations(range(p), 4):
            if time.time() > deadline:
                valid = False
                break
            if (in_masks[quad[0]] & in_masks[quad[1]] & in_masks[quad[2]] & in_masks[quad[3]]) == 0:
                valid = False
                break
                
        if valid:
            return {'n': p, 'orientation': orientation}

    # 2. Local Search for n in [31, 66]
    # We will try to find a valid tournament for n starting from 31 upwards.
    # To save time, we can try a few n values with limited time each.
    
    # Helper to check validity
    def is_valid(in_masks, n):
        for quad in itertools.combinations(range(n), 4):
            if (in_masks[quad[0]] & in_masks[quad[1]] & in_masks[quad[2]] & in_masks[quad[3]]) == 0:
                return False
        return True

    # Helper to count bad sets (for optimization)
    def count_bad_sets(in_masks, n):
        bad_count = 0
        for quad in itertools.combinations(range(n), 4):
            if (in_masks[quad[0]] & in_masks[quad[1]] & in_masks[quad[2]] & in_masks[quad[3]]) == 0:
                bad_count += 1
        return bad_count

    # Local search function
    def local_search(n, time_limit):
        # Initialize random tournament
        in_masks = [0] * n
        orientation = [0] * (n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    # i -> j
                    in_masks[j] |= (1 << i)
                    orientation[idx] = 0
                else:
                    # j -> i
                    in_masks[i] |= (1 << j)
                    orientation[idx] = 1
                idx += 1
        
        # Precompute 4-sets containing each vertex
        # sets_with[v] = list of tuples (a,b,c,d) containing v
        # Actually, we just need to iterate over all 4-sets containing i or j when flipping (i,j)
        # To make it faster, we can precompute the list of all 4-sets and store them as tuples of indices.
        all_quads = list(itertools.combinations(range(n), 4))
        
        # Map quad to index? Or just iterate?
        # Iterating over all_quads is O(C(n,4)).
        # For n=50, C(50,4) ~ 230k.
        # If we check all 230k sets per flip, it's slow.
        # We need to only check sets containing i or j.
        
        # Precompute: for each vertex v, list of quads containing v.
        quads_containing = [[] for _ in range(n)]
        for q in all_quads:
            for v in q:
                quads_containing[v].append(q)
        
        # Current bad count
        bad_count = count_bad_sets(in_masks, n)
        
        # If already valid, return
        if bad_count == 0:
            return in_masks, orientation, True

        # Local search loop
        start_ls = time.time()
        temperature = 1.0
        cooling = 0.99
        
        # Greedy hill climbing with random restarts or simulated annealing
        # Let's try simple hill climbing with random restarts if stuck.
        
        best_bad_count = bad_count
        best_in_masks = in_masks[:]
        best_orientation = orientation[:]
        
        while time.time() - start_ls < time_limit:
            # Pick random edge (i, j)
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Determine current direction
            # If bit i is set in in_masks[j], then i -> j.
            # If bit j is set in in_masks[i], then j -> i.
            
            # Flip edge
            # We need to update in_masks[i] and in_masks[j]
            
            # Save old masks
            old_in_i = in_masks[i]
            old_in_j = in_masks[j]
            
            # Flip
            if (in_masks[j] >> i) & 1:
                # i -> j currently. Flip to j -> i.
                # Remove i from In(j), add j to In(i)
                in_masks[j] ^= (1 << i)
                in_masks[i] ^= (1 << j)
            else:
                # j -> i currently. Flip to i -> j.
                # Remove j from In(i), add i to In(j)
                in_masks[i] ^= (1 << j)
                in_masks[j] ^= (1 << i)
                
            # Calculate delta in bad_count
            # Only quads containing i or j are affected.
            # Union of quads_containing[i] and quads_containing[j]
            # We need to avoid double counting if a quad contains both i and j.
            
            # Create a set of quads to check?
            # For n=50, quads_containing[i] has ~ C(49,3) = 18k items.
            # Union is ~ 30k items.
            # Checking 30k intersections is fast.
            
            # To avoid double counting, we can just iterate over the union.
            # Since we just need the delta, we can:
            # 1. Count bad quads in the union BEFORE flip (we have old masks, but we already flipped... 
            #    So we should calculate delta by checking status before and after for each quad in union).
            
            # Let's revert masks to calculate "before" status?
            # Or just store the status of each quad in the union before the flip.
            
            # Efficient way:
            # 1. Identify unique quads in union of quads_containing[i] and quads_containing[j].
            # 2. For each quad, check if it was bad before (using old masks) and is bad after (using new masks).
            
            # Since we already flipped in_masks, we need to use old_in_i and old_in_j for "before" check.
            # But the quad might contain i or j, so we need to substitute the mask.
            
            # This is getting complicated to optimize perfectly in Python.
            # Let's try a simpler approach: 
            # Just re-evaluate the whole bad_count if n is small (<= 40).
            # For n=31, C(31,4) = 31k. Re-evaluating 31k sets is ~ 0.01s.
            # We can do ~100 flips per second.
            # For n=50, C(50,4) = 230k. Re-evaluating is ~ 0.1s.
            # We can do ~10 flips per second.
            
            # Given the time limit, let's use the "re-evaluate all" strategy for n <= 45.
            # For n > 45, it might be too slow. But we start from n=31.
            
            # Let's implement the "re-evaluate all" version for simplicity and correctness.
            new_bad_count = count_bad_sets(in_masks, n)
            
            if new_bad_count < bad_count:
                bad_count = new_bad_count
                if bad_count == 0:
                    return in_masks, orientation, True
                # Update orientation
                # Find index of (i,j) in orientation
                # i < j or j < i?
                # In orientation list, index is determined by lexicographic order.
                # If i < j, index is sum_{k=0}^{i-1} (n-k) + (j-i-1) ?
                # Let's just rebuild orientation from in_masks at the end if valid.
                # Or update it now.
                if i > j:
                    i, j = j, i
                # Calculate index
                # Number of pairs (a,b) with a < b and (a,b) < (i,j)
                # Pairs starting with k < i: sum_{k=0}^{i-1} (n - k - 1)
                # Pairs starting with i: j - i - 1
                idx = 0
                for k in range(i):
                    idx += n - k - 1
                idx += j - i - 1
                # Flip bit in orientation
                orientation[idx] ^= 1
            else:
                # Revert
                in_masks[i] = old_in_i
                in_masks[j] = old_in_j

        # If not found, return best found (though it's invalid)
        return best_in_masks, best_orientation, False

    # Try n from 31 to 66
    # We allocate time based on n.
    # Smaller n is easier, so we can spend more time there?
    # Or just try each n for 2 seconds.
    
    # To maximize chance, let's try n=31, 32, ..., 40 with 2s each.
    # If none work, try 41..50 with 2s each.
    
    for n in range(31, 67):
        if time.time() > deadline - 5:
            break
            
        # Allow more time for smaller n?
        # Let's just give 1.5 seconds to each n.
        time_limit = 1.5
        
        in_masks, orientation, found = local_search(n, time_limit)
        
        if found:
            # Re-construct orientation properly to be safe
            # The local search updated orientation, but let's verify or rebuild.
            # Rebuilding is safer.
            final_orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (in_masks[j] >> i) & 1:
                        final_orientation.append(0) # i -> j
                    else:
                        final_orientation.append(1) # j -> i
                        
            # Final validation
            if is_valid(in_masks, n):
                return {'n': n, 'orientation': final_orientation}
            
    # Fallback to 67 (Paley)
    p = 67
    squares = set((x * x) % p for x in range(1, p))
    orientation = []
    in_masks = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                in_masks[j] |= (1 << i)
                orientation.append(0)
            else:
                in_masks[i] |= (1 << j)
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}
