"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate initial code: Direct sum of ternary tetracode and all tails
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all tails of length n-4
    tail_len = n - 4
    tails = list(itertools.product(range(q), repeat=tail_len))
    
    # Construct initial codewords
    codewords = [h + list(tail) for h in tetracode for tail in tails]
    N = len(codewords)
    
    # 2. Precompute masks for each codeword
    # A codeword covers itself and all words at Hamming distance 1.
    # Total points in universe: q^n
    total_points = q ** n
    
    # Map codeword to index in universe for quick lookup? 
    # We need to identify which points are covered by which codeword.
    # Instead of mapping every point to a list of codewords (memory heavy), 
    # we can compute the mask for each codeword as a Python integer (bitset).
    
    # Helper to convert a tuple to an index in base-q
    def to_index(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute masks
    masks = [0] * N
    for i, c in enumerate(codewords):
        mask = 0
        # The codeword itself
        mask |= (1 << to_index(c))
        
        # Neighbors at distance 1
        for pos in range(n):
            orig_val = c[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    # Create neighbor
                    neighbor = c.copy()
                    neighbor[pos] = new_val
                    mask |= (1 << to_index(neighbor))
        masks[i] = mask

    # 3. Greedy Reduction
    # Maintain coverage count for each point
    # Using a list of ints for counts is faster than bit operations for this specific "redundancy" check
    # If count[p] > 1 for all p in mask[i], then i is redundant.
    
    # Initialize counts
    counts = [0] * total_points
    
    for i in range(N):
        mask = masks[i]
        # Iterate through bits set in mask
        # Since mask is sparse (17 bits set), we can just iterate the 17 neighbors again
        # Or extract bits. Extracting 17 bits from a big int is fast.
        # But we already have the neighbors. Let's just re-calculate or store neighbors.
        # Storing neighbors is more memory but faster.
        pass # We'll just use the mask to update counts? 
        # Actually, updating counts via bit iteration on a 6561-bit int is slow in Python if done naively.
        # Better: Store the list of indices for each codeword's neighborhood.
        
    # Let's re-do the neighborhood storage for efficiency
    neighborhoods = []
    for i, c in enumerate(codewords):
        neigh_indices = [to_index(c)]
        for pos in range(n):
            orig_val = c[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    neighbor = c.copy()
                    neighbor[pos] = new_val
                    neigh_indices.append(to_index(neighbor))
        neighborhoods.append(neigh_indices)
        # Update counts
        for idx in neigh_indices:
            counts[idx] += 1

    # Active codewords
    active = [True] * N
    
    # Iterative removal
    # We try to remove codewords one by one.
    # To be efficient, we can just loop through indices.
    # If a codeword is redundant, remove it and update counts.
    # We might need multiple passes if removing one makes another redundant?
    # No, removing a codeword only *decreases* counts. It cannot make a previously non-redundant codeword redundant 
    # in the sense that it *needed* the removed one to be non-unique? 
    # Wait. If A and B both cover P. A is removed. P is still covered by B. 
    # Does this make B redundant? No, B might cover other points that A also covered, but those are still covered by B.
    # Redundancy of B depends on whether *all* points in B's neighborhood are covered by *other* active codewords.
    # Removing A reduces the coverage of points in A's neighborhood.
    # If P was covered only by A and B, and we remove A, P is now covered only by B.
    # B was redundant before? If B was redundant, it meant every point in B was covered by >=2 others.
    # If we remove A, the count for points in A & B drops by 1.
    # It is possible that B was redundant, we remove A, and now B is NOT redundant?
    # No, we only remove if it IS redundant.
    # If we remove A, we update counts. This might make *other* codewords redundant that were not before?
    # No, removing coverage makes it *harder* for others to be redundant.
    # So, a single pass is sufficient?
    # Let's check: 
    # If C is redundant, it means for all p in C, count[p] >= 2.
    # We remove C. Counts for p in C decrease by 1.
    # For any other codeword D, if D was redundant, count[p] >= 2 for all p in D.
    # After removing C, for p in D, if p is also in C, count[p] becomes count[p]-1.
    # It might drop to 1. Then D is no longer redundant.
    # So we should NOT remove D in this pass if it relies on C? 
    # But we already checked D was redundant *before* removing C.
    # If we remove C, D might become essential.
    # So we should be careful. 
    # Standard greedy set cover reduction usually does:
    # while True:
    #   found = False
    #   for i in active:
    #     if redundant(i):
    #        remove(i)
    #        found = True
    #        break # or continue?
    #   if not found: break
    # 
    # If we remove one, we should restart or continue carefully.
    # Given N is small (729), we can just do a while loop with a full scan.
    
    while True:
        time_remaining = deadline - time.time()
        if time_remaining < 5.0:
            break
            
        removed_any = False
        for i in range(N):
            if not active[i]:
                continue
            # Check if i is redundant
            redundant = True
            for p in neighborhoods[i]:
                if counts[p] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove i
                active[i] = False
                removed_any = True
                for p in neighborhoods[i]:
                    counts[p] -= 1
                # Restart scan? Or continue?
                # If we continue, we might remove a codeword that is now non-redundant due to the removal of i?
                # No, removal of i *decreases* counts.
                # If a codeword j was redundant (counts >= 2), and we remove i (which covers some of j's points),
                # counts for j's points might drop to 1.
                # So j might NO LONGER be redundant.
                # But we haven't checked j yet?
                # If we check j *after* removing i, we will see counts <= 1 and NOT remove j. This is correct.
                # If we checked j *before* removing i, we would have removed it.
                # So the order matters.
                # To be safe and simple, just restart the scan or break and start a new pass.
                # A single pass might miss opportunities or make mistakes if not ordered right.
                # However, since we only remove if *currently* redundant, and removing makes things *less* redundant,
                # any codeword we haven't visited yet that was redundant *before* the removal might not be redundant *after*.
                # So if we visit it later, we will correctly decide not to remove it.
                # So a single forward pass is safe (it might not remove *all* possible, but it won't remove *invalid* ones).
                # To maximize removal, we can loop until no removals happen.
                pass 
        
        if not removed_any:
            break

    # Construct final list
    result = [codewords[i] for i in range(N) if active[i]]
    
    return result

# EVOLVE-BLOCK-END
