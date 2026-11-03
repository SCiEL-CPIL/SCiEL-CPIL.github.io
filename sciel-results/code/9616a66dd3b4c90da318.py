import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors (radius 1 ball)
    neighbors = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        neighbors.append(idxs)
        
    # Precompute cover bitsets
    cover_bitsets = []
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
        cover_bitsets.append(bs)
        
    full_mask = (1 << total_words) - 1
    
    # Precompute "distant" masks: words at distance >= 3
    # Used for packing lower bound
    # dist(u, v) >= 3 is equivalent to NOT (dist(u, v) <= 2)
    # We can compute "close" masks (dist <= 2) and invert.
    # Or just compute directly.
    # For n=7, q=3, total_words = 2187.
    # dist(u,v) <= 2 means they differ in at most 2 positions.
    
    # To save time, we can compute "conflict" masks for the packing bound.
    # conflict[i] = bitset of words at distance < 3 from i.
    conflict_masks = []
    for i in range(total_words):
        w_i = idx_to_tuple(i)
        mask = 0
        for j in range(total_words):
            w_j = idx_to_tuple(j)
            dist = 0
            for k in range(n):
                if w_i[k] != w_j[k]:
                    dist += 1
            if dist < 3:
                mask |= (1 << j)
        conflict_masks.append(mask)
        
    # Greedy Packing Lower Bound
    def packing_lb(uncovered_mask):
        count = 0
        u = uncovered_mask
        while u:
            # Pick a word. To make the bound tighter, we might want to pick a word 
            # that has many conflicts, or few? 
            # Standard greedy for independent set: pick vertex, remove neighbors.
            # Here we want a large independent set in the graph where edges connect dist < 3.
            # So we pick a vertex, and remove all its neighbors (and itself).
            # The size of the resulting set is the independent set size.
            # To maximize the independent set, we should probably pick vertices with low degree?
            # But for a LOWER BOUND on the cover size, we just need ANY independent set.
            # A larger independent set gives a better (higher) lower bound.
            # So we want a greedy heuristic that finds a large independent set.
            # Strategy: Pick the vertex with the fewest remaining neighbors in the induced subgraph?
            # That's expensive.
            # Simpler: Pick the vertex with the lowest index?
            # Or random?
            # Let's stick to lowest index for speed and determinism, or maybe 
            # the one with the fewest bits in (u & conflict_masks[idx])?
            
            # For speed, let's just pick the lowest set bit.
            lsb = u & -u
            idx = lsb.bit_length() - 1
            count += 1
            # Remove idx and all its conflicts
            u &= ~conflict_masks[idx]
        return count

    # Initial Greedy Construction
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = full_mask
        code = []
        
        while uncovered_mask != 0:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
        
        return code

    best_code = greedy_construct(0)
    best_size = len(best_code)
    
    # Branch and Bound
    def branch_and_bound(uncovered_mask, current_code, current_size):
        nonlocal best_code, best_size
        
        if time.time() > deadline - 1.0:
            return
            
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_code = list(current_code)
            return
            
        # Lower Bound
        lb = packing_lb(uncovered_mask)
        if current_size + lb >= best_size:
            return
            
        # Select pivot word
        # Choose a word in uncovered_mask.
        # Heuristic: Choose a word that has the fewest candidate coverers in the "remaining" sense?
        # Or just the first one.
        # To improve, choose a word w such that the set of candidates N[w] is small?
        # All N[w] have size 1 + n*(q-1) = 1 + 7*2 = 15. So size is constant.
        # However, some candidates might be "better" (cover more).
        
        # Let's pick the word with the highest "pressure" or just the first set bit.
        # Picking the first set bit is fast.
        lsb = uncovered_mask & -uncovered_mask
        pivot_idx = lsb.bit_length() - 1
        
        # Candidates are neighbors of pivot_idx
        candidates = neighbors[pivot_idx]
        
        # Sort candidates by gain (how many new words they cover)
        # This is a heuristic to find a good solution early, which tightens the bound.
        def gain_score(c):
            return (uncovered_mask & cover_bitsets[c]).bit_count()
            
        candidates.sort(key=gain_score, reverse=True)
        
        for c in candidates:
            # If we pick c, we cover cover_bitsets[c]
            new_uncovered = uncovered_mask & ~cover_bitsets[c]
            
            # Optional: If new_uncovered is same as uncovered_mask (shouldn't happen if c covers pivot), skip
            # But c is in neighbors[pivot], so it covers pivot.
            
            current_code.append(c)
            branch_and_bound(new_uncovered, current_code, current_size + 1)
            current_code.pop()
            
            # If we found a solution with size == current_size + 1 + lb_min_possible?
            # If best_size == current_size + 1, we can't do better than this branch (since lb >= 1).
            # Actually, if best_size <= current_size + 1, we can prune?
            # No, we might find a solution of size current_size + 1 in this branch, 
            # but we are looking for < best_size.
            # If best_size == current_size + 1, any solution in this branch has size >= current_size + 1.
            # So it cannot improve best_size. We can break.
            if best_size <= current_size + 1:
                break

    # Run B&B
    # We need a good initial upper bound.
    # Let's run a few greedy passes to tighten it.
    for i in range(5):
        code = greedy_construct(i)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
    # Start B&B from empty set
    branch_and_bound(full_mask, [], 0)
    
    # If B&B didn't finish or didn't improve, we might want to do a local search on the best found
    # But B&B is exact if it finishes. If it times out, it returns the best incumbent.
    
    # Convert result
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
