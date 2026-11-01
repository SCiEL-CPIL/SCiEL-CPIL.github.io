import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # Precompute all t-subsets and k-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Map t-subset to index
    t_sub_to_idx = {tsub: i for i, tsub in enumerate(all_t_subs)}
    
    # Precompute coverage structures
    # block_to_t[i]: list of t-subset indices covered by k-subset i
    block_to_t = [[] for _ in range(n_k_subs)]
    # t_to_blocks[j]: list of k-subset indices covering t-subset j
    t_to_blocks = [[] for _ in range(n_t_subs)]
    
    for i, ksub in enumerate(all_k_subs):
        # Generate all t-subsets of ksub
        for tsub in itertools.combinations(ksub, t):
            idx = t_sub_to_idx[tsub]
            block_to_t[i].append(idx)
            t_to_blocks[idx].append(i)
            
    # Greedy Initialization
    selected = set()
    counts = np.zeros(n_t_subs, dtype=np.int8)
    uncovered_count = n_t_subs
    
    # Fast Greedy
    # We maintain a list of t-indices with count 0
    # To avoid O(N) scan for min count, we just pick a random uncovered t
    # and find the best block covering it.
    
    rng = random.Random(42)
    
    # Initial uncovered list
    uncovered_t_indices = list(range(n_t_subs))
    
    while uncovered_t_indices:
        # Pick a random uncovered t
        t_idx = rng.choice(uncovered_t_indices)
        
        # Get candidates covering this t
        candidates = t_to_blocks[t_idx]
        
        best_block = -1
        best_gain = 0
        
        for c in candidates:
            if c in selected:
                continue
            # Calculate gain: number of currently uncovered t's this block covers
            gain = 0
            for t_i in block_to_t[c]:
                if counts[t_i] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = c
        
        if best_block == -1:
            # Should not happen if we have a full cover
            break
            
        # Add best_block
        selected.add(best_block)
        
        # Update counts and uncovered list
        # We need to remove t's from uncovered_t_indices if they become covered
        # This is O(N) if we do it naively. 
        # Optimization: Just mark them covered, and filter the list occasionally or use a set.
        # Given small size, let's just update counts and filter the list every 10 steps or when it gets small.
        # Actually, just updating counts is fine. We pick from uncovered_t_indices.
        # If we pick a t that is no longer uncovered (count > 0), we skip?
        # No, we pick from the list. If the list contains covered t's, we waste time.
        # Let's filter the list after each addition.
        
        for t_i in block_to_t[best_block]:
            counts[t_i] += 1
        
        # Filter uncovered_t_indices
        # This is O(N_t). N_t = 3876. 150 steps * 3876 = 500k ops. Fast.
        uncovered_t_indices = [t_i for t_i in uncovered_t_indices if counts[t_i] == 0]
        
    current_blocks = list(selected)
    
    # Local Search
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Phase 1: Remove redundant blocks
        # Shuffle selected to avoid bias
        selected_list = list(selected)
        rng.shuffle(selected_list)
        
        for b in selected_list:
            if time.time() - start_time > time_limit:
                break
            if b not in selected:
                continue
                
            # Check if b is redundant
            redundant = True
            for t_i in block_to_t[b]:
                if counts[t_i] == 1:
                    redundant = False
                    break
                    
            if redundant:
                # Remove b
                selected.remove(b)
                for t_i in block_to_t[b]:
                    counts[t_i] -= 1
                improved = True
                break # Restart loop to re-evaluate with new counts
                
        if improved:
            continue
            
        # Phase 2: 2-for-1 Swap
        # Try to replace two blocks with one to reduce count by 1
        selected_list = list(selected)
        rng.shuffle(selected_list)
        
        n_sel = len(selected_list)
        # To avoid O(N^2) every iteration, we can limit the number of pairs we try per pass
        # Or just try all pairs if N is small (~150). 150*150 = 22500 pairs.
        # For each pair, finding c is fast.
        
        found_swap = False
        # Randomize pair order
        pairs = []
        for i in range(n_sel):
            for j in range(i+1, n_sel):
                pairs.append((selected_list[i], selected_list[j]))
        rng.shuffle(pairs)
        
        for b1, b2 in pairs:
            if time.time() - start_time > time_limit:
                break
            if b1 not in selected or b2 not in selected:
                continue
                
            # Determine must_cover t's
            # t in b1 AND b2: needs counts[t] >= 2 (if c covers) or >= 3 (if not).
            # t in (b1 U b2) \ (b1 I b2): needs counts[t] >= 1 (if c covers) or >= 2 (if not).
            
            # Let's identify critical t's that MUST be covered by c
            must_cover = []
            
            # Get t's in b1 and b2
            t_set_b1 = set(block_to_t[b1])
            t_set_b2 = set(block_to_t[b2])
            t_intersection = t_set_b1 & t_set_b2
            t_union_only = (t_set_b1 | t_set_b2) - t_intersection
            
            for t_i in t_intersection:
                if counts[t_i] == 2:
                    # If c doesn't cover, count becomes 0. So c MUST cover.
                    must_cover.append(t_i)
                # If counts[t_i] >= 3, c doesn't need to cover.
                
            for t_i in t_union_only:
                if counts[t_i] == 1:
                    # If c doesn't cover, count becomes 0. So c MUST cover.
                    must_cover.append(t_i)
                # If counts[t_i] >= 2, c doesn't need to cover.
                
            if must_cover:
                # Pick a t from must_cover to find candidates
                t_target = must_cover[0]
                candidates = t_to_blocks[t_target]
                
                valid_c = -1
                for c in candidates:
                    if c in selected:
                        continue
                    # Check if c covers all t in must_cover
                    c_covers_all = True
                    # Optimization: Convert must_cover to set for O(1) lookup? 
                    # Or just iterate must_cover and check if t in block_to_t[c]?
                    # block_to_t[c] is a list. Checking membership is O(len).
                    # Let's make block_to_t sets for faster lookup? 
                    # We can precompute block_to_t_sets.
                    
                    # For now, let's just check.
                    # Since must_cover is small, and block_to_t[c] is 35 items, 
                    # we can create a set of block_to_t[c] on the fly? 
                    # Or precompute.
                    
                    # Let's precompute block_to_t_sets in the main precomp phase? 
                    # No, let's just do a simple check.
                    # We can check if all t in must_cover are in block_to_t[c].
                    # We can convert block_to_t[c] to a set once per candidate?
                    # Or better: Precompute a bitmask or set for each block?
                    # Let's assume we can check quickly.
                    
                    # Actually, we can just iterate must_cover and check if t is in the list block_to_t[c].
                    # To make this O(1), we can use a global array of sets?
                    # Let's stick to list for memory, but convert to set in the inner loop?
                    # No, that's slow.
                    
                    # Alternative: 
                    # We know c covers t_target.
                    # We need to check other t's in must_cover.
                    # Let's just do:
                    c_t_set = set(block_to_t[c])
                    if all(t in c_t_set for t in must_cover):
                        valid_c = c
                        break
                
                if valid_c != -1:
                    # Perform swap: remove b1, b2; add valid_c
                    selected.remove(b1)
                    selected.remove(b2)
                    selected.add(valid_c)
                    
                    # Update counts
                    for t_i in block_to_t[b1]:
                        counts[t_i] -= 1
                    for t_i in block_to_t[b2]:
                        counts[t_i] -= 1
                    for t_i in block_to_t[valid_c]:
                        counts[t_i] += 1
                        
                    found_swap = True
                    break # Restart local search loop
                    
            else:
                # must_cover is empty.
                # We can pick any c not in selected.
                # To be safe, pick one that maximizes coverage of "tight" t's?
                # Or just a random one.
                # Let's pick a random block not in selected.
                # We need to ensure it doesn't break validity?
                # If must_cover is empty, it means for all t in b1 U b2, counts[t] is high enough
                # such that removing b1, b2 and adding ANY c keeps counts >= 1.
                # Wait, does adding c affect t's NOT in b1 U b2?
                # No, c adds coverage to t's in c.
                # If c covers a t NOT in b1 U b2, counts[t] increases. Safe.
                # If c covers a t IN b1 U b2, counts[t] changes by -1 (or -2) + 1.
                # We already verified counts are high enough.
                
                # So we just need to find ANY c not in selected.
                # We can pick a random one.
                # To avoid picking a block that makes the solution "worse" in terms of future swaps,
                # we can pick one that covers many t's with count == 1?
                # But for now, just pick a random valid c.
                
                # Find a random c not in selected
                # We can iterate all_k_subs until we find one not in selected.
                # Since selected is small, this is fast.
                while True:
                    c = rng.randint(0, n_k_subs - 1)
                    if c not in selected:
                        break
                        
                # Perform swap
                selected.remove(b1)
                selected.remove(b2)
                selected.add(c)
                
                # Update counts
                for t_i in block_to_t[b1]:
                    counts[t_i] -= 1
                for t_i in block_to_t[b2]:
                    counts[t_i] -= 1
                for t_i in block_to_t[c]:
                    counts[t_i] += 1
                    
                found_swap = True
                break
                
        if found_swap:
            improved = True
            continue
            
        # If no swap found, we might be stuck.
        # We can try 1-for-1 swaps to escape local minima?
        # Or just stop.
        break

    return [list(all_k_subs[i]) for i in selected]
