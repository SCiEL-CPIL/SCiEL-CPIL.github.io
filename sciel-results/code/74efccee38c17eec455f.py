import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    # Precompute all k-subsets for potential replacement search
    # This might be memory heavy for large v, but v=19 is small.
    # C(19, 8) = 75582. Storing them as tuples is fine.
    all_k_blocks = list(itertools.combinations(range(v), k))
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances with different randomizations
    for seed in range(50): # Max 50 restarts, but check time
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # 1. Greedy Construction
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Start a new block
            # Pick a random uncovered t-subset to start, to break symmetry
            # The seed program picked min(uncovered). Randomizing this helps.
            start_set = random.choice(list(uncovered))
            block = list(start_set)
            
            # Greedily add points to the block
            while len(block) < k:
                block_set = set(block)
                gains = {}
                
                # Calculate gain for each candidate point
                # Gain(x) = number of uncovered t-sets that would be covered if x is added
                # A t-set S is covered by block + x if S \ {x} is a subset of block and x not in block
                # And S is currently uncovered.
                
                # Optimization:
                # For each candidate x, we check subsets of block of size t-1.
                # If S = (subset of block of size t-1) + x is in uncovered, it's a gain.
                
                # Precompute subsets of current block of size t-1
                # This is constant per block step
                block_subsets_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in block_subsets_t_minus_1:
                        # Form candidate t-set
                        # sub is a tuple of t-1 elements
                        # We need to check if sorted(sub + (x,)) is in uncovered
                        # To avoid sorting cost, we can use a set of tuples?
                        # Uncovered is a set of tuples.
                        # We need to form the tuple in sorted order.
                        # Since sub is from combinations, it is sorted.
                        # We can insert x into sub to maintain order?
                        # Or just sort. t=5 is small.
                        candidate = tuple(sorted(sub + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    gains[x] = gain
                
                # Pick the point with max gain
                # If tie, pick random among max to diversify
                max_gain = max(gains.values()) if gains else 0
                if max_gain == 0:
                    # No point adds coverage? This shouldn't happen if uncovered is not empty
                    # and we haven't filled the block, unless the uncovered sets are disjoint from block?
                    # Actually, if block has < k points, and there is an uncovered set,
                    # there must be some point that helps?
                    # Not necessarily. E.g. uncovered set is {0,1,2,3,4}, block is {5,6}.
                    # Adding 7 doesn't help. But we can add 0,1,2,3,4.
                    # So max_gain should be > 0 if we pick the right points.
                    # If max_gain is 0, it means no single point addition covers any uncovered set.
                    # This implies that no uncovered set is a superset of (block + x).
                    # We should just pick a random point to keep building.
                    candidates = [x for x in range(v) if x not in block_set]
                    block.append(random.choice(candidates))
                else:
                    candidates = [x for x, g in gains.items() if g == max_gain]
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        if not uncovered:
            # 2. Local Search
            current_blocks = list(blocks)
            
            # Precompute set of blocks for fast lookup
            block_set = set(current_blocks)
            
            # Helper to compute unique subsets for a block
            def get_unique_subsets(b):
                unique = []
                # Count coverage for all t-subsets in b
                # We need global counts.
                # To be efficient, we should maintain global counts.
                # But for simplicity in the loop, let's recompute counts for the whole set?
                # C(19,5) = 11628. 300 blocks * 56 subsets = 16800 updates.
                # Recomputing counts for 300 blocks: 300 * 56 = 16800 checks.
                # This is fast enough.
                pass
            
            # Better: Maintain a count array/dict.
            # Map t-subset to index? Or just use the tuple as key in a dict.
            
            def compute_counts(blocks_list):
                counts = {}
                for b in blocks_list:
                    for sub in itertools.combinations(b, t):
                        counts[sub] = counts.get(sub, 0) + 1
                return counts
            
            # Local Search Loop
            improved = True
            while improved and time.time() < deadline:
                improved = False
                counts = compute_counts(current_blocks)
                
                # Try to remove
                for i, b in enumerate(list(current_blocks)):
                    unique_subs = []
                    for sub in itertools.combinations(b, t):
                        if counts[sub] == 1:
                            unique_subs.append(sub)
                    
                    if not unique_subs:
                        # Can remove
                        current_blocks.pop(i)
                        improved = True
                        break # Restart loop with new state
                
                if improved:
                    continue
                
                # Try to replace
                # Sort blocks by number of unique subsets to target "bad" blocks
                block_uniques = []
                for b in current_blocks:
                    unique_subs = [sub for sub in itertools.combinations(b, t) if counts[sub] == 1]
                    block_uniques.append((len(unique_subs), b, unique_subs))
                
                block_uniques.sort(key=lambda x: x[0])
                
                for count_uniq, b, unique_subs in block_uniques:
                    if count_uniq == 0:
                        # Should have been removed
                        continue
                    
                    # Try to replace b with b' that covers all unique_subs
                    # Condition: unique_subs must be coverable by a single k-block
                    # This is true if the union of unique_subs is small enough?
                    # No, each unique sub is size t.
                    # We need a block B' of size k such that for all S in unique_subs, S subset B'.
                    
                    # If count_uniq is small, we can search.
                    # Heuristic: If count_uniq == 1, let S be the unique sub.
                    # We need B' such that S subset B'.
                    # There are C(v-t, k-t) such blocks.
                    # We can pick one randomly, or one that maximizes coverage of other single-covered sets?
                    # Let's just pick a random one different from b.
                    
                    if count_uniq == 1:
                        S = unique_subs[0]
                        S_set = set(S)
                        
                        # Find candidates
                        # To avoid generating all C(19,8) every time, we can pre-generate or sample.
                        # Let's sample a few.
                        replaced = False
                        for _ in range(10): # Try 10 random replacements
                            # Generate random block containing S
                            remaining = [x for x in range(v) if x not in S_set]
                            if len(remaining) < k - t:
                                continue
                            extra = random.sample(remaining, k - t)
                            new_block = tuple(sorted(list(S) + extra))
                            
                            if new_block == b:
                                continue
                            
                            # Check if new_block is already in current_blocks
                            if new_block in block_set:
                                continue
                            
                            # Check if new_block covers all unique_subs (which is just S)
                            # It does by construction.
                            
                            # Perform replacement
                            idx = current_blocks.index(b)
                            current_blocks[idx] = new_block
                            block_set.discard(b)
                            block_set.add(new_block)
                            improved = True
                            replaced = True
                            break
                        
                        if replaced:
                            break # Restart loop
        
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks
                
    # If no valid covering was found (should not happen), return greedy result or empty
    if best_blocks is None:
        # Fallback: return the last greedy result if it was valid, or just a dummy
        # But the loop only updates best_blocks if uncovered is empty.
        # If the first greedy run didn't finish or failed, we might have nothing.
        # However, greedy always finishes for these parameters.
        return []
        
    return [list(b) for b in best_blocks]
