import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # 3. Precompute coverage masks for each block
    block_masks = [0] * num_k
    for i, block in enumerate(k_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = t_index_map[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    full_mask = (1 << num_t) - 1
    
    def _greedy_pass(seed):
        """One greedy set cover pass."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_idx = -1
            best_count = -1
            
            # Randomize tie-breaking for diversity
            indices = list(range(num_k))
            if seed % 2 == 0:
                rng.shuffle(indices)
            
            for i in indices:
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        return selected

    def _remove_redundant(selected_indices):
        """Remove redundant blocks from a valid covering."""
        if not selected_indices:
            return selected_indices
        
        kept = list(selected_indices)
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            new_kept = []
            # Pre-compute union of all kept blocks? 
            # Or just check each one against the rest.
            for i in range(len(kept)):
                idx = kept[i]
                others_mask = 0
                for j in range(len(kept)):
                    if j != i:
                        others_mask |= block_masks[kept[j]]
                
                if (block_masks[idx] & ~others_mask) == 0:
                    changed = True
                else:
                    new_kept.append(idx)
            
            if changed:
                kept = new_kept
        
        return kept

    def _local_search(selected_indices, max_steps=500):
        """Local search to minimize number of blocks.
        Includes 1-for-0, 2-for-1, and 1-for-1 swaps."""
        if time.time() > deadline:
            return selected_indices
        
        rng = random.Random()
        current = set(selected_indices)
        best = set(current)
        best_len = len(current)
        
        # Helper to compute unique coverage for each block in current set
        def get_unique_covers(S):
            uniques = {}
            if not S:
                return uniques
            # Total coverage of all blocks in S
            # We compute unique for each A: M_A \ (Union of M_X for X in S\{A})
            # To speed up, we can just iterate.
            for A in S:
                others_cov = 0
                for B in S:
                    if B != A:
                        others_cov |= block_masks[B]
                uniques[A] = block_masks[A] & ~others_cov
            return uniques

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            # 1. Remove redundant blocks (1-for-0)
            changed = True
            while changed and time.time() < deadline - 2.0:
                changed = False
                uniques = get_unique_covers(current)
                for A in list(current):
                    if uniques[A] == 0:
                        current.discard(A)
                        changed = True
                        if len(current) < best_len:
                            best = set(current)
                            best_len = len(current)
                
            if len(current) < best_len:
                best = set(current)
                best_len = len(current)
                
            # 2. Try 2-for-1 swaps
            # Find pairs (A, B) in current that can be replaced by one block C not in current
            # Condition: Unique_A | Unique_B is covered by C
            # Optimization: For each C not in current, find A, B in current such that Unique_A | Unique_B subset of M_C
            
            if time.time() < deadline - 5.0: # 2-for-1 is expensive, use less time
                uniques = get_unique_covers(current)
                # List of blocks in current
                curr_list = list(current)
                # We will iterate over candidate blocks C
                # To make it faster, we only check C if it has high potential?
                # Or just sample? No, we want to find the best improvement.
                
                # To avoid O(N*M) being too slow, we limit the search
                # We can sample some C's or just do it if N is small enough.
                # N ~ 340, M ~ 75k. 340*75k = 25M. 
                # Let's try it.
                
                found_swap = False
                # Iterate over candidate blocks C
                # We can shuffle C's to randomize which swap we find
                candidate_indices = [i for i in range(num_k) if i not in current]
                rng.shuffle(candidate_indices)
                
                for C in candidate_indices:
                    if time.time() > deadline - 3.0:
                        break
                    MC = block_masks[C]
                    # Find A, B in current such that (Unique_A | Unique_B) & ~MC == 0
                    # This is equivalent to Unique_A & ~MC == 0 AND Unique_B & ~MC == 0
                    # So we need A, B such that Unique_A subset MC and Unique_B subset MC
                    compatible = []
                    for A in curr_list:
                        if uniques[A] & ~MC == 0:
                            compatible.append(A)
                    
                    if len(compatible) >= 2:
                        # Pick two, e.g., the first two
                        A, B = compatible[0], compatible[1]
                        # Perform swap
                        current.discard(A)
                        current.discard(B)
                        current.add(C)
                        found_swap = True
                        if len(current) < best_len:
                            best = set(current)
                            best_len = len(current)
                        break # Restart loop with new state
            
            if found_swap:
                continue
            
            # 3. Try 1-for-1 swaps
            swap_done = False
            uniques = get_unique_covers(current)
            for A in list(current):
                if time.time() > deadline - 2.0:
                    break
                UA = uniques[A]
                if UA == 0:
                    # Should have been removed in step 1
                    current.discard(A)
                    if len(current) < best_len:
                        best = set(current)
                        best_len = len(current)
                    swap_done = True
                    break
                
                # Find C not in current such that UA subset MC
                # And C adds minimal extra? Or just any?
                # We want to replace A with C.
                # Condition: UA subset MC.
                # We also want to ensure C is "good". 
                # Let's find C that covers UA and has high overlap with current coverage?
                
                best_C = -1
                # To speed up, we can sample or just scan
                # Scanning 75k blocks is ~0.1s? 
                # 340 blocks * 0.1s = 34s. Too slow if we do it for all.
                # We only do it for one A at a time?
                
                # Let's just find ANY C that covers UA.
                # If we find one, we swap.
                # To be smarter, we can prefer C that covers more of the "rest" of UA?
                # No, UA is what A uniquely covers. C must cover UA.
                
                # Optimization: Pre-filter C?
                # Just scan.
                for C in range(num_k):
                    if C in current:
                        continue
                    if UA & ~block_masks[C] == 0:
                        # C covers UA
                        # Check if swapping improves anything?
                        # Size stays same.
                        # We might get stuck in a loop.
                        # To avoid loops, we can track visited states?
                        # Or just randomize.
                        current.discard(A)
                        current.add(C)
                        swap_done = True
                        break
                
                if swap_done:
                    break
            
            if not swap_done:
                # Stuck: try adding a random block to escape local minimum
                if len(current) < num_k:
                    candidates = [i for i in range(num_k) if i not in current]
                    if candidates:
                        add_idx = rng.choice(candidates)
                        current.add(add_idx)
        
        return best
    
    def _polish(selected_indices):
        """Final polishing."""
        if time.time() > deadline:
            return selected_indices
        
        result = _remove_redundant(selected_indices)
        
        # Try a few rounds of local search
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            result = _local_search(result, max_steps=200)
            result = _remove_redundant(result)
        
        return result
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Run multiple randomized greedy passes
    attempt = 0
    while time.time() < deadline - 30.0:  # Leave 30s for polishing
        selected = _greedy_pass(attempt)
        if selected is None:
            break
        
        selected = _remove_redundant(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        
        if best_len <= 338: # Target
            break
        
        attempt += 1
    
    # Phase 2: Polish the best solution found
    if best_blocks is not None and time.time() < deadline - 1.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(k_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_blocks]
        
        polished = _polish(best_indices)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = [list(k_blocks[i]) for i in polished]
    
    # Fallback
    if best_blocks is None:
        best_blocks = []
        
    return best_blocks
