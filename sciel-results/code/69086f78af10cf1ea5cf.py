import itertools
import numpy as np
import time
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # block_triple_indices[i] is a list of triple indices in block i
    # triple_to_blocks[j] is a list of block indices containing triple j
    block_triple_indices = [None] * n_blocks
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        idxs = [triple_to_idx[sub] for sub in subs]
        block_triple_indices[i] = idxs
        for idx in idxs:
            triple_to_blocks[idx].append(i)

    # Bitset representation for fast coverage checks
    # Each triple is represented by a Python int bitmask
    # block_bitset[i] = bitmask of triples covered by block i
    # selected_bitset = OR of all selected block bitsets
    block_bitset = [0] * n_blocks
    for i in range(n_blocks):
        bm = 0
        for idx in block_triple_indices[i]:
            bm |= (1 << idx)
        block_bitset[i] = bm

    # For local search: for each block, which other blocks share each triple
    # We'll use a simpler approach: for a given triple, we have triple_to_blocks

    def greedy_cover(random_seed, time_limit=None):
        """Greedy using ratio (newly covered / block size) with tie-break by absolute gain."""
        rng = np.random.RandomState(random_seed)
        gain = np.full(n_blocks, k_t, dtype=np.int32)
        covered_bitset = 0
        selected_blocks = []
        selected_set = set()
        remaining = n_triples
        
        while remaining > 0:
            if time_limit is not None and time.time() > time_limit:
                break
            # Find max gain (all blocks have same size k, so ratio = gain/k)
            max_gain = int(np.max(gain))
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Randomly pick one
            best_idx = int(candidates[rng.randint(len(candidates))])
            
            selected_blocks.append(list(all_blocks[best_idx]))
            selected_set.add(best_idx)
            
            # Update: find newly covered triples
            new_triples = block_bitset[best_idx] & ~covered_bitset
            if new_triples == 0:
                continue
            covered_bitset |= new_triples
            
            # Count and update remaining
            newly_covered_count = bin(new_triples).count('1')
            remaining -= newly_covered_count
            
            # Update gains: for each newly covered triple, decrement gain of all blocks containing it
            nt = new_triples
            while nt:
                low_bit = nt & (-nt)
                triple_idx = low_bit.bit_length() - 1
                nt ^= low_bit
                for b_idx in triple_to_blocks[triple_idx]:
                    gain[b_idx] -= 1
                        
        return selected_blocks, remaining

    def local_search(blocks, time_limit=None):
        """Remove redundant blocks and try swaps to reduce count."""
        blocks = [list(b) for b in blocks]
        
        # Build coverage count for each triple
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                triple_count[triple_to_idx[sub]] += 1
        
        # Phase A: Remove redundant blocks (those where every triple is covered >= 2 times)
        changed = True
        while changed:
            if time_limit is not None and time.time() > time_limit:
                break
            changed = False
            
            for i in range(len(blocks)):
                if time_limit is not None and time.time() > time_limit:
                    break
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    if triple_count[triple_to_idx[sub]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(blocks[i], t):
                        triple_count[triple_to_idx[sub]] -= 1
                    blocks.pop(i)
                    changed = True
                    break  # restart
        
        # Phase B: Try swaps - remove block i, add block j that covers all uniquely-covered triples
        # Precompute: for each block, the set of triples it uniquely covers (count==1)
        # We'll do this iteratively
        swap_improved = True
        while swap_improved:
            if time_limit is not None and time.time() > time_limit:
                break
            swap_improved = False
            
            for i in range(len(blocks)):
                if time_limit is not None and time.time() > time_limit:
                    break
                
                # Find triples uniquely covered by block i
                unique_triples = []
                for sub in itertools.combinations(blocks[i], t):
                    t_idx = triple_to_idx[sub]
                    if triple_count[t_idx] == 1:
                        unique_triples.append(t_idx)
                
                if not unique_triples:
                    # Block is redundant (should have been caught in Phase A, but just in case)
                    for sub in itertools.combinations(blocks[i], t):
                        triple_count[triple_to_idx[sub]] -= 1
                    blocks.pop(i)
                    swap_improved = True
                    break
                
                # Find a block j (not i) that covers all unique_triples
                # And is not already in the solution
                # Check against all candidate blocks (not just current ones) for better swaps
                # But first try current blocks minus i
                found_swap = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    # Check if block j covers all unique_triples
                    covers_all = True
                    for t_idx in unique_triples:
                        if triple_count[t_idx] == 1:
                            # This triple is only in block i, so block j must contain it
                            # Check if block j contains this triple
                            # We need to check membership
                            block_j_set = set(blocks[j])
                            triple_tuple = all_triples[t_idx]
                            if not all(pt in block_j_set for pt in triple_tuple):
                                covers_all = False
                                break
                    if covers_all:
                        # Perform swap: remove i, add j (but j is already there, so just remove i)
                        # Wait, if j is already in the solution, we just remove i
                        for sub in itertools.combinations(blocks[i], t):
                            triple_count[triple_to_idx[sub]] -= 1
                        blocks.pop(i)
                        swap_improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
        
        return blocks

    def local_search_with_external(blocks, time_limit=None):
        """Enhanced local search that also tries adding external blocks to enable removals."""
        blocks = [list(b) for b in blocks]
        
        # Build coverage count
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                triple_count[triple_to_idx[sub]] += 1
        
        # First do standard reduction
        changed = True
        while changed:
            if time_limit is not None and time.time() > time_limit:
                break
            changed = False
            for i in range(len(blocks)):
                if time_limit is not None and time.time() > time_limit:
                    break
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    if triple_count[triple_to_idx[sub]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(blocks[i], t):
                        triple_count[triple_to_idx[sub]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        # Try 1-for-1 swaps with external blocks
        # For each block i, find its unique triples, then search all_blocks for a replacement
        current_block_set = set(tuple(b) for b in blocks)
        
        improved = True
        while improved:
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            
            for i in range(len(blocks)):
                if time_limit is not None and time.time() > time_limit:
                    break
                
                unique_triples = []
                for sub in itertools.combinations(blocks[i], t):
                    t_idx = triple_to_idx[sub]
                    if triple_count[t_idx] == 1:
                        unique_triples.append(t_idx)
                
                if not unique_triples:
                    for sub in itertools.combinations(blocks[i], t):
                        triple_count[triple_to_idx[sub]] -= 1
                    removed = blocks.pop(i)
                    current_block_set.discard(tuple(removed))
                    improved = True
                    break
                
                # Search for a replacement block that covers all unique_triples
                # and is not in current_block_set
                # Use bitsets for speed: a block covers all unique_triples if 
                # (block_bitset[j] & unique_mask) == unique_mask
                unique_mask = 0
                for t_idx in unique_triples:
                    unique_mask |= (1 << t_idx)
                
                # First check current blocks (excluding i)
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if (block_bitset[j] & unique_mask) == unique_mask:
                        # Block j already covers all unique triples, just remove i
                        for sub in itertools.combinations(blocks[i], t):
                            triple_count[triple_to_idx[sub]] -= 1
                        removed = blocks.pop(i)
                        current_block_set.discard(tuple(removed))
                        improved = True
                        break
                
                if improved:
                    break
                
                # Check external blocks
                # Sample a subset of external blocks for speed
                external_candidates = []
                for j in range(n_blocks):
                    if tuple(all_blocks[j]) not in current_block_set:
                        external_candidates.append(j)
                
                # Shuffle and try up to a limit
                rng_ls = np.random.RandomState(hash(tuple(sorted(blocks[i]))) % 2**31)
                max_checks = min(len(external_candidates), 500)
                if max_checks > 0:
                    rng_ls.shuffle(external_candidates[:max_checks]) if max_checks < len(external_candidates) else None
                    # Just try first max_checks after shuffling
                    if max_checks < len(external_candidates):
                        perm = rng_ls.permutation(len(external_candidates))[:max_checks]
                        candidates_to_check = [external_candidates[p] for p in perm]
                    else:
                        candidates_to_check = external_candidates
                    
                    for j in candidates_to_check:
                        if (block_bitset[j] & unique_mask) == unique_mask:
                            # Replace block i with block j
                            # Remove i
                            for sub in itertools.combinations(blocks[i], t):
                                triple_count[triple_to_idx[sub]] -= 1
                            removed = blocks.pop(i)
                            current_block_set.discard(tuple(removed))
                            # Add j
                            blocks.append(list(all_blocks[j]))
                            current_block_set.add(tuple(all_blocks[j]))
                            for sub in itertools.combinations(all_blocks[j], t):
                                triple_count[triple_to_idx[sub]] += 1
                            improved = True
                            break
                    if improved:
                        break
        
        return blocks

    # Run multiple times with time budget
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy restarts (fast due to bitsets)
    n_restarts = 0
    greedy_deadline = deadline - 45  # leave 45s for local search
    while True:
        if time.time() > greedy_deadline:
            break
        blocks, remaining = greedy_cover(n_restarts, time_limit=greedy_deadline)
        if remaining == 0:  # valid cover
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
        n_restarts += 1
    
    # Phase 2: Local search on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        improved = local_search_with_external(best_blocks, time_limit=deadline - 5)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
    
    return best_blocks
