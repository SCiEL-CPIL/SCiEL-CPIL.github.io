import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start = time.time()
    DEADLINE = 145.0

    def time_ok():
        return time.time() < DEADLINE - 5.0

    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    comb_to_idx = {c: i for i, c in enumerate(all_t)}

    # Precompute pair-to-triplets
    pair_to_triplets = {}
    for i, triplet in enumerate(all_t):
        for pair in itertools.combinations(triplet, t - 1):
            if pair not in pair_to_triplets:
                pair_to_triplets[pair] = []
            pair_to_triplets[pair].append(i)

    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        while uncovered.any():
            if not time_ok():
                return None
            
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            start_idx = unc_idx[rng.randrange(len(unc_idx))]
            block = list(all_t[start_idx])
            block_set = set(block)

            while len(block) < k:
                if not time_ok():
                    return None
                
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                
                best_gain = -1
                best_candidate = None
                
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        idx = comb_to_idx.get(triplet)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = p
                    if best_gain == len(pairs_in_block):
                        break

                if best_candidate is None:
                    best_candidate = candidates[0]
                
                block.append(best_candidate)
                block_set.add(best_candidate)

            block.sort()
            blocks.append(block)

            for tsub in itertools.combinations(block, t):
                idx = comb_to_idx[tsub]
                uncovered[idx] = False

        return blocks

    def optimize(blocks):
        if not blocks:
            return blocks
        
        count = np.zeros(n_t, dtype=np.int32)
        block_triplets = []
        for b in blocks:
            t_ids = []
            for tsub in itertools.combinations(b, t):
                idx = comb_to_idx[tsub]
                count[idx] += 1
                t_ids.append(idx)
            block_triplets.append(t_ids)
        
        improved = True
        iteration = 0
        while improved and time_ok():
            improved = False
            iteration += 1
            new_blocks = []
            new_block_triplets = []
            # Map block content to index in new_blocks to check for duplicates
            # We need to handle the case where we replace B_i with B_new which is same as B_j
            
            # First, determine which blocks to keep/replace
            # We iterate and decide
            keep_flags = [True] * len(blocks)
            replacement_blocks = {} # i -> new_block
            
            for i in range(len(blocks)):
                if not time_ok():
                    return blocks
                
                unique_ids = set()
                for tid in block_triplets[i]:
                    if count[tid] == 1:
                        unique_ids.add(tid)
                
                if not unique_ids:
                    keep_flags[i] = False
                    improved = True
                    continue
                
                req_points = set()
                for tid in unique_ids:
                    req_points.update(all_t[tid])
                
                if len(req_points) > k:
                    # Cannot replace, keep
                    continue
                
                # Check if any other block covers unique_ids
                covered_by_other = False
                for j in range(len(blocks)):
                    if i == j:
                        continue
                    # Check if block j covers all unique_ids
                    # We can use set intersection
                    # If set(unique_ids) is subset of set(block_triplets[j])
                    if unique_ids.issubset(set(block_triplets[j])):
                        covered_by_other = True
                        break
                
                if covered_by_other:
                    keep_flags[i] = False
                    improved = True
                    continue
                
                # Try to construct new block
                # Fill with req_points + random
                new_block = list(req_points)
                available = [x for x in range(v) if x not in req_points]
                # Shuffle available to get variety
                random.shuffle(available)
                while len(new_block) < k and available:
                    new_block.append(available.pop(0))
                new_block.sort()
                
                # Check if new_block is same as old
                if new_block == blocks[i]:
                    continue
                
                # Check if new_block is same as any other block we are keeping
                # We will handle duplicates in the final assembly
                replacement_blocks[i] = new_block
                keep_flags[i] = False # We are replacing, so old is removed
                improved = True

            # Assemble new list
            # We need to ensure distinctness
            seen_blocks = set()
            final_blocks = []
            final_block_triplets = []
            
            # First, add the replaced blocks
            for i in range(len(blocks)):
                if i in replacement_blocks:
                    nb = replacement_blocks[i]
                    nb_tuple = tuple(nb)
                    if nb_tuple not in seen_blocks:
                        seen_blocks.add(nb_tuple)
                        final_blocks.append(nb)
                        # Calculate triplets
                        t_ids = []
                        for tsub in itertools.combinations(nb, t):
                            idx = comb_to_idx[tsub]
                            t_ids.append(idx)
                        final_block_triplets.append(t_ids)
            
            # Then add the kept blocks
            for i in range(len(blocks)):
                if keep_flags[i] and i not in replacement_blocks:
                    b = blocks[i]
                    b_tuple = tuple(b)
                    if b_tuple not in seen_blocks:
                        seen_blocks.add(b_tuple)
                        final_blocks.append(b)
                        final_block_triplets.append(block_triplets[i])
            
            if len(final_blocks) < len(blocks):
                blocks = final_blocks
                block_triplets = final_block_triplets
                # Recalculate counts? 
                # Counts are tricky if we just swap.
                # It's safer to recalculate counts from scratch each iteration if N is small.
                count = np.zeros(n_t, dtype=np.int32)
                block_triplets = []
                for b in blocks:
                    t_ids = []
                    for tsub in itertools.combinations(b, t):
                        idx = comb_to_idx[tsub]
                        count[idx] += 1
                        t_ids.append(idx)
                    block_triplets.append(t_ids)
            else:
                # No reduction in block count, but maybe structure changed?
                # If no block was removed or replaced, improved is False.
                # If replacement happened but count didn't drop (e.g. 1-for-1 swap), 
                # we might loop forever if we don't track state.
                # However, the "covered_by_other" check ensures we only remove if redundant.
                # The "replacement" case: we replace B_i with B_new.
                # If B_new is not identical to any other, we still have N blocks.
                # This might not help reduce N, but might help future removals.
                # To prevent infinite loops, we should only accept replacement if it leads to 
                # a different configuration.
                # Since we shuffle available points, B_new is likely different.
                # But if N doesn't drop, we might oscillate.
                # Strategy: Only accept replacement if it allows removing another block in the SAME pass?
                # Or simply: If N doesn't drop, stop?
                # No, 1-for-1 swaps are crucial.
                # Let's limit iterations.
                if iteration > 100:
                    break
                # Update state even if N didn't drop?
                blocks = final_blocks
                block_triplets = final_block_triplets
                count = np.zeros(n_t, dtype=np.int32)
                for b in blocks:
                    for tsub in itertools.combinations(b, t):
                        idx = comb_to_idx[tsub]
                        count[idx] += 1
                # Re-compute block_triplets
                block_triplets = []
                for b in blocks:
                    t_ids = []
                    for tsub in itertools.combinations(b, t):
                        idx = comb_to_idx[tsub]
                        t_ids.append(idx)
                    block_triplets.append(t_ids)

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    attempts = 0
    while time.time() < DEADLINE - 30:
        blocks = greedy_construct(attempts)
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                if best_count <= 76:
                    break
        attempts += 1
    
    if best_blocks is None:
        best_blocks = greedy_construct(0)
        if best_blocks is None:
            return []

    if time.time() < DEADLINE - 5:
        optimized_blocks = optimize(best_blocks)
        if optimized_blocks and len(optimized_blocks) < len(best_blocks):
            best_blocks = optimized_blocks

    return best_blocks
