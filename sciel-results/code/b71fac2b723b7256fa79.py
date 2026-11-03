import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Number of triples in a block
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Using a 2D numpy array for vectorized operations
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)
    
    # Precompute for each triple, which blocks cover it (for faster candidate finding)
    # triple_to_blocks[triple_idx] = list of block indices covering this triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx].append(i)
    
    def greedy_cover(use_weighted=False, random_tiebreak=False, rng=None, top_k=1):
        """Run greedy set cover."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        selected_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Calculate gains, penalizing blocks already selected
            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weight[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
            
            # Penalize already-selected blocks to encourage diversity
            for sidx in selected_set:
                gains[sidx] = 0
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if random_tiebreak and rng is not None:
                # Pick from top K candidates
                threshold = max_gain - 1e-9
                candidates = np.where(gains >= threshold)[0]
                if len(candidates) > 0:
                    if len(candidates) > top_k:
                        best_idx = rng.choice(candidates)
                    else:
                        best_idx = rng.choice(candidates)
                else:
                    best_idx = np.argmax(gains)
            else:
                best_idx = np.argmax(gains)
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected

    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution using incremental coverage counts."""
        if not block_indices:
            return list(block_indices)
        
        # coverage_count[triple] = number of blocks covering it
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_indices = list(block_indices)
        
        # Iterative removal
        while True:
            if time.time() > deadline:
                break
                
            removed_any = False
            new_indices = []
            
            for bidx in current_indices:
                my_triples = block_triples[bidx]
                # Check if any triple in this block is covered only by this block
                if np.min(coverage_count[my_triples]) > 1:
                    # Redundant, remove it
                    coverage_count[my_triples] -= 1
                    removed_any = True
                else:
                    new_indices.append(bidx)
            
            current_indices = new_indices
            
            if not removed_any:
                break
                
        return current_indices

    def try_merge_two_blocks(block_a, block_b, coverage_count, current_blocks):
        """
        Try to replace two blocks with one block that covers all unique triples of both.
        Returns True if merge was successful, False otherwise.
        """
        # Get the union of triples covered by both blocks
        triples_a = block_triples[block_a]
        triples_b = block_triples[block_b]
        
        # Unique triples: those covered by exactly one of the two blocks
        # A triple is "unique to the pair" if it's covered by a or b but not by any other block
        # Actually, we need: after removing both a and b, which triples become uncovered?
        # Those are triples where coverage_count == 1 and the only covering block is a or b.
        
        # Temporarily remove both blocks
        coverage_count[triples_a] -= 1
        coverage_count[triples_b] -= 1
        
        # Find triples that are now uncovered
        uncovered_after_removal = (coverage_count == 0)
        
        # Restore
        coverage_count[triples_a] += 1
        coverage_count[triples_b] += 1
        
        # If no triples become uncovered, we can simply remove one block (handled by peel)
        # If some triples become uncovered, we need a single block that covers all of them
        # AND all triples that were uniquely covered by a or b
        
        # The set of triples that MUST be covered by the new block:
        # - All triples that were uniquely covered by a (coverage_count==1, covered by a)
        # - All triples that were uniquely covered by b (coverage_count==1, covered by b)
        
        # But wait, we need to be more careful. Let's think again.
        # When we remove both a and b, some triples may lose their only coverage.
        # We need to find a single block that covers ALL such triples.
        
        # Get triples uniquely covered by a
        unique_to_a = triples_a[coverage_count[triples_a] == 1]
        # Get triples uniquely covered by b  
        unique_to_b = triples_b[coverage_count[triples_b] == 1]
        
        # The new block must cover the union of unique_to_a and unique_to_b
        # But also, any triple that is covered by both a and b but by no other block
        # would be lost. Let's handle this more carefully.
        
        # Actually, the correct approach:
        # After removing a and b, the uncovered triples are those with coverage_count == 0.
        # We computed that above. Let's redo it properly.
        
        # Save current coverage
        save_count = coverage_count.copy()
        coverage_count[triples_a] -= 1
        coverage_count[triples_b] -= 1
        must_cover = np.where(coverage_count == 0)[0]
        coverage_count[:] = save_count
        
        if len(must_cover) == 0:
            # No triples become uncovered, so we can just remove one block
            # This will be handled by peel_redundant
            return False
        
        # Now find a block that covers all triples in must_cover
        # must_cover is a set of triple indices
        # We need to find a block b such that all triples in must_cover are in block_triples[b]
        
        # Strategy: For each triple in must_cover, get the list of blocks covering it.
        # The intersection of these lists gives candidate blocks.
        
        if len(must_cover) > 10:
            # Too many constraints, skip (unlikely to find a single block)
            return False
        
        # Get candidate blocks as intersection of block lists for each must_cover triple
        candidate_blocks = None
        for triple_idx in must_cover:
            blocks_covering = triple_to_blocks[triple_idx]
            if candidate_blocks is None:
                candidate_blocks = set(blocks_covering)
            else:
                candidate_blocks = candidate_blocks.intersection(set(blocks_covering))
            if not candidate_blocks:
                break
        
        if not candidate_blocks:
            return False
        
        # Remove a and b from candidates
        candidate_blocks.discard(block_a)
        candidate_blocks.discard(block_b)
        
        if not candidate_blocks:
            return False
        
        # Pick a random candidate
        cand_list = list(candidate_blocks)
        new_block = random.choice(cand_list)
        
        # Verify: does new_block cover all must_cover triples?
        # (Should be guaranteed by construction, but let's be safe)
        new_triples = block_triples[new_block]
        for tc in must_cover:
            if tc not in new_triples:
                # This shouldn't happen, but bail out
                return False
        
        # Perform the merge: remove a and b, add new_block
        coverage_count[triples_a] -= 1
        coverage_count[triples_b] -= 1
        coverage_count[new_triples] += 1
        
        # Update current_blocks
        # Remove block_a and block_b
        new_current = []
        for b in current_blocks:
            if b != block_a and b != block_b:
                new_current.append(b)
        new_current.append(new_block)
        
        current_blocks.clear()
        current_blocks.extend(new_current)
        
        return True

    def local_search(block_indices, rng, time_limit):
        """
        Local search to minimize block count.
        Moves:
        1. Remove redundant blocks.
        2. Swap a block with another that covers the same 'critical' triples but has better overlap.
        3. Merge: Replace two blocks with one that covers all their unique triples.
        4. Kick: Remove a random block, repair greedily.
        """
        if not block_indices:
            return list(block_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        # Helper to update best
        def update_best():
            nonlocal best_blocks, best_size
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
        
        no_progress_count = 0
        max_no_progress = 50
        
        while time.time() < time_limit:
            # 1. Try to remove redundant blocks
            peeled = peel_redundant(current_blocks)
            if len(peeled) < len(current_blocks):
                current_blocks = peeled
                # Rebuild coverage counts
                coverage_count = np.zeros(n_triples, dtype=np.int32)
                for bidx in current_blocks:
                    coverage_count[block_triples[bidx]] += 1
                update_best()
                no_progress_count = 0
                continue
            
            # 2. Try merge moves
            merge_success = False
            if len(current_blocks) >= 2:
                # Try random pairs
                for _ in range(20):
                    i = rng.randrange(len(current_blocks))
                    j = rng.randrange(len(current_blocks))
                    if i == j:
                        continue
                    if try_merge_two_blocks(current_blocks[i], current_blocks[j], 
                                            coverage_count, current_blocks):
                        merge_success = True
                        update_best()
                        no_progress_count = 0
                        break
            
            if merge_success:
                continue
            
            # 3. Try swaps
            candidates_for_swap = []
            for bidx in current_blocks:
                my_triples = block_triples[bidx]
                if np.any(coverage_count[my_triples] == 1):
                    candidates_for_swap.append(bidx)
            
            if candidates_for_swap:
                old_bidx = rng.choice(candidates_for_swap)
                old_triples = block_triples[old_bidx]
                critical_triples = old_triples[coverage_count[old_triples] == 1]
                
                if len(critical_triples) > 0:
                    best_candidate = -1
                    best_score = -1
                    
                    # Use triple_to_blocks for faster candidate finding
                    # Find blocks that cover all critical_triples
                    if len(critical_triples) <= 5:
                        candidate_set = None
                        for tc in critical_triples:
                            blocks_covering = triple_to_blocks[tc]
                            if candidate_set is None:
                                candidate_set = set(blocks_covering)
                            else:
                                candidate_set = candidate_set.intersection(set(blocks_covering))
                            if not candidate_set:
                                break
                        
                        if candidate_set:
                            candidate_set.discard(old_bidx)
                            cand_list = list(candidate_set)
                            # Sample up to 50 candidates
                            if len(cand_list) > 50:
                                cand_list = rng.sample(cand_list, 50)
                            
                            for cand_idx in cand_list:
                                cand_triples = block_triples[cand_idx]
                                overlap_score = np.sum(coverage_count[cand_triples] >= 2)
                                if overlap_score > best_score:
                                    best_score = overlap_score
                                    best_candidate = cand_idx
                    else:
                        # Fallback to random sampling
                        for _ in range(100):
                            cand_idx = rng.randrange(n_blocks)
                            if cand_idx == old_bidx:
                                continue
                            cand_triples = block_triples[cand_idx]
                            if not np.all(np.isin(critical_triples, cand_triples)):
                                continue
                            overlap_score = np.sum(coverage_count[cand_triples] >= 2)
                            if overlap_score > best_score:
                                best_score = overlap_score
                                best_candidate = cand_idx
                    
                    if best_candidate >= 0:
                        coverage_count[old_triples] -= 1
                        coverage_count[block_triples[best_candidate]] += 1
                        for i, b in enumerate(current_blocks):
                            if b == old_bidx:
                                current_blocks[i] = best_candidate
                                break
                        no_progress_count = 0
                        continue
            
            # 4. Kick
            if rng.random() < 0.15 or no_progress_count > max_no_progress:
                if len(current_blocks) > 1:
                    j = rng.randrange(len(current_blocks))
                    removed_bidx = current_blocks[j]
                    coverage_count[block_triples[removed_bidx]] -= 1
                    current_blocks.pop(j)
                    
                    uncovered_mask = (coverage_count == 0)
                    while np.any(uncovered_mask):
                        if time.time() > time_limit:
                            break
                        gains = np.sum(uncovered_mask[block_triples], axis=1)
                        best_add = np.argmax(gains)
                        if gains[best_add] == 0:
                            break
                        coverage_count[block_triples[best_add]] += 1
                        current_blocks.append(best_add)
                        uncovered_mask = (coverage_count == 0)
                    
                    peeled = peel_redundant(current_blocks)
                    if len(peeled) < len(current_blocks):
                        current_blocks = peeled
                        coverage_count = np.zeros(n_triples, dtype=np.int32)
                        for bidx in current_blocks:
                            coverage_count[block_triples[bidx]] += 1
                        update_best()
                        no_progress_count = 0
                    else:
                        no_progress_count += 1
                else:
                    no_progress_count += 1
            else:
                no_progress_count += 1
        
        return best_blocks

    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Strategy 1: Deterministic weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = peeled
            
    # Strategy 2: Deterministic unweighted greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = peeled
            
    # Strategy 3+: Random restarts with weighted greedy + Local Search
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    # Run LS for a few restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Randomized greedy
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng, top_k=5)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply Local Search
            ls_time_limit = deadline - 5.0
            ls_result = local_search(peeled, rng, ls_time_limit)
            ls_size = len(ls_result)
            
            if ls_size < size:
                size = ls_size
                peeled = ls_result
            
            if size < best_size:
                best_size = size
                best_solution = peeled
                
    # Final polish
    if best_solution is not None:
        final_peeled = peel_redundant(best_solution)
        best_solution = final_peeled
        best_size = len(final_peeled)
        
    if best_solution is None:
        best_solution = []
        
    # Convert indices to lists of points
    return [list(all_blocks[i]) for i in best_solution]
