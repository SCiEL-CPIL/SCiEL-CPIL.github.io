import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets and their coverage of t-subsets
    all_k_subsets = list(itertools.combinations(all_points, k))
    n_k_subsets = len(all_k_subsets)
    
    # cover_matrix[i] contains the indices of t-subsets covered by all_k_subsets[i]
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = triple_to_idx[triple]
            
    # Efficient way to build triple_to_blocks
    # Using a list of lists first is faster in Python for appending, then convert to numpy
    t2b_lists = [[] for _ in range(n_triples)]
    for i in range(n_k_subsets):
        for triple_idx in cover_matrix[i]:
            t2b_lists[triple_idx].append(i)
    triple_to_blocks = [np.array(lst, dtype=np.int32) for lst in t2b_lists]

    best_blocks = None
    best_len = float('inf')
    
    def fast_greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            # Slight randomization among top candidates for diversity
            if len(candidates) > 1 and rng.random() < 0.3:
                # Pick from top 20% of candidates
                top_n = max(1, len(candidates) // 5)
                chosen_idx = candidates[rng.randint(0, top_n - 1)]
            else:
                chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            blocks.append(chosen_idx)
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks

    def local_search(block_indices):
        # block_indices is a list of indices into all_k_subsets
        if not block_indices:
            return block_indices
            
        # Use a set for O(1) lookup of current blocks
        current_set = set(block_indices)
        
        # Initialize coverage counts
        coverage_counts = np.zeros(n_triples, dtype=np.int32)
        for idx in block_indices:
            coverage_counts[cover_matrix[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            
            # 1. Try to remove redundant blocks
            to_remove = []
            for idx in block_indices:
                # If no triple in this block is covered only by this block, it's redundant
                if not np.any(coverage_counts[cover_matrix[idx]] == 1):
                    to_remove.append(idx)
            
            if to_remove:
                for idx in to_remove:
                    current_set.remove(idx)
                    coverage_counts[cover_matrix[idx]] -= 1
                block_indices = [idx for idx in block_indices if idx not in to_remove]
                changed = True
                continue # Restart loop as coverage changed significantly
            
            # 2. Try to swap blocks - improved version with better candidate selection
            swap_found = False
            for i in range(len(block_indices)):
                if time.time() > deadline - 5:
                    break
                idx = block_indices[i]
                
                # Find unique triples in this block
                mask = cover_matrix[idx]
                unique_triples = mask[coverage_counts[mask] == 1]
                
                if len(unique_triples) == 0:
                    continue
                
                # Find candidate replacement blocks
                # A candidate must cover ALL unique triples of idx
                candidates = None
                for tri in unique_triples:
                    blocks_covering_tri = triple_to_blocks[tri]
                    if candidates is None:
                        candidates = set(blocks_covering_tri.tolist())
                    else:
                        candidates.intersection_update(blocks_covering_tri.tolist())
                    
                    if not candidates:
                        break
                
                if candidates is None or not candidates:
                    continue
                
                # Remove current block from candidates
                candidates.discard(idx)
                
                if candidates:
                    # Score candidates: prefer those that cover more already-covered triples (more overlap)
                    # and are not in current set
                    best_candidate = None
                    best_score = -1
                    for cand in list(candidates)[:50]:  # Limit to 50 candidates for speed
                        if cand in current_set:
                            continue
                        # Score: number of triples this candidate covers that are already covered (overlap)
                        cand_mask = cover_matrix[cand]
                        overlap = np.sum(coverage_counts[cand_mask] > 0)
                        # Also consider: how many new triples does it cover? (should be 0 for valid swap)
                        new_triples = np.sum(coverage_counts[cand_mask] == 0)
                        # We want new_triples == 0 for a valid swap
                        if new_triples > 0:
                            continue
                        score = overlap
                        if score > best_score:
                            best_score = score
                            best_candidate = cand
                    
                    if best_candidate is not None:
                        # Perform swap
                        current_set.remove(idx)
                        coverage_counts[cover_matrix[idx]] -= 1
                        
                        current_set.add(best_candidate)
                        coverage_counts[cover_matrix[best_candidate]] += 1
                        
                        block_indices[i] = best_candidate
                        changed = True
                        swap_found = True
                        break # Restart loop after swap to re-evaluate removals
            
            # 3. Try 2-for-2 swaps: replace two blocks with two different blocks
            if not swap_found and len(block_indices) >= 2:
                # Sample a few pairs to try
                n_pairs_to_try = min(20, len(block_indices) * (len(block_indices) - 1) // 2)
                pair_indices = list(itertools.combinations(range(len(block_indices)), 2))
                random.shuffle(pair_indices)
                
                for i, j in pair_indices[:n_pairs_to_try]:
                    if time.time() > deadline - 5:
                        break
                    
                    idx1 = block_indices[i]
                    idx2 = block_indices[j]
                    
                    # Find unique triples that would be lost if we remove both blocks
                    mask1 = cover_matrix[idx1]
                    mask2 = cover_matrix[idx2]
                    
                    # Unique triples for idx1 (covered only by idx1)
                    unique1 = mask1[coverage_counts[mask1] == 1]
                    # Unique triples for idx2 (covered only by idx2)
                    unique2 = mask2[coverage_counts[mask2] == 1]
                    
                    # Also consider triples covered by both but no others
                    both_covered = np.intersect1d(mask1, mask2)
                    unique_both = both_covered[coverage_counts[both_covered] == 2]
                    
                    all_unique = np.concatenate([unique1, unique2, unique_both])
                    if len(all_unique) == 0:
                        continue
                    
                    # Find candidates that cover all these unique triples
                    candidates = None
                    for tri in all_unique:
                        blocks_covering_tri = triple_to_blocks[tri]
                        if candidates is None:
                            candidates = set(blocks_covering_tri.tolist())
                        else:
                            candidates.intersection_update(blocks_covering_tri.tolist())
                        
                        if not candidates:
                            break
                    
                    if candidates is None or len(candidates) < 2:
                        continue
                    
                    # Remove current blocks from candidates
                    candidates.discard(idx1)
                    candidates.discard(idx2)
                    
                    if len(candidates) < 2:
                        continue
                    
                    # Pick two distinct candidates
                    cand_list = list(candidates)[:30]
                    if len(cand_list) < 2:
                        continue
                    
                    # Try a few pairs of candidates
                    cand_pairs = list(itertools.combinations(cand_list, 2))[:10]
                    random.shuffle(cand_pairs)
                    
                    for c1, c2 in cand_pairs:
                        if c1 in current_set or c2 in current_set:
                            continue
                        
                        # Check if this swap is valid: the two new blocks must cover all unique triples
                        # and not leave any triple uncovered
                        new_mask1 = cover_matrix[c1]
                        new_mask2 = cover_matrix[c2]
                        
                        # Simulate: remove idx1, idx2; add c1, c2
                        # Check coverage for all triples in the union of affected sets
                        affected = np.union1d(np.union1d(mask1, mask2), np.union1d(new_mask1, new_mask2))
                        
                        # Compute new coverage counts for affected triples
                        old_counts = coverage_counts[affected].copy()
                        
                        # Remove contributions from idx1 and idx2
                        for tri in affected:
                            if tri in mask1:
                                old_counts[np.where(affected == tri)[0][0]] -= 1
                            if tri in mask2:
                                old_counts[np.where(affected == tri)[0][0]] -= 1
                            if tri in new_mask1:
                                old_counts[np.where(affected == tri)[0][0]] += 1
                            if tri in new_mask2:
                                old_counts[np.where(affected == tri)[0][0]] += 1
                        
                        if np.any(old_counts < 1):
                            continue
                        
                        # Valid swap! Perform it
                        current_set.remove(idx1)
                        current_set.remove(idx2)
                        coverage_counts[mask1] -= 1
                        coverage_counts[mask2] -= 1
                        
                        current_set.add(c1)
                        current_set.add(c2)
                        coverage_counts[new_mask1] += 1
                        coverage_counts[new_mask2] += 1
                        
                        block_indices[i] = c1
                        block_indices[j] = c2
                        changed = True
                        swap_found = True
                        break
                    
                    if swap_found:
                        break
            
            if not changed:
                break
                
        return block_indices

    # Main loop with adaptive restarts
    restart_count = 0
    no_improve_count = 0
    while time.time() < deadline - 5:
        block_indices = fast_greedy_cover(restart_count)
        block_indices = local_search(block_indices)
        
        if len(block_indices) < best_len:
            best_len = len(block_indices)
            best_blocks = block_indices[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        restart_count += 1
        
        # If we've had many restarts without improvement, try a different strategy
        if no_improve_count > 20 and best_blocks is not None:
            # Perturb the best solution and do local search
            perturbed = best_blocks[:]
            rng = random.Random(restart_count)
            # Remove 1-2 random blocks and re-add via greedy
            n_remove = min(2, len(perturbed))
            for _ in range(n_remove):
                idx = rng.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
            
            # Re-greedy to fill gaps
            uncovered = np.ones(n_triples, dtype=bool)
            for idx in perturbed:
                uncovered[cover_matrix[idx]] = False
            
            while uncovered.any():
                gains = np.sum(uncovered[cover_matrix], axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
                perturbed.append(chosen_idx)
                uncovered[cover_matrix[chosen_idx]] = False
            
            perturbed = local_search(perturbed)
            
            if len(perturbed) < best_len:
                best_len = len(perturbed)
                best_blocks = perturbed[:]
                no_improve_count = 0
        
    return [list(all_k_subsets[i]) for i in best_blocks] if best_blocks is not None else []
