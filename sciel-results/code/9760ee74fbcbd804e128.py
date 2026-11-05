import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_masks[i] is a bitmask of t-subset indices covered by block i
    block_masks = [0] * n_k_blocks
    # Also store the list of t-subset indices for each block for fast iteration
    block_t_indices = [[] for _ in range(n_k_blocks)]
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
        
    # Precompute for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(n_t_subs)]
    for i in range(n_k_blocks):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)

    # 3. Dominance pruning: remove blocks whose coverage is a subset of another block's coverage
    # A block i is dominated by block j if block_masks[i] | block_masks[j] == block_masks[j]
    # This means every t-subset covered by i is also covered by j.
    # We can safely remove i.
    
    # Efficient check: for each pair, check if mask_i is subset of mask_j
    # mask_i & ~mask_j == 0 means i is dominated by j
    
    # First pass: remove blocks that are subsets of another single block
    keep_blocks = [True] * n_k_blocks
    
    # Sort blocks by number of covered t-subsets (descending) so larger blocks come first
    # This helps: if block i has fewer t-subsets than block j, i could be dominated by j
    # But we need to check all pairs where |mask_i| <= |mask_j|
    
    # Compute size of each block's coverage
    block_sizes = [mask.bit_count() for mask in block_masks]
    
    # Sort indices by size descending
    sorted_indices = sorted(range(n_k_blocks), key=lambda x: block_sizes[x], reverse=True)
    
    # For each block in order, check if it's dominated by any previously kept block
    # This is O(n^2) in worst case but with early termination
    # n_k_blocks = C(19,7) = 50388, so n^2 is ~2.5 billion which is too slow
    # Need a smarter approach
    
    # Alternative: Only check dominance for blocks that share at least one t-subset
    # Use the t_to_blocks structure
    # For each t-subset, all blocks covering it are candidates for dominance relationships
    
    # Even better: two-cover dominance
    # Block i is 2-dominated if there exist blocks j, k such that mask_i | mask_j | mask_k == mask_j | mask_k
    # This is even harder to check
    
    # Let's use a simpler but effective pruning:
    # Remove blocks that are exact subsets of another block's coverage
    # Use bitset operations with numpy for speed
    
    # Convert masks to numpy arrays for vectorized subset checks
    # But Python ints with bit_count are already fast
    
    # Let's do pairwise dominance check with early termination
    # Only check pairs where block_sizes[i] <= block_sizes[j]
    
    # Group blocks by size to reduce comparisons
    from collections import defaultdict
    size_groups = defaultdict(list)
    for i in range(n_k_blocks):
        size_groups[block_sizes[i]].append(i)
    
    sizes_sorted = sorted(size_groups.keys(), reverse=True)
    
    # For each size s, check blocks of size s against blocks of size >= s
    # This is still potentially expensive
    
    # Let's use a different strategy: random sampling for dominance
    # For each block, sample a few other blocks and check if dominated
    
    # Actually, let's just do the full pairwise check but with a time limit
    # and break early if we find dominance
    
    pruning_deadline = time.time() + 30  # Spend at most 30s on pruning
    
    # Build a list of blocks sorted by size descending
    # For each block i, check against blocks j with size >= size_i
    # Stop checking j once we've checked enough
    
    max_checks_per_block = 100  # Limit checks per block for speed
    
    removed_count = 0
    for i in range(n_k_blocks):
        if time.time() > pruning_deadline:
            break
        if not keep_blocks[i]:
            continue
        
        mask_i = block_masks[i]
        size_i = block_sizes[i]
        
        # Check against a random sample of larger or equal sized blocks
        candidates = [j for j in range(n_k_blocks) if j != i and keep_blocks[j] and block_sizes[j] >= size_i]
        
        # Sample up to max_checks_per_block candidates
        if len(candidates) > max_checks_per_block:
            random.shuffle(candidates)
            candidates = candidates[:max_checks_per_block]
        
        for j in candidates:
            if time.time() > pruning_deadline:
                break
            if (mask_i & ~block_masks[j]) == 0:
                # Block i is dominated by block j
                keep_blocks[i] = False
                removed_count += 1
                break
    
    # Rebuild data structures with only kept blocks
    if removed_count > 0:
        new_block_list = []
        new_mask_list = []
        new_t_indices_list = []
        old_to_new = {}
        
        for i in range(n_k_blocks):
            if keep_blocks[i]:
                old_to_new[i] = len(new_block_list)
                new_block_list.append(all_k_blocks[i])
                new_mask_list.append(block_masks[i])
                new_t_indices_list.append(block_t_indices[i])
        
        # Remap t_to_blocks
        new_t_to_blocks = [[] for _ in range(n_t_subs)]
        for old_i, new_i in old_to_new.items():
            for idx in block_t_indices[old_i]:
                new_t_to_blocks[idx].append(new_i)
        
        all_k_blocks = new_block_list
        block_masks = new_mask_list
        block_t_indices = new_t_indices_list
        t_to_blocks = new_t_to_blocks
        n_k_blocks = len(all_k_blocks)
        
        # Recompute block_sizes
        block_sizes = [mask.bit_count() for mask in block_masks]

    full_mask = (1 << n_t_subs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper: Fast Greedy construction using bitmask operations
    def greedy_construct(randomize=False, weight_mode='uniform'):
        """
        Greedy set cover with optional randomization.
        weight_mode: 'uniform', 'inverse_freq', 'inverse_sqrt'
        """
        if time.time() > deadline - 20:
            return []
        
        uncovered_mask = full_mask
        selected = []
        
        # Precompute weights if needed
        if weight_mode != 'uniform':
            # Count how many times each t-subset appears across all blocks
            # This is expensive, so only compute if needed
            pass
        
        while uncovered_mask != 0:
            if time.time() > deadline - 20:
                break
            
            best_idx = -1
            best_gain = -1
            best_candidates = []
            
            # For each block, compute gain = number of newly covered t-subsets
            # Use bitmask AND + bit_count for speed
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            if randomize and len(best_candidates) > 1:
                # Use inverse frequency weighting for randomization
                # Prefer blocks that cover rarer t-subsets
                if weight_mode == 'inverse_freq':
                    # Compute weight based on rarity of uncovered t-subsets
                    weights = []
                    for c in best_candidates:
                        w = 0
                        for idx in block_t_indices[c]:
                            if (uncovered_mask >> idx) & 1:
                                w += 1.0 / len(t_to_blocks[idx])
                        weights.append(w)
                    # Weighted random choice
                    total_w = sum(weights)
                    if total_w > 0:
                        r = random.random() * total_w
                        cumulative = 0
                        for idx_c, w in zip(best_candidates, weights):
                            cumulative += w
                            if r <= cumulative:
                                best_idx = idx_c
                                break
                    else:
                        best_idx = random.choice(best_candidates)
                else:
                    best_idx = random.choice(best_candidates)
            
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        
        return selected

    # Helper: Local Search with efficient 2-1 exchanges
    def local_search(selected):
        if not selected:
            return selected
        
        current = list(selected)
        
        # Build cover_count
        cover_count = [0] * n_t_subs
        for bi in current:
            for idx in block_t_indices[bi]:
                cover_count[idx] += 1
        
        # Set of selected blocks for fast lookup
        selected_set = set(current)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Phase 1: Remove redundant blocks
            new_current = []
            for bi in current:
                is_redundant = True
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_t_indices[bi]:
                        cover_count[idx] -= 1
                    selected_set.remove(bi)
                    improved = True
                else:
                    new_current.append(bi)
            current = new_current
            
            if time.time() > deadline - 10:
                break
                
            # Phase 2: 2-1 Exchange (remove 2 blocks, add 1 if needed)
            n = len(current)
            if n < 2:
                continue
                
            # Build a list of (block_idx, its_t_indices) for quick access
            # We'll iterate over pairs in random order
            pair_list = []
            for i in range(n):
                for j in range(i+1, n):
                    pair_list.append((i, j))
            random.shuffle(pair_list)
            
            found_improvement = False
            for i, j in pair_list:
                if time.time() > deadline - 10:
                    break
                if i >= len(current) or j >= len(current):
                    continue
                
                bi = current[i]
                bj = current[j]
                
                # Decrement counts for both blocks
                for idx in block_t_indices[bi]:
                    cover_count[idx] -= 1
                for idx in block_t_indices[bj]:
                    cover_count[idx] -= 1
                
                # Check if any t-subset became uncovered
                uncovered_indices = []
                # Use a set to avoid duplicates
                uncovered_set = set()
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 0:
                        uncovered_set.add(idx)
                for idx in block_t_indices[bj]:
                    if cover_count[idx] == 0:
                        uncovered_set.add(idx)
                
                if not uncovered_set:
                    # Can remove both blocks
                    current.remove(bj)
                    current.remove(bi)
                    selected_set.remove(bj)
                    selected_set.remove(bi)
                    improved = True
                    found_improvement = True
                    break
                else:
                    # Try to find a block bk that covers all uncovered_indices
                    # and is not bi or bj
                    uncovered_indices = list(uncovered_set)
                    # Sort uncovered by number of covering blocks (rarity first)
                    uncovered_indices.sort(key=lambda x: len(t_to_blocks[x]))
                    
                    candidates = None
                    for idx in uncovered_indices:
                        blocks_covering = t_to_blocks[idx]
                        if candidates is None:
                            candidates = set(blocks_covering)
                        else:
                            candidates.intersection_update(blocks_covering)
                        if not candidates:
                            break
                    
                    if candidates:
                        # Filter: bk must not be in selected_set (to avoid adding duplicate)
                        # and must not be bi or bj
                        valid = [bk for bk in candidates if bk not in selected_set and bk != bi and bk != bj]
                        
                        if valid:
                            bk = random.choice(valid)
                            
                            # Remove bi, bj
                            current.remove(bj)
                            current.remove(bi)
                            selected_set.remove(bj)
                            selected_set.remove(bi)
                            
                            # Add bk
                            current.append(bk)
                            selected_set.add(bk)
                            
                            # Update cover_count: we already decremented bi, bj
                            # Now increment for bk
                            for idx in block_t_indices[bk]:
                                cover_count[idx] += 1
                            
                            improved = True
                            found_improvement = True
                            break
                    else:
                        # No valid replacement, restore counts
                        for idx in block_t_indices[bi]:
                            cover_count[idx] += 1
                        for idx in block_t_indices[bj]:
                            cover_count[idx] += 1
                
            if found_improvement:
                continue
            
            # Phase 3: 1-1 Exchange (replace one block with another)
            if time.time() < deadline - 10:
                n = len(current)
                if n >= 1:
                    # For each selected block, try replacing it with a non-selected block
                    # that covers all t-subsets that would become uncovered
                    replacement_found = False
                    for pos in range(n):
                        if time.time() > deadline - 10:
                            break
                        bi = current[pos]
                        
                        # Find t-subsets that only bi covers
                        only_bi_indices = []
                        for idx in block_t_indices[bi]:
                            if cover_count[idx] == 1:
                                only_bi_indices.append(idx)
                        
                        if not only_bi_indices:
                            # bi is redundant, should have been caught in Phase 1
                            continue
                        
                        # Find blocks that cover all only_bi_indices
                        # Sort by rarity
                        only_bi_indices.sort(key=lambda x: len(t_to_blocks[x]))
                        
                        candidates = None
                        for idx in only_bi_indices:
                            blocks_covering = t_to_blocks[idx]
                            if candidates is None:
                                candidates = set(blocks_covering)
                            else:
                                candidates.intersection_update(blocks_covering)
                            if not candidates:
                                break
                        
                        if candidates:
                            # Filter: must not be in selected_set
                            valid = [bk for bk in candidates if bk not in selected_set]
                            if valid:
                                # Choose the one with best "efficiency" (covers most new t-subsets)
                                best_bk = None
                                best_gain = -1
                                for bk in valid:
                                    gain = 0
                                    for idx in block_t_indices[bk]:
                                        if cover_count[idx] == 0:
                                            gain += 1
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_bk = bk
                                
                                if best_bk is not None:
                                    # Replace bi with best_bk
                                    # First, decrement bi
                                    for idx in block_t_indices[bi]:
                                        cover_count[idx] -= 1
                                    # Then, increment best_bk
                                    for idx in block_t_indices[best_bk]:
                                        cover_count[idx] += 1
                                    
                                    current[pos] = best_bk
                                    selected_set.remove(bi)
                                    selected_set.add(best_bk)
                                    improved = True
                                    replacement_found = True
                                    break
                    
                    if replacement_found:
                        continue
        
        return current

    # Main Loop: Multiple Restarts with diverse weightings
    max_restarts = 20
    weight_modes = ['uniform', 'inverse_freq', 'inverse_sqrt']
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Cycle through weight modes for diversity
        wm = weight_modes[r % len(weight_modes)]
        
        # Greedy with randomization
        selected = greedy_construct(randomize=True, weight_mode=wm)
        
        if not selected:
            continue
        
        # Local Search
        selected = local_search(selected)
        
        # Verify and update best
        cov = 0
        for bi in selected:
            cov |= block_masks[bi]
        if cov == full_mask and len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)
            
        # If we have a good solution, spend extra time polishing it
        if best_count <= 152 and time.time() < deadline - 30:
            # Do additional local search passes on the best solution
            polished = local_search(list(best_blocks))
            cov2 = 0
            for bi in polished:
                cov2 |= block_masks[bi]
            if cov2 == full_mask and len(polished) < best_count:
                best_count = len(polished)
                best_blocks = list(polished)

    if best_blocks is None:
        # Fallback
        selected = greedy_construct(randomize=False)
        selected = local_search(selected)
        best_blocks = selected

    return [list(all_k_blocks[i]) for i in best_blocks]
