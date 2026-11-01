# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    rng = random.Random(42)

    # Precompute all t-subsets as bitmasks
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    n_tsub = len(tsub_masks)
    idx_of = {m: i for i, m in enumerate(tsub_masks)}

    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)

    # Precompute t-subset indices for each k-subset
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            indices.append(idx_of[m])
        k_sub_tsub_indices.append(indices)

    # Build mask-to-index lookup for k-subsets
    mask_to_idx = {}
    for i, comb in enumerate(all_k_subsets):
        m = 0
        for p in comb:
            m |= (1 << p)
        mask_to_idx[m] = i

    # Flat array of t-subset indices with offsets
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)

    def greedy_cover(uncovered, deadline):
        """Greedy set cover: repeatedly pick block with max gain."""
        blocks = []
        selected_set = set()
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        
        while uncovered.any() and time.time() < deadline:
            gains = np.zeros(n_ksub, dtype=np.int32)
            for i in range(n_ksub):
                start = offsets_arr[i]
                end = offsets_arr[i + 1]
                if start < end:
                    gains[i] = np.sum(uncovered[all_indices_arr[start:end]])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[0]
            blocks.append(best_idx)
            selected_set.add(best_idx)
            
            start = offsets_arr[best_idx]
            end = offsets_arr[best_idx + 1]
            idxs = all_indices_arr[start:end]
            cover_count[idxs] += 1
            uncovered[idxs] = False
        
        return blocks, selected_set, cover_count

    def try_redundant_removal(blocks, selected_set, cover_count, deadline):
        """Remove any redundant blocks (those with no t-subset uniquely covered)."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                start = offsets_arr[b_idx]
                end = offsets_arr[b_idx + 1]
                idxs = all_indices_arr[start:end]
                if np.any(cover_count[idxs] == 1):
                    continue  # Not redundant
                # Remove
                cover_count[idxs] -= 1
                blocks.pop(i)
                selected_set.remove(b_idx)
                changed = True
                break
        return blocks, selected_set, cover_count

    def try_k_for_k1_exchange(blocks, selected_set, cover_count, deadline):
        """
        Try to remove k blocks and add k-1 new blocks (net reduction of 1).
        Actually: remove k blocks, add k-1 blocks => net -1.
        Strategy: pick k blocks whose union of covered t-subsets can be covered
        by k-1 other blocks (not in selected_set).
        
        For efficiency, we try small k values (2, 3, 4).
        """
        improved = False
        for k_remove in [2, 3, 4]:
            if time.time() > deadline:
                break
            if len(blocks) < k_remove:
                break
            
            # Sample random subsets of k_remove blocks to try
            n_trials = 50
            for trial in range(n_trials):
                if time.time() > deadline:
                    break
                
                # Randomly select k_remove blocks to remove
                to_remove = rng.sample(range(len(blocks)), k_remove)
                remove_set = set(blocks[i] for i in to_remove)
                
                # Find all t-subsets that become uncovered if we remove these blocks
                # A t-subset is affected if its cover_count would drop to 0
                affected_tsubs = set()
                for bi in to_remove:
                    b_idx = blocks[bi]
                    start = offsets_arr[b_idx]
                    end = offsets_arr[b_idx + 1]
                    idxs = all_indices_arr[start:end]
                    for idx in idxs:
                        if cover_count[idx] == 1:
                            affected_tsubs.add(int(idx))
                
                if not affected_tsubs:
                    continue  # No t-subset becomes uncovered; just remove redundantly
                
                # Try to find k_remove - 1 blocks that cover all affected t-subsets
                n_add = k_remove - 1
                # Candidate blocks: those not in selected_set that cover at least one affected t-subset
                # For speed, we'll try random candidates
                
                # Build a list of candidate block indices that could help
                # A block helps if it covers at least one affected t-subset
                # We'll sample candidates
                
                found_cover = False
                for add_trial in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Pick random candidate blocks
                    add_candidates = []
                    for _ in range(n_add * 5):
                        cand = rng.randint(0, n_ksub - 1)
                        if cand in selected_set:
                            continue
                        # Check if this block covers any affected t-subset
                        start = offsets_arr[cand]
                        end = offsets_arr[cand + 1]
                        idxs = all_indices_arr[start:end]
                        covers_affected = False
                        for idx in idxs:
                            if int(idx) in affected_tsubs:
                                covers_affected = True
                                break
                        if covers_affected:
                            add_candidates.append(cand)
                    
                    if len(add_candidates) < n_add:
                        continue
                    
                    # Pick n_add from candidates
                    add_blocks = rng.sample(add_candidates, n_add)
                    
                    # Verify: do these add_blocks cover all affected t-subsets?
                    covered_by_add = set()
                    for ab in add_blocks:
                        start = offsets_arr[ab]
                        end = offsets_arr[ab + 1]
                        idxs = all_indices_arr[start:end]
                        for idx in idxs:
                            covered_by_add.add(int(idx))
                    
                    if affected_tsubs.issubset(covered_by_add):
                        # Valid exchange! Apply it.
                        # Remove the k_remove blocks
                        for bi in sorted(to_remove, reverse=True):
                            b_idx = blocks[bi]
                            start = offsets_arr[b_idx]
                            end = offsets_arr[b_idx + 1]
                            idxs = all_indices_arr[start:end]
                            cover_count[idxs] -= 1
                            blocks.pop(bi)
                            selected_set.remove(b_idx)
                        
                        # Add the new blocks
                        for ab in add_blocks:
                            blocks.append(ab)
                            selected_set.add(ab)
                            start = offsets_arr[ab]
                            end = offsets_arr[ab + 1]
                            idxs = all_indices_arr[start:end]
                            cover_count[idxs] += 1
                        
                        found_cover = True
                        improved = True
                        break
                
                if found_cover:
                    break
        
        return improved

    # Phase 1: Greedy initialization
    uncovered = np.ones(n_tsub, dtype=bool)
    blocks, selected_set, cover_count = greedy_cover(uncovered, deadline - 30.0)
    
    # Phase 2: Remove redundant blocks
    blocks, selected_set, cover_count = try_redundant_removal(blocks, selected_set, cover_count, deadline - 20.0)
    
    # Phase 3: Local search with k-for-(k-1) exchanges
    best_blocks = list(blocks)
    best_count = len(blocks)
    best_selected = set(selected_set)
    best_cover = cover_count.copy()
    
    no_improve = 0
    while time.time() < deadline - 5.0:
        # Try redundant removal
        blocks, selected_set, cover_count = try_redundant_removal(blocks, selected_set, cover_count, deadline - 5.0)
        
        # Try exchanges
        improved = try_k_for_k1_exchange(blocks, selected_set, cover_count, deadline - 5.0)
        
        if len(blocks) < best_count:
            best_blocks = list(blocks)
            best_count = len(blocks)
            best_selected = set(selected_set)
            best_cover = cover_count.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                break
    
    # Use best found
    blocks = best_blocks
    selected_set = best_selected
    cover_count = best_cover
    
    # Final redundant removal
    blocks, selected_set, cover_count = try_redundant_removal(blocks, selected_set, cover_count, deadline - 2.0)
    
    # Verify and repair if needed
    uncovered = np.ones(n_tsub, dtype=bool)
    for b_idx in blocks:
        start = offsets_arr[b_idx]
        end = offsets_arr[b_idx + 1]
        idxs = all_indices_arr[start:end]
        uncovered[idxs] = False
    
    uncovered_indices = np.where(uncovered)[0]
    for idx in uncovered_indices:
        if time.time() > deadline - 1.0:
            break
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        if need > 0:
            extra = rng.sample(remaining_points, need)
        else:
            extra = []
        block = have + extra
        block.sort()
        mask = 0
        for p in block:
            mask |= (1 << p)
        b_idx = mask_to_idx.get(mask)
        if b_idx is not None and b_idx not in selected_set:
            blocks.append(b_idx)
            selected_set.add(b_idx)
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx + 1]
            idxs = all_indices_arr[start:end]
            cover_count[idxs] += 1
            uncovered[idxs] = False
    
    # Final result
    result = []
    for idx in blocks:
        result.append(list(all_k_subsets[idx]))
    
    return result

# EVOLVE-BLOCK-END
