import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Enumerate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Enumerate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # 3. Precompute coverage relationships
    masks = [0] * num_k_subsets
    mask_indices = [[] for _ in range(num_k_subsets)]
    blocks_covering_t = [[] for _ in range(num_t_subsets)]

    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
            blocks_covering_t[idx].append(i)
        masks[i] = mask
        mask_indices[i] = indices

    best_blocks = None
    best_count = float('inf')

    # Run multiple random restarts
    # Greedy is now O(Steps * CandidatesPerStep) instead of O(Steps * AllBlocks)
    # CandidatesPerStep is ~364 (binom(14,3)) vs 75582.
    # This is ~200x faster. We can afford many more restarts.
    num_restarts = 500 

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Greedy Construction
        uncovered = (1 << num_t_subsets) - 1
        current_blocks_indices = []
        used_blocks = set()
        
        while uncovered and time.time() < deadline:
            # Find an uncovered t-subset index
            # Using lowest set bit for speed
            if uncovered == 0:
                break
            # Get index of lowest set bit
            # (uncovered & -uncovered) isolates the lowest set bit
            # bit_length() - 1 gives the index
            t_idx = (uncovered & -uncovered).bit_length() - 1
            
            # Get candidates: all k-blocks containing this t-subset
            candidates = blocks_covering_t[t_idx]
            
            max_gain = 0
            best_candidates = []
            
            for c in candidates:
                if c in used_blocks:
                    continue
                gain = (masks[c] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [c]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(c)
            
            if max_gain == 0:
                # Should not happen if uncovered != 0, because blocks_covering_t[t_idx]
                # must contain at least one block, and that block covers t_idx,
                # so gain >= 1.
                break
            
            # Pick random from best candidates
            chosen_idx = random.choice(best_candidates)
            current_blocks_indices.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]
            
            if uncovered == 0:
                break

        if uncovered != 0:
            continue

        # Local Search
        local_indices = list(current_blocks_indices)
        
        # Initialize coverage counts
        counts = [0] * num_t_subsets
        for bi in local_indices:
            for idx in mask_indices[bi]:
                counts[idx] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant blocks
            for pos in range(len(local_indices)):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]
                is_redundant = True
                for idx in mask_indices[bi]:
                    if counts[idx] < 2:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in mask_indices[bi]:
                        counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break # Restart loop

            if improved:
                continue

            # Phase 2: 1-for-1 Swaps
            n_blocks = len(local_indices)
            for pos in range(n_blocks):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]
                
                # Find t-subsets covered ONLY by bi
                critical_ts = []
                for idx in mask_indices[bi]:
                    if counts[idx] == 1:
                        critical_ts.append(idx)
                
                if not critical_ts:
                    # Redundant, remove
                    for idx in mask_indices[bi]:
                        counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break

                # Find a replacement block that covers all critical_ts
                # Intersect lists of blocks covering each critical t-subset
                candidates = set(blocks_covering_t[critical_ts[0]])
                
                for idx in critical_ts[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(blocks_covering_t[idx])
                
                if candidates:
                    valid_candidates = [c for c in candidates if c not in local_indices]
                    if valid_candidates:
                        new_bi = random.choice(valid_candidates)
                        # Perform swap
                        for idx in mask_indices[bi]:
                            counts[idx] -= 1
                        for idx in mask_indices[new_bi]:
                            counts[idx] += 1
                        local_indices[pos] = new_bi
                        improved = True
                        break

        # Update best
        if len(local_indices) < best_count:
            best_count = len(local_indices)
            best_blocks = [list(k_subsets[i]) for i in local_indices]

    if best_blocks is None:
        # Fallback
        if time.time() < deadline - 5:
            uncovered = (1 << num_t_subsets) - 1
            current_blocks_indices = []
            used_blocks = set()
            while uncovered:
                if time.time() > deadline: break
                t_idx = (uncovered & -uncovered).bit_length() - 1
                candidates = blocks_covering_t[t_idx]
                max_gain = 0
                best_candidates = []
                for c in candidates:
                    if c in used_blocks:
                        continue
                    gain = (masks[c] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [c]
                    elif gain == max_gain and gain > 0:
                        best_candidates.append(c)
                if max_gain == 0:
                    break
                chosen_idx = random.choice(best_candidates)
                current_blocks_indices.append(chosen_idx)
                used_blocks.add(chosen_idx)
                uncovered &= ~masks[chosen_idx]
            if uncovered == 0:
                best_blocks = [list(k_subsets[i]) for i in current_blocks_indices]
    
    return best_blocks
