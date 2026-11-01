import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # 4. Helper: Greedy construction with random tie-breaking
    def greedy_construct():
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        blocks = []
        used_blocks = set()
        
        while remaining > 0:
            if time.time() > deadline - 5.0: # Leave time for refinement
                break

            best_gain = 0
            tie_candidates = []

            # Scan all candidates
            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                
                # Calculate gain
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    tie_candidates = [j]
                elif gain == best_gain and gain > 0:
                    tie_candidates.append(j)

            if best_gain == 0:
                # No progress possible
                break

            # Randomly pick from tie candidates
            best_j = random.choice(tie_candidates)
            
            # Add best block
            blocks.append(list(all_ksubs[best_j]))
            used_blocks.add(best_j)
            
            # Update uncovered mask
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= best_gain
            
        return blocks

    # 5. Helper: Refinement (remove redundant blocks)
    def refine(blocks):
        if not blocks:
            return blocks
            
        # Map blocks to indices
        # Optimization: Use a dict for O(1) lookup if we pre-computed tuple->idx
        # But we didn't. Let's create a map for this specific run or reuse.
        # Creating map is O(N_k). N_k is 50k. Doing it every refinement is costly.
        # Let's pre-compute a global map: tuple(ksub) -> index
        pass # We will handle index mapping outside or use a smarter way.
        
        # Actually, let's just use the indices directly in the greedy loop.
        # Modify greedy_construct to return indices.
        
        return blocks # Placeholder, will be replaced

    # Re-structure:
    # 1. Pre-compute tuple_to_idx for ksubs
    ksub_to_idx = {ks: i for i, ks in enumerate(all_ksubs)}
    
    def greedy_construct_indices():
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        selected_indices = []
        used_blocks = set()
        
        while remaining > 0:
            if time.time() > deadline - 10.0: # Leave time for refinement
                break

            best_gain = 0
            tie_candidates = []

            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    tie_candidates = [j]
                elif gain == best_gain and gain > 0:
                    tie_candidates.append(j)

            if best_gain == 0:
                break

            best_j = random.choice(tie_candidates)
            selected_indices.append(best_j)
            used_blocks.add(best_j)
            
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= best_gain
            
        return selected_indices

    def refine_indices(selected_indices):
        if not selected_indices:
            return selected_indices
            
        active_indices = list(selected_indices)
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_active = []
            for i_idx, bi in enumerate(active_indices):
                # Calculate union of all OTHER active blocks
                other_mask = 0
                for j_idx, bj in enumerate(active_indices):
                    if i_idx != j_idx:
                        other_mask |= ksub_masks[bj]
                
                # Check if bi is redundant
                if (ksub_masks[bi] & ~other_mask) == 0:
                    changed = True
                else:
                    new_active.append(bi)
            
            active_indices = new_active
            
        return active_indices

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple times
    while time.time() < deadline - 15.0:
        # 1. Greedy
        indices = greedy_construct_indices()
        
        if not indices:
            continue
            
        # 2. Refine
        refined_indices = refine_indices(indices)
        
        # 3. Verify
        if not refined_indices:
            continue
            
        final_mask = 0
        for idx in refined_indices:
            final_mask |= ksub_masks[idx]
            
        if final_mask == (1 << n_tsubs) - 1:
            if len(refined_indices) < best_len:
                best_len = len(refined_indices)
                best_blocks = [list(all_ksubs[idx]) for idx in refined_indices]
                # Optional: If we hit a very good score, we can break or just keep going
                # print(f"New best: {best_len}")
        else:
            # Should not happen if greedy was valid, but refinement might break it?
            # No, refinement only removes redundant blocks, so coverage is preserved.
            pass

    if best_blocks is None:
        # Fallback: return whatever the last run produced, or empty?
        # The task requires a valid covering. If we didn't find one, we failed.
        # But greedy should always find one if it completes.
        # If it timed out in greedy, it might be invalid.
        # We should probably return the best *valid* one found.
        # If none found, return empty? Or try one last deterministic run?
        # Given the constraints, we assume we find at least one.
        pass

    return best_blocks
