import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

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

    # 4. Initialize uncovered bitset (all 1s for n_tsubs bits)
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs

    selected_indices = []
    used_blocks = set()

    # 5. Greedy selection loop with numpy-vectorized gain computation
    #    Build a numpy array of the low 64 bits of each mask for fast vectorized AND
    #    For gains > 64, we still need the full int, but most early iterations have small gains
    #    Strategy: use numpy for the first pass to narrow candidates, then refine with Python ints
    
    while remaining > 0:
        if time.time() > deadline - 30.0:
            break

        # Fast path: compute gains using Python int bit operations
        # For n_ksubs ~ 50k, this is ~50k big-int ANDs per iteration
        # With ~150 iterations, that's ~7.5M big-int ops total
        # Each big-int is ~300k bits = ~40KB, so ~300GB of memory bandwidth
        # That might be too slow. Let's use a hybrid approach.
        
        # Hybrid: compute gain for all candidates using numpy on the low 64 bits
        # to get a fast upper bound, then only compute exact gain for top candidates
        
        # Extract low 64 bits of each mask into a numpy array
        # Actually, let's just do the Python loop but with a time check
        # and early termination if we find a perfect gain
        
        best_j = -1
        best_gain = 0
        
        # Use numpy to speed up: convert masks to numpy uint64 arrays for low bits
        # But we need exact gains. Let's just do the Python loop with optimizations.
        
        # Optimization: precompute which indices are not used
        unused = [j for j in range(n_ksubs) if j not in used_blocks]
        
        # Find max gain among unused blocks
        # Use a two-pass approach: first get approximate gains via low-64-bit numpy,
        # then exact for top ~200 candidates
        
        # Extract low 64 bits
        low_bits = np.array([m & 0xFFFFFFFFFFFFFFFF for m in ksub_masks], dtype=np.uint64)
        # Uncovered low 64 bits
        unc_low = uncovered_mask & 0xFFFFFFFFFFFFFFFF
        unc_low_arr = np.uint64(unc_low)
        
        # Approximate gain: popcount of (low_bits[j] & unc_low)
        # This underestimates the true gain (misses higher bits), but gives a lower bound
        # Actually it gives the exact count of covered low-64 tsubs, which is a subset of total gain
        approx_gains = np.array([bin(int(x & unc_low_arr)).count('1') for x in low_bits], dtype=np.int32)
        
        # Mask out used blocks
        approx_gains[list(used_blocks)] = 0
        
        # Get top candidates by approximate gain
        # Sort indices by approx gain descending
        sorted_idx = np.argsort(-approx_gains)
        
        # Take top 500 candidates for exact evaluation
        top_candidates = sorted_idx[:500]
        
        for j in top_candidates:
            j = int(j)
            if j in used_blocks:
                continue
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_j = j
                if best_gain == n_tsubs:  # Perfect, can't do better
                    break
        
        # If top 500 didn't find anything good, check remaining (shouldn't happen often)
        if best_gain == 0:
            for j in unused:
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    if best_gain > 0:
                        break
        
        if best_j == -1 or best_gain == 0:
            break

        selected_indices.append(best_j)
        used_blocks.add(best_j)
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # 6. Post-processing: Remove redundant blocks (O(n^2) with prefix/suffix OR)
    if time.time() < deadline - 10.0 and len(selected_indices) > 0:
        active_indices = list(selected_indices)
        n_active = len(active_indices)
        
        # Precompute prefix ORs and suffix ORs for O(1) "union of all except i"
        prefix_or = [0] * (n_active + 1)
        for i in range(n_active):
            prefix_or[i + 1] = prefix_or[i] | ksub_masks[active_indices[i]]
        
        suffix_or = [0] * (n_active + 1)
        for i in range(n_active - 1, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | ksub_masks[active_indices[i]]
        
        # A block at position i is redundant if:
        # ksub_masks[active_indices[i]] & ~(prefix_or[i] | suffix_or[i+1]) == 0
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_active = []
            n_active = len(active_indices)
            
            # Recompute prefix/suffix
            prefix_or = [0] * (n_active + 1)
            for i in range(n_active):
                prefix_or[i + 1] = prefix_or[i] | ksub_masks[active_indices[i]]
            suffix_or = [0] * (n_active + 1)
            for i in range(n_active - 1, -1, -1):
                suffix_or[i] = suffix_or[i + 1] | ksub_masks[active_indices[i]]
            
            for i in range(n_active):
                other_mask = prefix_or[i] | suffix_or[i + 1]
                if (ksub_masks[active_indices[i]] & ~other_mask) == 0:
                    changed = True
                else:
                    new_active.append(active_indices[i])
            
            active_indices = new_active

        if active_indices:
            final_blocks = [list(all_ksubs[idx]) for idx in active_indices]
            final_mask = 0
            for idx in active_indices:
                final_mask |= ksub_masks[idx]
            if final_mask == (1 << n_tsubs) - 1:
                selected_indices = active_indices

    # 7. Local search: try to replace one block with another to reduce count
    #    Actually, we can't reduce count by replacing 1 with 1.
    #    We can try: for each block, see if removing it and adding one new block
    #    still covers everything. If so, we haven't reduced count.
    #    Better: try to find if two blocks can be replaced by one block.
    #    Or: try to find a block that, when removed, the uncovered tsubs can be
    #    covered by a single new block.
    
    if time.time() < deadline - 5.0 and len(selected_indices) > 1:
        current_count = len(selected_indices)
        
        # For each selected block, try removing it and see if the newly uncovered
        # tsubs can be covered by a single block
        for i in range(current_count):
            if time.time() > deadline - 5.0:
                break
            
            bi = selected_indices[i]
            # What would be uncovered if we remove bi?
            others_mask = 0
            for j in range(current_count):
                if j != i:
                    others_mask |= ksub_masks[selected_indices[j]]
            
            newly_uncovered = ksub_masks[bi] & ~others_mask
            if newly_uncovered == 0:
                # Already redundant (should have been caught)
                continue
            
            # Find a single block that covers all newly_uncovered
            # i.e., find j such that (newly_uncovered & ~ksub_masks[j]) == 0
            found = False
            for j in range(n_ksubs):
                if (newly_uncovered & ~ksub_masks[j]) == 0:
                    # Block j covers all newly uncovered tsubs
                    # Check if j is not already in selected_indices (or is bi itself)
                    if j != bi:
                        # Replace bi with j? No, we're removing bi and adding j.
                        # But we need j to not be already selected (unless j == bi, which we skip)
                        if j not in set(selected_indices) or j == bi:
                            # This would keep the same count, not reduce it
                            pass
                    found = True
                    break
            
            if not found:
                continue
            
            # Try: can we remove bi and NOT add anything? (Already handled by redundancy check)
            # Try: can we remove bi and add a block that also covers some other block's unique parts?
            # This is getting complex. Let's try a different approach:
            # Try to find two blocks that can be replaced by one.
            pass
        
        # Alternative local search: try to find a block not in selected that, when added,
        # allows removing two existing blocks.
        # For each candidate block c not in selected:
        #   If adding c allows removing blocks a and b (i.e., all tsubs of a and b
        #   are covered by c + (selected - {a,b})), then replace {a,b} with {c}.
        
        selected_set = set(selected_indices)
        n_sel = len(selected_indices)
        
        if n_sel >= 2:
            # Precompute for each selected block, what unique tsubs it covers
            # (i.e., tsubs only covered by that block among selected)
            
            # For each pair (a, b) in selected, check if there exists c not in selected
            # such that c covers all unique tsubs of a and b.
            
            # This is O(n_sel^2 * n_ksubs) which is ~150^2 * 50000 = 1.1B, too slow.
            # Let's be more targeted: for each selected block a, find its unique tsubs.
            # Then for each other selected block b, find unique tsubs of b.
            # Check if any single block c covers both unique sets.
            
            # First, compute unique coverage for each selected block
            unique_masks = [0] * n_sel
            for i in range(n_sel):
                others = 0
                for j in range(n_sel):
                    if i != j:
                        others |= ksub_masks[selected_indices[j]]
                unique_masks[i] = ksub_masks[selected_indices[i]] & ~others
            
            # For each pair (i, j), the combined unique set is unique_masks[i] | unique_masks[j]
            # We need a block c (not in selected) such that c covers this combined set.
            # i.e., (combined_unique & ~ksub_masks[c]) == 0
            
            # Precompute: for each tsub index, which selected blocks cover it?
            # Actually, let's just iterate over pairs and for each, scan all ksubs.
            # To speed up, only consider pairs where unique_masks[i] and unique_masks[j] are non-empty.
            
            improved = False
            for i in range(n_sel):
                if time.time() > deadline - 5.0:
                    break
                if unique_masks[i] == 0:
                    continue
                for j in range(i + 1, n_sel):
                    if time.time() > deadline - 5.0:
                        break
                    if unique_masks[j] == 0:
                        continue
                    
                    combined = unique_masks[i] | unique_masks[j]
                    
                    # Find a block c that covers combined
                    for c in range(n_ksubs):
                        if c in selected_set:
                            continue
                        if (combined & ~ksub_masks[c]) == 0:
                            # Block c covers all unique tsubs of i and j
                            # New selection: remove i and j, add c
                            new_selected = [selected_indices[x] for x in range(n_sel) if x != i and x != j]
                            new_selected.append(c)
                            
                            # Verify validity
                            new_mask = 0
                            for idx in new_selected:
                                new_mask |= ksub_masks[idx]
                            if new_mask == (1 << n_tsubs) - 1:
                                selected_indices = new_selected
                                selected_set = set(selected_indices)
                                n_sel = len(selected_indices)
                                improved = True
                                break
                    
                    if improved:
                        break
                if improved:
                    break

    # Build final blocks
    blocks = [list(all_ksubs[idx]) for idx in selected_indices]
    
    # Final validity check
    if blocks:
        final_mask = 0
        for idx in selected_indices:
            final_mask |= ksub_masks[idx]
        if final_mask != (1 << n_tsubs) - 1:
            # Fallback: rebuild with greedy
            selected_indices = []
            used_blocks = set()
            uncovered_mask = (1 << n_tsubs) - 1
            remaining = n_tsubs
            while remaining > 0:
                best_j = -1
                best_gain = 0
                for j in range(n_ksubs):
                    if j in used_blocks:
                        continue
                    gain = (uncovered_mask & ksub_masks[j]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j == -1 or best_gain == 0:
                    break
                selected_indices.append(best_j)
                used_blocks.add(best_j)
                uncovered_mask &= ~ksub_masks[best_j]
                remaining -= best_gain
            blocks = [list(all_ksubs[idx]) for idx in selected_indices]
    
    return blocks
