import itertools
import time
import random
from collections import defaultdict
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    # Also store list of t-subset indices for each block (for faster count updates)
    block_t_indices = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
    
    # Precompute for each t-subset, which blocks cover it (reverse index)
    # t_to_blocks[j] = list of block indices that cover t_subset j
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_blocks):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)
    
    # Precompute numpy array of block masks for vectorized operations
    # Since n_t can be large, we use uint64 arrays for each 64-bit chunk
    n_chunks = (n_t + 63) // 64
    block_masks_np = np.zeros((n_blocks, n_chunks), dtype=np.uint64)
    for i in range(n_blocks):
        mask = block_masks[i]
        for c in range(n_chunks):
            block_masks_np[i, c] = (mask >> (64 * c)) & 0xFFFFFFFFFFFFFFFF
    
    def greedy_cover(weights=None):
        """Greedy set cover with optional Lagrangian weights."""
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        # Precompute weight array for fast access
        weights_np = np.array(weights, dtype=np.float64) if weights is not None else None
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            best_block_idx = -1
            best_score = 0
            
            if weights_np is None:
                # Fast path: use bit_count on all blocks
                # Convert uncovered_mask to numpy uint64 chunks
                unc_chunks = np.array([(uncovered_mask >> (64 * c)) & 0xFFFFFFFFFFFFFFFF for c in range(n_chunks)], dtype=np.uint64)
                # For each block, compute intersection count
                # We need to count bits in (uncovered_mask & block_masks[i])
                # Use a loop but with early termination if we can't beat best
                for i in range(n_blocks):
                    bm = block_masks[i]
                    inter = uncovered_mask & bm
                    if inter == 0:
                        continue
                    gain = inter.bit_count()
                    if gain > best_score:
                        best_score = gain
                        best_block_idx = i
            else:
                # Weighted path
                for i in range(n_blocks):
                    bm = block_masks[i]
                    inter = uncovered_mask & bm
                    if inter == 0:
                        continue
                    # Sum weights of bits set in inter
                    w_sum = 0.0
                    m = inter
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        w_sum += weights_np[idx]
                        m ^= lsb
                    gain = w_sum
                    if gain > best_score:
                        best_score = gain
                        best_block_idx = i
                    
            if best_block_idx == -1:
                break
                
            selected.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return selected
    
    def primal_dual_cover():
        """Primal-dual approximation for set cover.
        
        Iteratively raises dual prices for uncovered t-subsets and selects
        blocks with the highest dual-weight-to-cost ratio.
        """
        selected = []
        selected_set = set()
        uncovered_mask = (1 << n_t) - 1
        dual_prices = [0.0] * n_t
        
        while uncovered_mask and time.time() < deadline:
            # Find the block with maximum sum of dual prices over uncovered elements it covers
            best_block = -1
            best_ratio = 0.0
            
            for i in range(n_blocks):
                bm = block_masks[i]
                inter = uncovered_mask & bm
                if inter == 0:
                    continue
                
                # Sum of dual prices for uncovered elements covered by this block
                w_sum = 0.0
                m = inter
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    w_sum += dual_prices[idx]
                    m ^= lsb
                
                if w_sum > best_ratio:
                    best_ratio = w_sum
                    best_block = i
            
            if best_block == -1:
                break
            
            # Raise dual prices for uncovered elements covered by best_block
            # until at least one element's dual price reaches its "cost" (1.0)
            bm = block_masks[best_block]
            inter = uncovered_mask & bm
            
            # Find the minimum amount to raise all dual prices in inter
            # such that at least one reaches 1.0
            min_headroom = 1.0
            m = inter
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                headroom = 1.0 - dual_prices[idx]
                if headroom < min_headroom:
                    min_headroom = headroom
                m ^= lsb
            
            # Raise dual prices
            m = inter
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                dual_prices[idx] += min_headroom
                m ^= lsb
            
            selected.append(best_block)
            selected_set.add(best_block)
            uncovered_mask &= ~bm
        
        return selected
    
    def local_search(selected):
        """Aggressive local search: removal + swap-based improvement."""
        if not selected:
            return selected
        
        # Initialize counts
        counts = [0] * n_t
        for b_idx in selected:
            for idx in block_t_indices[b_idx]:
                counts[idx] += 1
        
        selected_set = set(selected)
        
        # Phase 1: Iterative removal of redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected:
                mask = block_masks[b_idx]
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Phase 2: Swap-based local search (1-swap) with weighted scoring
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            # Recompute counts for current selected
            counts = [0] * n_t
            for b_idx in new_selected:
                for idx in block_t_indices[b_idx]:
                    counts[idx] += 1
            
            # For each block in selected, check if we can swap it out
            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline:
                    break
                # Find t-subsets uniquely covered by b_idx
                unique_indices = []
                for idx in block_t_indices[b_idx]:
                    if counts[idx] == 1:
                        unique_indices.append(idx)
                
                if not unique_indices:
                    # This block is redundant, remove it
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break
                
                # Try to find a block B (not in new_set) that covers all unique_indices
                best_swap = -1
                best_gain = 0
                for candidate in t_to_blocks[unique_indices[0]]:
                    if candidate in new_set:
                        continue
                    # Check if candidate covers all unique_indices
                    cand_mask = block_masks[candidate]
                    covers_all = True
                    for ui in unique_indices:
                        if not (cand_mask >> ui) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Compute additional gain with soft-min weighting
                    extra = 0
                    orig_mask = block_masks[b_idx]
                    diff = cand_mask & ~orig_mask
                    for idx in block_t_indices[candidate]:
                        if (cand_mask >> idx) & 1 and not (orig_mask >> idx) & 1:
                            if counts[idx] == 0:
                                extra += 10
                            elif counts[idx] == 1:
                                extra += 1
                    
                    if extra > best_gain:
                        best_gain = extra
                        best_swap = candidate
                
                if best_swap != -1 and best_gain > 0:
                    # Perform the swap
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    for idx in block_t_indices[best_swap]:
                        counts[idx] += 1
                    new_selected.remove(b_idx)
                    new_selected.append(best_swap)
                    new_set.discard(b_idx)
                    new_set.add(best_swap)
                    improved = True
                    break
            
            selected = new_selected
        
        # Phase 3: 2-swap local search (replace 2 blocks with 2 different blocks)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            # Recompute counts
            counts = [0] * n_t
            for b_idx in new_selected:
                for idx in block_t_indices[b_idx]:
                    counts[idx] += 1
            
            n_sel = len(new_selected)
            if n_sel < 2:
                break
            
            # Try all pairs of blocks to replace
            for i in range(n_sel):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_sel):
                    b1 = new_selected[i]
                    b2 = new_selected[j]
                    
                    # Find t-subsets uniquely covered by b1 or b2
                    unique_to_pair = []
                    # t-subsets covered by b1 but not b2 and count==1
                    mask1 = block_masks[b1]
                    mask2 = block_masks[b2]
                    for idx in block_t_indices[b1]:
                        if (mask1 >> idx) & 1 and not (mask2 >> idx) & 1 and counts[idx] == 1:
                            unique_to_pair.append(idx)
                    # t-subsets covered by b2 but not b1 and count==1
                    for idx in block_t_indices[b2]:
                        if (mask2 >> idx) & 1 and not (mask1 >> idx) & 1 and counts[idx] == 1:
                            unique_to_pair.append(idx)
                    
                    if not unique_to_pair:
                        # Both blocks are redundant, remove both
                        for idx in block_t_indices[b1]:
                            counts[idx] -= 1
                        for idx in block_t_indices[b2]:
                            counts[idx] -= 1
                        new_selected.remove(b1)
                        new_selected.remove(b2)
                        new_set.discard(b1)
                        new_set.discard(b2)
                        improved = True
                        break
                    
                    # Try to find 2 blocks that cover all unique_to_pair
                    # Use the first unique t-subset to generate candidates
                    candidates_from_first = t_to_blocks[unique_to_pair[0]]
                    
                    best_pair = None
                    best_score = 0
                    
                    for c1 in candidates_from_first:
                        if c1 in new_set or c1 == b1 or c1 == b2:
                            continue
                        c1_mask = block_masks[c1]
                        # Check if c1 covers some of unique_to_pair
                        remaining = [ui for ui in unique_to_pair if not (c1_mask >> ui) & 1]
                        if not remaining:
                            # c1 alone covers all, need a second block that's "good"
                            # Find c2 that maximizes additional coverage
                            best_c2 = -1
                            best_c2_score = 0
                            for c2 in t_to_blocks[unique_to_pair[0]]:
                                if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                    continue
                                c2_mask = block_masks[c2]
                                # Score: additional t-subsets covered
                                diff = (c2_mask & ~mask1 & ~mask2)
                                score = diff.bit_count()
                                if score > best_c2_score:
                                    best_c2_score = score
                                    best_c2 = c2
                            if best_c2 != -1 and best_c2_score > 0:
                                total_score = best_c2_score
                                if total_score > best_score:
                                    best_score = total_score
                                    best_pair = (c1, c2)
                        else:
                            # Need c2 to cover 'remaining'
                            for c2 in t_to_blocks[remaining[0]]:
                                if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                    continue
                                c2_mask = block_masks[c2]
                                # Check if c2 covers all of remaining
                                covers = True
                                for ui in remaining:
                                    if not (c2_mask >> ui) & 1:
                                        covers = False
                                        break
                                if not covers:
                                    continue
                                # Score: additional t-subsets covered by c1 and c2 together
                                combined_new = (c1_mask | c2_mask) & ~mask1 & ~mask2
                                score = combined_new.bit_count()
                                if score > best_score:
                                    best_score = score
                                    best_pair = (c1, c2)
                    
                    if best_pair and best_score > 0:
                        c1, c2 = best_pair
                        # Perform the 2-swap
                        for idx in block_t_indices[b1]:
                            counts[idx] -= 1
                        for idx in block_t_indices[b2]:
                            counts[idx] -= 1
                        for idx in block_t_indices[c1]:
                            counts[idx] += 1
                        for idx in block_t_indices[c2]:
                            counts[idx] += 1
                        new_selected.remove(b1)
                        new_selected.remove(b2)
                        new_selected.append(c1)
                        new_selected.append(c2)
                        new_set.discard(b1)
                        new_set.discard(b2)
                        new_set.add(c1)
                        new_set.add(c2)
                        improved = True
                        break
                
                if improved:
                    break
            
            selected = new_selected
        
        return selected
    
    def shrink_solution(selected):
        """Try to remove blocks by checking if they can be replaced by fewer blocks."""
        if not selected:
            return selected
        
        selected = list(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute counts once per pass
            counts = [0] * n_t
            for b in selected:
                for idx in block_t_indices[b]:
                    counts[idx] += 1
            
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                b_idx = selected[i]
                
                # Find t-subsets that are only covered by this block
                unique_to_this = []
                for idx in block_t_indices[b_idx]:
                    if counts[idx] == 1:
                        unique_to_this.append(idx)
                
                if not unique_to_this:
                    # Remove this block
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    selected.pop(i)
                    improved = True
                    break
        
        return selected
    
    # Multi-restart with different strategies
    best_solution = None
    best_len = float('inf')
    
    # Strategy 1: Simple greedy
    sol1 = greedy_cover()
    sol1 = local_search(sol1)
    if sol1:
        best_solution = sol1
        best_len = len(sol1)
    
    # Strategy 2: Primal-dual approximation
    if time.time() < deadline - 40:
        sol2 = primal_dual_cover()
        sol2 = local_search(sol2)
        if sol2 and len(sol2) < best_len:
            best_solution = sol2
            best_len = len(sol2)
    
    # Strategy 3: Lagrangian-guided greedy with smooth surrogate
    weights = [1.0] * n_t
    for iteration in range(6):
        if time.time() > deadline - 30:
            break
        
        sol = greedy_cover(weights)
        if not sol:
            break
        
        covered_mask = 0
        for b_idx in sol:
            covered_mask |= block_masks[b_idx]
        
        # Use soft-min via log-sum-exp for smoother weight updates
        new_weights = [0.0] * n_t
        m = (1 << n_t) - 1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if not (covered_mask >> idx) & 1:
                new_weights[idx] = weights[idx] * 2.5
            else:
                new_weights[idx] = weights[idx] * 0.7
            m ^= lsb
        weights = new_weights
        
        sol = local_search(sol)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Strategy 4: Randomized greedy (multiple trials) with temperature-based selection
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        
        uncovered_mask = (1 << n_t) - 1
        selected = []
        temperature = 0.9  # Start with higher temperature for exploration
        
        while uncovered_mask and time.time() < deadline - 10:
            gains = []
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                gains.append(gain)
            
            if max(gains) == 0:
                break
            
            # Use softmax-like selection with temperature
            max_gain = max(gains)
            if max_gain == 0:
                break
            
            # Compute probabilities based on gains and temperature
            exp_gains = [math.exp(g / temperature) if g > 0 else 0 for g in gains]
            total_exp = sum(exp_gains)
            if total_exp == 0:
                break
            
            probs = [eg / total_exp for eg in exp_gains]
            
            # Sample a block based on probabilities
            r = random.random()
            cumulative = 0
            chosen = -1
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen = i
                    break
            
            if chosen == -1:
                chosen = n_blocks - 1
            
            selected.append(chosen)
            uncovered_mask &= ~block_masks[chosen]
            
            # Cool down temperature
            temperature *= 0.95
        
        if selected:
            selected = local_search(selected)
            if selected and len(selected) < best_len:
                best_solution = selected
                best_len = len(selected)
    
    # Final: Try to improve best_solution with more aggressive local search and shrink
    if best_solution and time.time() < deadline - 5:
        best_solution = local_search(best_solution)
        best_solution = shrink_solution(best_solution)
        if time.time() < deadline - 2:
            best_solution = local_search(best_solution)
        
        # Additional polish: try 3-swap local search if time permits
        if time.time() < deadline - 1:
            # Implement a lightweight 3-swap
            selected = list(best_solution)
            selected_set = set(selected)
            counts = [0] * n_t
            for b_idx in selected:
                for idx in block_t_indices[b_idx]:
                    counts[idx] += 1
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                n_sel = len(selected)
                if n_sel < 3:
                    break
                
                # Try to find 3 blocks to replace with 3 better blocks
                for i in range(n_sel):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n_sel):
                        if time.time() > deadline:
                            break
                        for l in range(j + 1, n_sel):
                            if time.time() > deadline:
                                break
                            b1, b2, b3 = selected[i], selected[j], selected[l]
                            
                            # Find unique t-subsets for this triple
                            unique_to_triple = []
                            combined_mask = block_masks[b1] | block_masks[b2] | block_masks[b3]
                            all_indices_in_triple = set(block_t_indices[b1]) | set(block_t_indices[b2]) | set(block_t_indices[b3])
                            for idx in all_indices_in_triple:
                                if (combined_mask >> idx) & 1 and counts[idx] == 1:
                                    unique_to_triple.append(idx)
                            
                            if not unique_to_triple:
                                # All three are redundant
                                for idx in block_t_indices[b1]:
                                    counts[idx] -= 1
                                for idx in block_t_indices[b2]:
                                    counts[idx] -= 1
                                for idx in block_t_indices[b3]:
                                    counts[idx] -= 1
                                selected.remove(b1)
                                selected.remove(b2)
                                selected.remove(b3)
                                selected_set.discard(b1)
                                selected_set.discard(b2)
                                selected_set.discard(b3)
                                improved = True
                                break
                            
                            # Try to find 3 blocks that cover all unique_to_triple
                            candidates_from_first = t_to_blocks[unique_to_triple[0]]
                            best_triple = None
                            best_score = 0
                            
                            for c1 in candidates_from_first[:10]:  # Limit for efficiency
                                if c1 in selected_set or c1 in (b1, b2, b3):
                                    continue
                                c1_mask = block_masks[c1]
                                remaining1 = [ui for ui in unique_to_triple if not (c1_mask >> ui) & 1]
                                if not remaining1:
                                    # c1 covers all, find c2, c3 for additional coverage
                                    for c2 in t_to_blocks[unique_to_triple[0]][:5]:
                                        if c2 in selected_set or c2 in (b1, b2, b3) or c2 == c1:
                                            continue
                                        c2_mask = block_masks[c2]
                                        remaining2 = [ui for ui in unique_to_triple if not (c2_mask >> ui) & 1]
                                        if not remaining2:
                                            for c3 in t_to_blocks[unique_to_triple[0]][:3]:
                                                if c3 in selected_set or c3 in (b1, b2, b3) or c3 in (c1, c2):
                                                    continue
                                                c3_mask = block_masks[c3]
                                                combined_new = (c1_mask | c2_mask | c3_mask) & ~combined_mask
                                                score = combined_new.bit_count()
                                                if score > best_score:
                                                    best_score = score
                                                    best_triple = (c1, c2, c3)
                                else:
                                    for c2 in t_to_blocks[remaining1[0]][:5]:
                                        if c2 in selected_set or c2 in (b1, b2, b3) or c2 == c1:
                                            continue
                                        c2_mask = block_masks[c2]
                                        covers = True
                                        for ui in remaining1:
                                            if not (c2_mask >> ui) & 1:
                                                covers = False
                                                break
                                        if not covers:
                                            continue
                                        remaining2 = [ui for ui in unique_to_triple if not ((c1_mask | c2_mask) >> ui) & 1]
                                        if not remaining2:
                                            for c3 in t_to_blocks[unique_to_triple[0]][:3]:
                                                if c3 in selected_set or c3 in (b1, b2, b3) or c3 in (c1, c2):
                                                    continue
                                                c3_mask = block_masks[c3]
                                                combined_new = (c1_mask | c2_mask | c3_mask) & ~combined_mask
                                                score = combined_new.bit_count()
                                                if score > best_score:
                                                    best_score = score
                                                    best_triple = (c1, c2, c3)
                                        else:
                                            for c3 in t_to_blocks[remaining2[0]][:3]:
                                                if c3 in selected_set or c3 in (b1, b2, b3) or c3 in (c1, c2):
                                                    continue
                                                c3_mask = block_masks[c3]
                                                covers = True
                                                for ui in remaining2:
                                                    if not (c3_mask >> ui) & 1:
                                                        covers = False
                                                        break
                                                if not covers:
                                                    continue
                                                combined_new = (c1_mask | c2_mask | c3_mask) & ~combined_mask
                                                score = combined_new.bit_count()
                                                if score > best_score:
                                                    best_score = score
                                                    best_triple = (c1, c2, c3)
                            
                            if best_triple and best_score > 0:
                                c1, c2, c3 = best_triple
                                for idx in block_t_indices[b1]:
                                    counts[idx] -= 1
                                for idx in block_t_indices[b2]:
                                    counts[idx] -= 1
                                for idx in block_t_indices[b3]:
                                    counts[idx] -= 1
                                for idx in block_t_indices[c1]:
                                    counts[idx] += 1
                                for idx in block_t_indices[c2]:
                                    counts[idx] += 1
                                for idx in block_t_indices[c3]:
                                    counts[idx] += 1
                                selected.remove(b1)
                                selected.remove(b2)
                                selected.remove(b3)
                                selected.extend([c1, c2, c3])
                                selected_set.discard(b1)
                                selected_set.discard(b2)
                                selected_set.discard(b3)
                                selected_set.add(c1)
                                selected_set.add(c2)
                                selected_set.add(c3)
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
                
                best_solution = selected
    
    # Convert indices back to blocks
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    else:
        return []
