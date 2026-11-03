"""
Covering design C(24,7,3) - Algebraic/Symmetric Construction Strategy.

Strategy:
1. Partition 24 points into 4 groups of 6 (G0, G1, G2, G3).
2. Construct a base set of blocks using a structured design that ensures
   high coverage density and symmetry. We utilize a "cyclic" or "affine"
   inspired generation where we pick blocks that maximize coverage of
   "rare" configurations (triples spanning distinct groups).
3. The core structural change is moving from "random weighted greedy" to
   "structured BIBD-like base + deterministic residual filling".
   We generate candidate blocks based on group interactions:
   - Type A: 5 from one group, 2 from another (covers internal + cross)
   - Type B: 4 from one, 3 from another
   - Type C: 3, 3, 1, 0 or 3, 2, 2, 0 (highly cross-group)
4. Use a fast bitset-based greedy to fill remaining holes.
5. Aggressive pruning: remove any block that doesn't cover a unique triple.
"""

import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()

    # Precompute all t-subsets (triples)
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Full mask for all triples
    full_mask = (1 << n_t) - 1

    # Precompute block cover masks for all possible blocks? 
    # C(24,7) = 346,104. Storing 346k big ints is feasible in memory but slow to build.
    # Instead, we will generate blocks on the fly or use a smaller candidate pool.
    
    # However, for speed, we precompute masks for a *structured* candidate pool first.
    
    # Define groups
    groups = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]
    ]
    
    # Helper to get mask for a block
    def get_block_mask(block):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        return mask

    # Strategy: Generate a diverse set of candidate blocks based on group structure.
    # We want blocks that cover many "hard" triples.
    # Triples within a group are "easy" (covered by many blocks).
    # Triples spanning 3 different groups are "harder".
    
    candidate_blocks = []
    seen_blocks = set()
    
    # Type 1: 5 from G_i, 2 from G_j (i != j)
    # This covers many internal triples in G_i and cross triples.
    for i in range(4):
        for j in range(4):
            if i == j: continue
            gi = groups[i]
            gj = groups[j]
            # Choose 5 from gi, 2 from gj
            # To keep pool manageable, we don't add ALL combinations, but a representative set.
            # Actually, for v=24, C(6,5)*C(6,2) = 6*15 = 90 blocks per pair (i,j).
            # 4*3*90 = 1080 blocks. This is small. We can add them all.
            for comb1 in itertools.combinations(gi, 5):
                for comb2 in itertools.combinations(gj, 2):
                    block = tuple(sorted(comb1 + comb2))
                    if block not in seen_blocks:
                        seen_blocks.add(block)
                        candidate_blocks.append(block)
                        
    # Type 2: 4 from G_i, 3 from G_j (i != j)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            gi = groups[i]
            gj = groups[j]
            for comb1 in itertools.combinations(gi, 4):
                for comb2 in itertools.combinations(gj, 3):
                    block = tuple(sorted(comb1 + comb2))
                    if block not in seen_blocks:
                        seen_blocks.add(block)
                        candidate_blocks.append(block)

    # Type 3: 3 from G_i, 3 from G_j, 1 from G_k (i,j,k distinct)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            for k in range(4):
                if k == i or k == j: continue
                gi = groups[i]
                gj = groups[j]
                gk = groups[k]
                for comb1 in itertools.combinations(gi, 3):
                    for comb2 in itertools.combinations(gj, 3):
                        for comb3 in itertools.combinations(gk, 1):
                            block = tuple(sorted(comb1 + comb2 + comb3))
                            if block not in seen_blocks:
                                seen_blocks.add(block)
                                candidate_blocks.append(block)
                                
    # Type 4: 3 from G_i, 2 from G_j, 2 from G_k (i,j,k distinct)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            for k in range(4):
                if k == i or k == j: continue
                gi = groups[i]
                gj = groups[j]
                gk = groups[k]
                for comb1 in itertools.combinations(gi, 3):
                    for comb2 in itertools.combinations(gj, 2):
                        for comb3 in itertools.combinations(gk, 2):
                            block = tuple(sorted(comb1 + comb2 + comb3))
                            if block not in seen_blocks:
                                seen_blocks.add(block)
                                candidate_blocks.append(block)
                                
    # Type 5: 3 from G_i, 3 from G_j, 1 from G_k, 0 from G_l (already covered in Type 3? No, Type 3 was 3,3,1. This is same.)
    # Let's add "balanced" 2,2,2,1
    for i in range(4):
        for j in range(4):
            if i == j: continue
            for k in range(4):
                if k == i or k == j: continue
                # l is the remaining group
                l = 6 - i - j - k
                gi = groups[i]
                gj = groups[j]
                gk = groups[k]
                gl = groups[l]
                for comb1 in itertools.combinations(gi, 2):
                    for comb2 in itertools.combinations(gj, 2):
                        for comb3 in itertools.combinations(gk, 2):
                            for comb4 in itertools.combinations(gl, 1):
                                block = tuple(sorted(comb1 + comb2 + comb3 + comb4))
                                if block not in seen_blocks:
                                    seen_blocks.add(block)
                                    candidate_blocks.append(block)

    # Type 6: Pure group blocks (7 from one group? No, group size is 6. So max 6 from one group.)
    # So we need at least 2 groups.
    # What about 6 from G_i, 1 from G_j?
    for i in range(4):
        for j in range(4):
            if i == j: continue
            gi = groups[i]
            gj = groups[j]
            # 6 from gi is the whole group
            for comb2 in itertools.combinations(gj, 1):
                block = tuple(sorted(gi + comb2))
                if block not in seen_blocks:
                    seen_blocks.add(block)
                    candidate_blocks.append(block)
                    
    # We now have a structured pool of candidates.
    # Let's also add some random blocks to ensure we don't miss a better structure.
    rng = random.Random(42)
    for _ in range(5000):
        block = tuple(sorted(rng.sample(range(v), k)))
        if block not in seen_blocks:
            seen_blocks.add(block)
            candidate_blocks.append(block)

    # Precompute masks for these candidates
    cand_masks = [get_block_mask(b) for b in candidate_blocks]
    n_cands = len(candidate_blocks)
    
    # Map block tuple to index in candidate list for quick lookup
    cand_index = {b: i for i, b in enumerate(candidate_blocks)}

    # Greedy Set Cover on the candidate pool
    # We use a weighted greedy: prioritize covering "rare" triples first.
    # First, estimate frequency of each triple in the candidate pool.
    t_freq = np.zeros(n_t, dtype=np.int32)
    for mask in cand_masks:
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            t_freq[idx] += 1
            m ^= lb
            
    # Weight: 1 / (freq + 1) to avoid div by zero, higher weight for rarer triples
    t_weights = 1.0 / (t_freq + 1.0)
    
    def greedy_cover_cands(seed=0, time_budget=30.0):
        rng = random.Random(seed)
        uncovered = full_mask
        selected_indices = []
        end_time = time.time() + time_budget
        
        # Precompute marginal gains? No, dynamic.
        # To speed up, we can use a "lazy" evaluation or just iterate.
        # With ~10k-50k candidates, iterating all each step is O(N^2 * bitops).
        # N=50k, steps ~80. 50k*80 = 4M iterations. Bitops on 2024-bit ints are fast.
        
        while uncovered and time.time() < end_time:
            best_idx = -1
            best_gain = -1.0
            
            # Optimization: If we have many candidates, we can sample or use a heap.
            # For now, linear scan is robust enough for 150s budget if optimized.
            # Let's try to speed up by checking only candidates that cover at least one uncovered triple.
            
            # Instead of scanning all, we can pick a random uncovered triple and look at candidates covering it.
            # This is the "randomized greedy" or "Las Vegas" approach.
            
            # Pick a random uncovered triple
            m = uncovered
            # Find a set bit
            if not m: break
            # Random shift to pick a random bit? No, just pick lowest for determinism or random.
            # Let's pick a random set bit.
            # Generate random index, check if set? No.
            # Just pick the lowest set bit for simplicity in this step, or random.
            # Random is better for diversity.
            
            # To pick a random set bit:
            # count bits, pick random rank, find k-th set bit.
            # bit_count is fast in Python 3.8+? Yes, int.bit_count()
            cnt = m.bit_count()
            if cnt == 0: break
            r = rng.randint(0, cnt-1)
            # Find (r+1)-th set bit
            tmp = m
            found = 0
            while found < r:
                lb = tmp & (-tmp)
                tmp ^= lb
                found += 1
            # Now 'lb' is the target bit? No, we stopped after 'found' removals.
            # The next lb is the (r+1)th.
            lb = tmp & (-tmp)
            tri_idx = lb.bit_length() - 1
            
            # Get all candidates covering tri_idx
            # We don't have a reverse index for candidates yet.
            # Building reverse index for 50k candidates * 35 triples each = 1.7M entries. Feasible.
            
            # Let's build reverse index once outside.
            
            # For now, fallback to linear scan if reverse index not built or too slow.
            # Actually, building reverse index is O(N * 35). Very fast.
            
            # We'll assume reverse index is passed or built.
            
            # Since we are inside a function, let's build reverse index in main scope and pass it.
            
            # RESTRUCTURE: Build reverse index in main.
            
            # For this iteration, just do linear scan to keep code simple, 
            # but limit to a subset if it's too slow?
            # Actually, 50k candidates * 80 steps = 4M checks.
            # Each check: mask & uncovered, then count bits (or weighted sum).
            # Weighted sum requires iterating set bits. That's slow.
            
            # Alternative: Just use "maximize number of new triples" (unweighted).
            # This is faster (bit_count) and often good enough.
            # Then refine with weights in a second pass?
            
            # Let's stick to unweighted greedy for speed, then prune.
            
            for idx in range(n_cands):
                new_cov = cand_masks[idx] & uncovered
                if new_cov:
                    gain = new_cov.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if gain == 35: # Max possible
                            break
                            
            if best_idx < 0:
                break
                
            selected_indices.append(best_idx)
            uncovered &= ~cand_masks[best_idx]
            
        return selected_indices

    # Build reverse index for candidates to allow faster "targeted" greedy
    # reverse_idx[tri_idx] = list of candidate indices covering tri_idx
    reverse_idx = [[] for _ in range(n_t)]
    for i, mask in enumerate(cand_masks):
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            reverse_idx[idx].append(i)
            m ^= lb

    def greedy_cover_targeted(seed=0, time_budget=30.0):
        rng = random.Random(seed)
        uncovered = full_mask
        selected_indices = []
        end_time = time.time() + time_budget
        
        while uncovered and time.time() < end_time:
            # Pick a random uncovered triple
            m = uncovered
            cnt = m.bit_count()
            if cnt == 0: break
            r = rng.randint(0, cnt-1)
            tmp = m
            found = 0
            while found < r:
                lb = tmp & (-tmp)
                tmp ^= lb
                found += 1
            lb = tmp & (-tmp)
            tri_idx = lb.bit_length() - 1
            
            # Candidates covering this triple
            cands = reverse_idx[tri_idx]
            if not cands:
                break
                
            best_idx = -1
            best_gain = -1
            
            # Evaluate these candidates
            for idx in cands:
                new_cov = cand_masks[idx] & uncovered
                if new_cov:
                    gain = new_cov.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if gain == 35:
                            break
                            
            if best_idx < 0:
                break
                
            selected_indices.append(best_idx)
            uncovered &= ~cand_masks[best_idx]
            
        return selected_indices

    def prune_redundant(blocks, time_budget=20.0):
        if not blocks:
            return blocks
        
        # Map block to its mask
        block_masks = []
        for b in blocks:
            if b in cand_index:
                block_masks.append(cand_masks[cand_index[b]])
            else:
                block_masks.append(get_block_mask(b))
                
        selected_set = set(range(len(blocks)))
        end_time = time.time() + time_budget
        changed = True
        
        # Reverse index for selected blocks
        # tri_idx -> set of block indices in selected_set
        tri_to_blocks = [[] for _ in range(n_t)]
        for bi, mask in enumerate(block_masks):
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                tri_to_blocks[idx].append(bi)
                m ^= lb
                
        while changed and time.time() < end_time:
            changed = False
            for bi in list(selected_set):
                if time.time() > end_time:
                    break
                # Check if bi is removable
                mask = block_masks[bi]
                m = mask
                removable = True
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    m ^= lb
                    # Check if any other selected block covers idx
                    covered = False
                    for other_bi in tri_to_blocks[idx]:
                        if other_bi != bi and other_bi in selected_set:
                            covered = True
                            break
                    if not covered:
                        removable = False
                        break
                if removable:
                    selected_set.discard(bi)
                    changed = True
                    
        return [blocks[bi] for bi in sorted(selected_set)]

    # Main Execution
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple targeted greedy runs
    for seed in range(30):
        if time.time() > deadline - 40.0:
            break
        sel_indices = greedy_cover_targeted(seed=seed, time_budget=5.0)
        if not sel_indices:
            continue
        blocks = [candidate_blocks[i] for i in sel_indices]
        
        # Prune
        blocks = prune_redundant(blocks, time_budget=5.0)
        
        # Validate
        cov = 0
        for b in blocks:
            if b in cand_index:
                cov |= cand_masks[cand_index[b]]
            else:
                cov |= get_block_mask(b)
                
        if cov == full_mask and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Try unweighted greedy on full pool if targeted didn't work well
    if best_blocks is None or len(best_blocks) > 80:
        for seed in range(10):
            if time.time() > deadline - 20.0:
                break
            sel_indices = greedy_cover_cands(seed=seed, time_budget=10.0)
            if not sel_indices:
                continue
            blocks = [candidate_blocks[i] for i in sel_indices]
            blocks = prune_redundant(blocks, time_budget=5.0)
            cov = 0
            for b in blocks:
                if b in cand_index:
                    cov |= cand_masks[cand_index[b]]
                else:
                    cov |= get_block_mask(b)
            if cov == full_mask and len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

    # Fallback: If we still don't have a valid cover, use a brute force greedy on ALL blocks
    # This is slow but ensures validity.
    if best_blocks is None or cov != full_mask:
        # Brute force greedy on all C(24,7) blocks
        all_blocks = list(itertools.combinations(range(v), k))
        # Precompute masks for all? Too slow/memory.
        # On the fly.
        uncovered = full_mask
        final_blocks = []
        end_time = deadline - 5.0
        for b in all_blocks:
            if time.time() > end_time:
                break
            if not uncovered:
                break
            mask = get_block_mask(b)
            new_cov = mask & uncovered
            if new_cov:
                # Heuristic: only add if it covers at least 1 new triple
                final_blocks.append(b)
                uncovered &= ~mask
        # Check if valid
        cov = 0
        for b in final_blocks:
            cov |= get_block_mask(b)
        if cov == full_mask:
            best_blocks = final_blocks
            best_len = len(final_blocks)

    # Final Prune
    if best_blocks:
        best_blocks = prune_redundant(best_blocks, time_budget=10.0)

    # Final Validation
    if best_blocks:
        cov = 0
        for b in best_blocks:
            if b in cand_index:
                cov |= cand_masks[cand_index[b]]
            else:
                cov |= get_block_mask(b)
        if cov == full_mask:
            return best_blocks
        else:
            # Emergency: return what we have? No, must be valid.
            # If invalid, return empty or try one more time.
            pass

    # If all else fails, return a simple valid (but large) cover
    # Construct a trivial cover: for each triple, add a block containing it.
    # This will be huge but valid.
    # Better: Greedy on all blocks with no time limit (might timeout, but we have margin)
    # Actually, the brute force above should have worked.
    
    # If we are here, something went wrong. Return empty list (score 0) or best effort.
    # Let's try a very simple greedy one last time.
    uncovered = full_mask
    final_blocks = []
    for b in itertools.combinations(range(v), k):
        if not uncovered:
            break
        mask = get_block_mask(b)
        new_cov = mask & uncovered
        if new_cov:
            final_blocks.append(b)
            uncovered &= ~mask
            
    return final_blocks

