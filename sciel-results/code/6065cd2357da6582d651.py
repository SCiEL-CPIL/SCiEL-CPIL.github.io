import itertools
import random
import numpy as np


def construct(v, k, t):
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    
    # Map each t-subset to an index
    t_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks and their covered t-subset indices
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # For each block, store the list of t-subset indices it covers
    block_covers = []
    for block in k_blocks:
        covered = set()
        for t_sub in itertools.combinations(block, t):
            covered.add(t_idx[t_sub])
        block_covers.append(frozenset(covered))
    
    # Initial uncovered set: all t-subsets
    uncovered = set(range(n_t))
    
    # Dual variables: weight for each t-subset, initialized to 1
    dual = np.ones(n_t, dtype=np.float64)
    
    blocks = []
    
    # Precompute for each block its list of covered indices (as list for speed)
    block_cover_lists = [list(cs) for cs in block_covers]
    
    # We'll use a max-heap-like approach: at each step, pick the block with max dual weight sum
    # But to keep it efficient, we recompute scores only when needed
    
    # Instead, we'll iteratively:
    # 1. Compute score for each block = sum of dual[i] for i in block's covered & uncovered
    # 2. Pick block with max score
    # 3. Add it, update uncovered, update duals
    
    # For efficiency with v=19, k=8, t=5: C(19,8) = 75582 blocks, C(19,5) = 11628 t-subsets
    # This is manageable.
    
    # Precompute incidence: for each t-subset, which blocks cover it? No, too big.
    # Instead, for each step, compute block scores.
    
    # To speed up, we can use numpy:
    # Create a sparse incidence matrix? Let's just do it in a loop with numpy vectorization where possible.
    
    # Actually, let's precompute for each block a numpy array of its covered indices
    block_covers_np = [np.array(cs, dtype=np.int32) for cs in block_cover_lists]
    
    # Track which blocks are still useful (cover at least one uncovered t-subset)
    # We'll just recompute scores each iteration
    
    while uncovered:
        # Compute score for each block: sum of dual[i] for i in block_covers[b] if i in uncovered
        # Use a set for uncovered for O(1) lookup
        unc_set = uncovered
        
        # For speed, compute scores using numpy where possible
        # But the "if i in unc_set" makes it tricky. Let's just do it with a list comprehension + numpy sum
        scores = []
        for bi in range(n_blocks):
            covers = block_covers_np[bi]
            # Filter to uncovered
            if len(covers) == 0:
                scores.append(0.0)
                continue
            # Convert to list, filter, sum dual
            # This is slow in pure Python. Let's try numpy boolean indexing with a mask
            # Create a boolean array of size n_t, True if uncovered
            # That's expensive per iteration. Instead, maintain a boolean array `unc_mask`
            
            # Let's restructure: maintain unc_mask as a numpy boolean array
            pass
        
        # Better approach: maintain unc_mask
        # Let me redo this more carefully
        
        pass
    
    # Let me rewrite the main loop more carefully
    
    uncovered = set(range(n_t))
    unc_mask = np.ones(n_t, dtype=bool)
    dual = np.ones(n_t, dtype=np.float64)
    blocks = []
    
    # Precompute block_covers as lists of ints
    block_covers_list = block_cover_lists
    
    while uncovered:
        # Compute score for each block
        # score[b] = sum(dual[i] for i in block_covers_list[b] if unc_mask[i])
        # Use numpy: for each block, take dual[indices] and sum where mask is True
        
        scores = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            indices = np.array(block_covers_list[bi], dtype=np.int32)
            if len(indices) == 0:
                continue
            # Get dual values for these indices
            dvals = dual[indices]
            # Mask: only count if uncovered
            mvals = unc_mask[indices]
            scores[bi] = np.sum(dvals[mvals])
        
        # Pick best block
        best_block_idx = int(np.argmax(scores))
        if scores[best_block_idx] <= 0:
            # No block covers any uncovered t-subset? Shouldn't happen if uncovered is non-empty
            # Fallback: pick first uncovered t-subset and make a block containing it
            first_uncovered_t = min(uncovered)
            t_sub = t_subsets[first_uncovered_t]
            # Find a block containing this t-subset
            for bi in range(n_blocks):
                if all(p in k_blocks[bi] for p in t_sub):
                    best_block_idx = bi
                    break
        
        block = list(k_blocks[best_block_idx])
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for i in block_covers_list[best_block_idx]:
            if i in uncovered:
                uncovered.remove(i)
                unc_mask[i] = False
        
        # Update dual: increase dual for newly uncovered... wait, dual increases for uncovered.
        # Actually in primal-dual, we increase dual of uncovered constraints.
        # Since all remaining uncovered have dual = 1, and we want to boost them, 
        # let's increase dual for all still-uncovered t-subsets slightly, or keep them at 1.
        # Simpler: just reset dual to 1 for uncovered, 0 for covered.
        # Or: keep dual as is, but the scoring already only counts uncovered.
        # Let's just set dual[i] = 1.0 for uncovered, 0 for covered, to keep it clean.
        # Actually, let's just recompute with unc_mask only, and ignore dual for simplicity.
        # The "dual-guided" aspect: we can increase dual for t-subsets that are covered by fewer blocks.
        
        # For now, let's just use the greedy with uncovered mask, which is equivalent to dual=1 for all uncovered.
        # To make it "dual-guided", let's weight by how "hard" each t-subset is to cover.
        # Precompute: for each t-subset, how many blocks cover it?
        
        pass
    
    # Let me simplify and just do a clean greedy with the mask approach
    # The dual-guided aspect will be: prioritize t-subsets with fewer covering blocks
    
    # Precompute coverage count for each t-subset
    t_cov_count = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_blocks):
        for i in block_covers_list[bi]:
            t_cov_count[i] += 1
    
    # Weight: t-subsets with fewer covering blocks get higher weight
    # weight[i] = 1 / (t_cov_count[i] + 1) or something
    base_weight = 1.0 / (t_cov_count + 1.0)
    
    # Now redo the loop with this weighting
    uncovered = set(range(n_t))
    unc_mask = np.ones(n_t, dtype=bool)
    blocks = []
    
    # Precompute weighted block scores initially
    # score[b] = sum(base_weight[i] * unc_mask[i] for i in block_covers_list[b])
    
    while uncovered:
        # Compute scores
        scores = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            indices = np.array(block_covers_list[bi], dtype=np.int32)
            if len(indices) == 0:
                continue
            dvals = base_weight[indices]
            mvals = unc_mask[indices]
            scores[bi] = np.sum(dvals[mvals])
        
        best_block_idx = int(np.argmax(scores))
        if scores[best_block_idx] <= 1e-10:
            # Fallback
            first_uncovered_t = min(uncovered)
            t_sub = t_subsets[first_uncovered_t]
            for bi in range(n_blocks):
                if all(p in k_blocks[bi] for p in t_sub):
                    best_block_idx = bi
                    break
        
        block = list(k_blocks[best_block_idx])
        block.sort()
        blocks.append(block)
        
        for i in block_covers_list[best_block_idx]:
            if i in uncovered:
                uncovered.remove(i)
                unc_mask[i] = False
        
        # If uncovered is empty, break
        if not uncovered:
            break
    
    return blocks
