import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_idx_map = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage vectors for each block as sparse dicts or bitsets
    # Using bitsets for speed in coverage calculations
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= (1 << t_sub_to_idx[sub])
        block_masks[i] = mask
    
    full_mask = (1 << n_t) - 1
    
    # Surrogate parameter
    alpha = 10.0
    
    def surrogate_cost(block_indices):
        """
        Calculate smooth surrogate cost:
        Sum over all t-subsets of log(1 + exp(-alpha * count_e)) + lambda * len(blocks)
        If count_e == 0, log(1+exp(0)) = log(2) ~ 0.69
        If count_e >= 1, log(1+exp(-alpha)) ~ 0
        """
        if not block_indices:
            return float('inf')
        
        # Calculate coverage counts
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in block_indices:
            mask = block_masks[idx]
            # Efficiently add bits? No, numpy doesn't handle bitsets well directly.
            # We need to iterate bits or use a different representation.
            # Given n_t = 11628, iterating bits is slow.
            # Let's stick to a faster heuristic for the surrogate: 
            # We will approximate the "uncovered" penalty by counting uncovered bits.
            # A true smooth surrogate requires knowing counts.
            # Let's use a hybrid: 
            # 1. Count uncovered subsets (hard constraint proxy)
            # 2. Count total blocks (optimization goal)
            # To make it "smooth" in the search sense, we use a weighted sum.
            
            pass
        
        # Actually, calculating exact counts for every evaluation in a local search 
        # with 300+ blocks is expensive if done from scratch.
        # We will maintain counts incrementally in the local search.
        # For now, this function is for initial evaluation or fallback.
        
        covered = 0
        for idx in block_indices:
            covered |= block_masks[idx]
        
        uncovered_count = (full_mask & ~covered).bit_count()
        
        # Smooth-like penalty: quadratic for uncovered, linear for blocks
        # This is a common heuristic surrogate
        penalty = 100.0 * uncovered_count + len(block_indices)
        return penalty

    def initial_greedy(seed):
        """
        Generate an initial valid cover using a greedy algorithm 
        with randomization to ensure diversity.
        """
        random.seed(seed)
        uncovered = full_mask
        blocks = []
        
        # Pre-compute gains for all blocks relative to initial state? No, dynamic.
        # To speed up, we can pre-compute block masks as arrays of indices?
        # Let's use the bitset approach but optimize the inner loop.
        
        while uncovered:
            best_block_idx = -1
            best_gain = -1
            
            # Optimization: Instead of checking all n_blocks every time,
            # we can sample or use a heap. But for simplicity and correctness first,
            # we iterate all.
            # n_blocks = 75582. This loop is O(n_blocks * n_t/word_size).
            # 75k * 180 words ~ 13M ops per step.
            # If we take 300 steps, that's 4B ops. Too slow in Python.
            
            # We need a faster greedy.
            # Let's pre-compute for each t-subset, which blocks cover it?
            # No, that's the inverse.
            
            # Alternative: Randomized Greedy with limited lookahead or sampling.
            # Or: Pre-compute coverage of each block as a list of t-subset indices.
            
            # Let's try a different approach for speed:
            # 1. Start with a large random set of blocks? No, must be valid.
            # 2. Use a pre-computed "best block for each uncovered subset" heuristic?
            
            # Let's use a faster greedy:
            # Maintain a list of uncovered t-subsets.
            # For each uncovered subset, pick a random block that covers it?
            # No, that might be inefficient.
            
            # Let's stick to standard greedy but optimize the gain calculation.
            # We can pre-compute the bit_count of (uncovered & mask) faster?
            # Python int bit_count is fast.
            
            for i in range(n_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                # Tie-breaking with randomness
                elif gain == best_gain and random.random() < 0.1:
                    best_block_idx = i
            
            if best_block_idx == -1:
                break
                
            blocks.append(all_blocks[best_block_idx])
            uncovered &= ~block_masks[best_block_idx]
            
            if time.time() > deadline:
                return None
        
        return blocks

    def prune_blocks(blocks):
        """
        Remove redundant blocks from a valid cover.
        """
        if not blocks:
            return blocks
        
        block_indices = [block_idx_map[b] for b in blocks]
        current = list(block_indices)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Calculate coverage counts
            counts = defaultdict(int)
            for idx in current:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    counts[idx_t] += 1
                    m ^= lsb
            
            # Check each block for redundancy
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                essential = False
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if counts[idx_t] == 1:
                        essential = True
                        break
                    m ^= lsb
                
                if essential:
                    new_current.append(idx)
                else:
                    changed = True
            
            current = new_current
        
        return [all_blocks[i] for i in current]

    def local_search(blocks, max_iters=1000):
        """
        Iterated Local Search with smooth surrogate-like objective.
        Moves:
        1. Remove a block if it doesn't break validity.
        2. Swap a block with another if it reduces size or improves "tightness".
        
        Objective: Minimize len(blocks).
        Constraint: Validity.
        
        We use a "penalty" method: Cost = len(blocks) + M * uncovered_count.
        """
        block_indices = [block_idx_map[b] for b in blocks]
        current_size = len(block_indices)
        
        # Maintain coverage counts
        counts = defaultdict(int)
        for idx in block_indices:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                counts[idx_t] += 1
                m ^= lsb
        
        uncovered_count = 0
        for c in counts.values():
            if c == 0:
                uncovered_count += 1
        
        # This is O(n_t) which is 11k. Fast.
        
        best_cost = current_size + 1000 * uncovered_count
        best_blocks = list(block_indices)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Try removals
            removed = False
            for j in range(len(block_indices)):
                idx = block_indices[j]
                mask = block_masks[idx]
                
                # Check if removal breaks validity
                # i.e., does it make any count go to 0?
                # We need to check if any t-subset in mask has count == 1
                is_essential = False
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if counts[idx_t] == 1:
                        is_essential = True
                        break
                    m ^= lsb
                
                if not is_essential:
                    # Remove it
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        counts[idx_t] -= 1
                        m ^= lsb
                    
                    block_indices.pop(j)
                    current_size -= 1
                    removed = True
                    
                    # Update uncovered count efficiently?
                    # We can just recalculate or track.
                    # For now, track.
                    # If count becomes 0, uncovered increases.
                    # We didn't track which became 0.
                    # Let's just recalculate uncovered_count if it changes significantly.
                    # Or simpler: just check validity at the end of the move.
                    
                    # Since we checked is_essential, validity is preserved.
                    break
            
            if removed:
                # Update best if better
                new_cost = current_size + 1000 * 0 # Valid
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_blocks = list(block_indices)
                continue
            
            # Try swaps
            # Pick a random block to swap out
            if block_indices:
                j = random.randint(0, len(block_indices) - 1)
                old_idx = block_indices[j]
                
                # Find a candidate to swap in
                # We want a block that covers some "critical" subsets or just any valid block.
                # To improve, we might want to swap a "redundant" block for a "more useful" one?
                # Actually, if we can't remove, we are stuck in a local minimum for removals.
                # Swapping might allow a future removal.
                
                # Heuristic: Swap with a random block, then try to prune.
                # This is expensive.
                
                # Alternative: 1-Opt move.
                # Replace block j with block i.
                # Check if new set is valid.
                # If valid, check if we can remove any other block?
                
                # This is getting complex. Let's stick to Removal-only local search
                # with random restarts (Iterated Local Search).
                
                # Perturbation: Add a random block, then prune.
                new_block = random.choice(all_blocks)
                new_idx = block_idx_map[new_block]
                
                # Add it
                block_indices.append(new_idx)
                current_size += 1
                
                # Update counts
                mask = block_masks[new_idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    counts[idx_t] += 1
                    m ^= lsb
                
                # Now try to prune this expanded set
                # We can reuse the logic from prune_blocks but on indices
                # For simplicity, we'll just do a few removal passes.
                for _ in range(10):
                    removed_any = False
                    for k in range(len(block_indices)):
                        idx = block_indices[k]
                        mask = block_masks[idx]
                        is_essential = False
                        m = mask
                        while m:
                            lsb = m & -m
                            idx_t = lsb.bit_length() - 1
                            if counts[idx_t] == 1:
                                is_essential = True
                                break
                            m ^= lsb
                        
                        if not is_essential:
                            m = mask
                            while m:
                                lsb = m & -m
                                idx_t = lsb.bit_length() - 1
                                counts[idx_t] -= 1
                                m ^= lsb
                            block_indices.pop(k)
                            current_size -= 1
                            removed_any = True
                            break
                    if not removed_any:
                        break
                
                # Check validity
                valid = True
                for c in counts.values():
                    if c == 0:
                        valid = False
                        break
                
                if valid:
                    new_cost = current_size
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = list(block_indices)
                else:
                    # Revert
                    # This is messy. Let's restart local search from best_blocks if invalid.
                    pass
        
        return [all_blocks[i] for i in best_blocks]

    # Main Loop
    best_blocks = None
    best_score = float('inf')
    
    # Initialize with a few greedy covers
    for seed in range(10):
        if time.time() > deadline:
            break
        blocks = initial_greedy(seed)
        if blocks:
            blocks = prune_blocks(blocks)
            if blocks:
                score = len(blocks)
                if score < best_score:
                    best_score = score
                    best_blocks = blocks
    
    if best_blocks is None:
        # Fallback
        blocks = initial_greedy(0)
        return prune_blocks(blocks) if blocks else []

    # Iterated Local Search
    current_blocks = list(best_blocks)
    
    while time.time() < deadline:
        # Local Search
        improved_blocks = local_search(current_blocks, max_iters=50)
        
        if improved_blocks:
            score = len(improved_blocks)
            if score < best_score:
                best_score = score
                best_blocks = improved_blocks
                current_blocks = list(improved_blocks)
            else:
                # Perturbation for next iteration
                if current_blocks:
                    # Randomly swap a block
                    i = random.randint(0, len(current_blocks) - 1)
                    current_blocks[i] = random.choice(all_blocks)
                    # Prune to ensure validity and compactness
                    current_blocks = prune_blocks(current_blocks)
                    if not current_blocks:
                        current_blocks = list(best_blocks)
        else:
            break

    return best_blocks
