import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_elapsed():
        return time.time() - start_time

    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    # We represent this as a sparse structure or dense numpy matrix if feasible
    # num_blocks ~ 75582, num_tsubs ~ 11628
    # Dense matrix would be 75582 * 11628 * 1 byte ~ 878 MB. Too big for fast ops?
    # Let's use a list of arrays for block -> tsub indices, and tsub -> block indices.
    
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(np.array(tsubs_in_b, dtype=np.int32))
        
    # Inverse: for each t-subset, which blocks cover it?
    # This is needed for greedy repair if we use it, but for probabilistic rounding
    # we just sample blocks. For reduction, we need to know if a t-subset is still covered.
    # We can maintain a coverage count array.
    
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Strategy: Probabilistic Rounding
    # We want to select a set of blocks S such that every t-subset is covered.
    # Let p be the probability of selecting a block.
    # To minimize |S|, we want p small, but p must be large enough to cover all t-subsets.
    # Expected number of blocks = num_blocks * p.
    # Probability a specific t-subset is NOT covered by a random block:
    # A random block contains a specific t-subset with prob:
    # C(v-t, k-t) / C(v, k)
    # Let q = C(v-t, k-t) / C(v, k).
    # Prob not covered by one block = 1-q.
    # Prob not covered by m blocks (independent approx) = (1-q)^m.
    # We want (1-q)^m <= 1/num_tsubs (union bound heuristic).
    # m >= log(1/num_tsubs) / log(1-q)
    
    # However, simple uniform sampling is suboptimal.
    # We can use a weighted sampling where blocks covering "rare" or "hard" t-subsets are picked more often?
    # Actually, in a symmetric design (v, k, t), all t-subsets are equally likely to be covered by a random block.
    # So uniform sampling is optimal for the first-order approximation.
    # But dependencies matter.
    
    # Let's calculate the theoretical lower bound on m for uniform sampling.
    q = math.comb(v-t, k-t) / math.comb(v, k)
    # We need m such that (1-q)^m * num_tsubs < 1 (high prob of full coverage)
    # Let's aim for (1-q)^m = 1e-6 or so.
    # m = log(1e-6) / log(1-q)
    
    # But we can do better with a "greedy repair" of the probabilistic set.
    
    def probabilistic_rounding(num_samples):
        # Sample num_samples blocks with replacement? No, distinct blocks.
        # We can sample without replacement if num_samples < num_blocks.
        # Or just sample indices.
        # If we sample with replacement, we might pick same block twice. We should deduplicate.
        # Let's sample a set of indices.
        
        # To ensure diversity, we can use a random permutation of all blocks and take the first num_samples?
        # Or just random.sample if num_samples is small.
        # num_blocks is ~75k.
        
        if num_samples >= num_blocks:
            return list(range(num_blocks))
        
        return random.sample(range(num_blocks), num_samples)
        
    def check_coverage_and_repair(block_indices):
        # Check if coverage is complete. If not, add blocks greedily.
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs[b_idx]] += 1
            
        uncovered_mask = cover_count == 0
        uncovered_count = np.sum(uncovered_mask)
        
        if uncovered_count == 0:
            return list(block_indices)
            
        # Greedy repair: add blocks that cover most uncovered t-subsets
        # To speed this up, we can precompute something?
        # For now, simple greedy loop.
        selected = set(block_indices)
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 5.0:
                break
                
            # Find a random uncovered t-subset
            uncov_indices = np.where(uncovered_mask)[0]
            if len(uncov_indices) == 0:
                break
            target_tsub_idx = uncov_indices[random.randint(0, len(uncov_indices)-1)]
            
            # Find best block covering this t-subset that also covers other uncovered ones
            candidates = tsub_blocks[target_tsub_idx]
            best_b = -1
            best_gain = -1
            
            # Sample candidates if too many
            if len(candidates) > 50:
                sampled = random.sample(candidates, 50)
            else:
                sampled = candidates
                
            for b_idx in sampled:
                if b_idx in selected:
                    continue
                # Count how many currently uncovered tsubs this block covers
                gain = np.sum(cover_count[block_tsubs[b_idx]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
                    
            if best_b == -1:
                # Fallback: just pick first candidate not in selected
                for b_idx in candidates:
                    if b_idx not in selected:
                        best_b = b_idx
                        break
                if best_b == -1:
                    break # Should not happen if coverage is possible
            
            selected.add(best_b)
            # Update coverage
            new_cover = block_tsubs[best_b]
            cover_count[new_cover] += 1
            # Update uncovered mask and count
            newly_covered = new_cover[cover_count[new_cover] == 1]
            if len(newly_covered) > 0:
                uncovered_mask[newly_covered] = False
                uncovered_count -= len(newly_covered)
                
        return list(selected)
        
    def vectorized_reduce(block_indices):
        # Remove redundant blocks.
        # A block is redundant if every t-subset it covers is covered by at least one OTHER block.
        # Equivalent: if we remove it, coverage doesn't drop to 0 anywhere.
        # We maintain cover_count.
        # If min(cover_count[block_tsubs[b]]) > 1 for all b in block, then b is redundant?
        # No, if cover_count[j] == 1 for some j in block, then block is essential for j.
        # So block is redundant iff min(cover_count[block_tsubs[b]]) >= 2.
        
        block_indices = list(set(block_indices))
        if not block_indices:
            return []
            
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs[b_idx]] += 1
            
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 5.0:
                break
                
            new_block_indices = []
            for b_idx in block_indices:
                # Check if essential
                # min cover count among its tsubs
                # If min >= 2, it's redundant
                if np.min(cover_count[block_tsubs[b_idx]]) >= 2:
                    # Remove it
                    cover_count[block_tsubs[b_idx]] -= 1
                    changed = True
                else:
                    new_block_indices.append(b_idx)
                    
            block_indices = new_block_indices
            
        return block_indices
        
    # Main Loop
    best_blocks = []
    best_size = float('inf')
    
    # Estimate optimal sample size
    # q = prob a random block covers a specific t-subset
    q = math.comb(v-t, k-t) / math.comb(v, k)
    # We want high probability of covering all.
    # Let's try a few sample sizes around the theoretical minimum.
    # Theoretical m ~ log(num_tsubs) / q  (since 1-q ~ 1 for small q? No, q is not small)
    # q for v=19, k=8, t=5:
    # C(14, 3) / C(19, 8) = 364 / 75582 ~ 0.0048
    # (1-q)^m = 1e-6 => m * q ~ 13.8 => m ~ 2875
    # Let's try sample sizes: 2000, 2500, 3000, 3500
    
    sample_sizes = [2000, 2500, 3000, 3500, 4000]
    
    # Run multiple trials
    trials = 0
    while get_elapsed() < deadline - 15.0:
        # Pick a sample size
        # Maybe start small and increase? Or random?
        # Let's cycle through them.
        s_size = sample_sizes[trials % len(sample_sizes)]
        
        # Sample blocks
        sampled_indices = probabilistic_rounding(s_size)
        
        # Repair
        repaired = check_coverage_and_repair(sampled_indices)
        
        # Reduce
        reduced = vectorized_reduce(repaired)
        
        # Check validity (just in case repair failed due to time)
        # If reduced is empty or invalid, skip
        if len(reduced) > 0:
             # Quick validity check? 
             # If repair succeeded, it should be valid. 
             # But if repair timed out, it might be invalid.
             # Let's assume valid if repair loop finished or uncovered_count was 0.
             # To be safe, we can do a final check.
             
             # Count size
             if len(reduced) < best_size:
                 best_size = len(reduced)
                 best_blocks = reduced
                 
        trials += 1
        # If we found a very good solution, we can stop early?
        # Lower bound is 264. Best known 335.
        # If we get close to 300, it's great.
        
        # Diversification: if we've tried all sample sizes, maybe perturb?
        # For now, just repeat with different random samples.
        
    # Final safety check and formatting
    if not best_blocks:
        # Fallback: simple greedy
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        while uncovered_count > 0 and get_elapsed() < deadline - 1.0:
            # Pick random uncovered
            while True:
                s_idx = random.randint(0, num_tsubs - 1)
                if uncovered[s_idx]:
                    break
            candidates = tsub_blocks[s_idx]
            best_b = -1
            best_gain = -1
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b != -1:
                selected_blocks.append(best_b)
                for j in block_tsubs[best_b]:
                    if uncovered[j]:
                        uncovered[j] = 0
                        uncovered_count -= 1
        best_blocks = selected_blocks
        best_blocks = vectorized_reduce(best_blocks)

    return [list(all_blocks[i]) for i in best_blocks]
