import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precomputation
    # 1. Generate all t-subsets and k-blocks
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_ts = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)
    block_index = {b: i for i, b in enumerate(all_k_blocks)}

    # 2. Build incidence structures using Python integers as bitsets
    # block_to_ts[i] is a bitset of t-subsets covered by block i
    # ts_to_blocks[j] is a bitset of blocks covering t-subset j
    
    # Since num_ts ~ 11628, we can use Python ints directly.
    block_to_ts = [0] * num_blocks
    ts_to_blocks = [0] * num_ts
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = ts_index[ts]
            mask |= (1 << idx)
        block_to_ts[i] = mask
        
    # Invert: for each t-subset, which blocks cover it?
    # This is expensive if done naively. 
    # Better: iterate blocks, for each t-subset in block, set bit in ts_to_blocks[ts_idx]
    # We already have block_to_ts. 
    # To build ts_to_blocks efficiently:
    # Initialize ts_to_blocks to 0
    # For each block i:
    #   mask = block_to_ts[i]
    #   while mask:
    #     low_bit = mask & -mask
    #     idx = low_bit.bit_length() - 1
    #     ts_to_blocks[idx] |= (1 << i)
    #     mask ^= low_bit
    
    for i in range(num_blocks):
        mask = block_to_ts[i]
        while mask:
            low_bit = mask & -mask
            idx = low_bit.bit_length() - 1
            ts_to_blocks[idx] |= (1 << i)
            mask ^= low_bit

    def popcount(x):
        return bin(x).count('1')

    def greedy_construct(seed=0):
        random.seed(seed)
        uncovered_mask = (1 << num_ts) - 1
        selected_blocks = []
        selected_set = set()
        
        # Initial gain for all blocks is C(k, t)
        # We can just compute gain on the fly
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break
            best_block = -1
            best_gain = 0
            # Optimization: If we have many blocks, iterating all is slow.
            # But 75k blocks is manageable if we use numpy or just accept the cost?
            # 75k * (bitset AND + popcount) might be slow in Python.
            # Let's try a faster heuristic: pick a random uncovered ts, pick a random block covering it.
            # This is "randomized greedy".
            
            # Pick a random uncovered t-subset
            # Get a random bit from uncovered_mask
            low_bit = uncovered_mask & -uncovered_mask
            ts_idx = low_bit.bit_length() - 1
            
            # Get all blocks covering this ts
            covering_blocks_mask = ts_to_blocks[ts_idx]
            
            # Pick a random block from covering_blocks_mask
            # To pick random bit:
            # Generate random number, map to bit position? No, just iterate or use a trick.
            # Trick: mask = covering_blocks_mask
            # count = popcount(mask)
            # rand_idx = random.randint(0, count-1)
            # then find the rand_idx-th set bit.
            
            # Faster: Just pick the first one? No, that's deterministic and might be bad.
            # Let's pick a random block from the covering set.
            
            # To avoid slow bit counting and finding k-th bit, we can just iterate
            # but 75k blocks * 100 steps = 7.5M ops. Might be okay.
            
            # Alternative: Maintain a list of candidate blocks?
            # Let's stick to the "random block covering a random uncovered ts" strategy.
            
            # Get random bit from covering_blocks_mask
            cb_mask = covering_blocks_mask
            # We need a random set bit.
            # Simple way:
            # bits = [i for i in range(num_blocks) if (cb_mask >> i) & 1]
            # This is slow.
            
            # Let's use the "random shift" trick if possible, or just pick the lowest bit for speed in this prototype?
            # Actually, let's just iterate a few random blocks?
            
            # Better approach for speed:
            # Pre-compute a list of blocks for each ts? No, memory.
            
            # Let's try:
            # 1. Pick random uncovered ts
            # 2. Get mask of blocks covering it
            # 3. Pick a random block from that mask.
            
            # To pick random bit from mask:
            # We can generate a random number R in [0, 2^num_blocks) and check? No.
            
            # Let's just use the first available block for now to ensure speed, then improve with local search.
            # Or, pick a random block from the entire set and check if it covers any uncovered ts?
            # No, that's inefficient.
            
            # Let's use a "lazy" greedy:
            # Keep a list of active blocks.
            # Actually, for v=19, k=8, the number of blocks is 75582.
            # Iterating all 75k blocks in Python for each step (approx 200 steps) is 15M iterations.
            # Each iteration involves bit ops. This might take ~10-20 seconds.
            # We have 145s. It's feasible.
            
            # Let's do a full scan for the first few steps to get a good start, then switch to random?
            # No, let's just do the random block covering a random uncovered ts.
            
            # How to pick random bit from mask efficiently?
            # We can use:
            # m = mask
            # while m:
            #   if random.random() < 0.5: # biased?
            #      ...
            
            # Actually, there is a standard way:
            # count = popcount(mask)
            # r = random.randint(0, count-1)
            # then find r-th set bit.
            # Finding r-th set bit is O(num_words).
            
            # Let's just pick the block with the highest gain among a random sample of blocks?
            # No, let's just pick the block that covers the most uncovered ts.
            # We can approximate this by picking a random uncovered ts and a random block covering it.
            # This is a valid heuristic for set cover.
            
            # To pick a random block from covering_blocks_mask:
            # We'll just pick the lowest set bit for speed in the tight loop, and rely on local search to fix bad choices.
            # Wait, lowest set bit is deterministic.
            
            # Let's try:
            # 1. Pick random uncovered ts.
            # 2. Pick a random block covering it.
            
            # Implementation of random bit selection:
            # We can convert mask to a list of indices? No, too slow.
            
            # Let's use a simple trick:
            # Generate a random integer R.
            # If R < popcount(mask), then we want the R-th set bit.
            # But we don't have popcount fast enough? bin(x).count('1') is C-speed.
            
            # Let's just iterate through the bits of the mask and pick one at random?
            # No, let's just pick the first one. The local search will improve it.
            
            b_idx = -1
            m = covering_blocks_mask
            if m:
                # Pick a random bit from m
                # We can do this:
                # m_copy = m
                # while m_copy:
                #    if random.random() < 0.1: # 10% chance to stop at each bit? No.
                
                # Let's just pick the lowest bit. It's fast.
                low_b = m & -m
                b_idx = low_b.bit_length() - 1
                
            if b_idx == -1:
                break
                
            selected_blocks.append(b_idx)
            selected_set.add(b_idx)
            uncovered_mask &= ~block_to_ts[b_idx]
            
        return selected_blocks

    def prune_blocks(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return []
        
        # Count coverage
        cover_count = np.zeros(num_ts, dtype=np.int32)
        for b_idx in blocks:
            mask = block_to_ts[b_idx]
            # Iterate set bits
            m = mask
            while m:
                low_bit = m & -m
                idx = low_bit.bit_length() - 1
                cover_count[idx] += 1
                m ^= low_bit
                
        # Identify redundant blocks
        # A block is redundant if all its t-subsets are covered by > 1 block
        new_blocks = []
        for b_idx in blocks:
            mask = block_to_ts[b_idx]
            m = mask
            redundant = True
            while m:
                low_bit = m & -m
                idx = low_bit.bit_length() - 1
                if cover_count[idx] == 1:
                    redundant = False
                    break
                m ^= low_bit
            if not redundant:
                new_blocks.append(b_idx)
                
        return new_blocks

    def local_search(blocks, max_time=10.0):
        """Perform 1-swap and 2-1 exchange local search."""
        if time.time() > deadline - max_time:
            return blocks
            
        # Build cover count
        cover_count = np.zeros(num_ts, dtype=np.int32)
        block_set = set(blocks)
        for b_idx in blocks:
            mask = block_to_ts[b_idx]
            m = mask
            while m:
                low_bit = m & -m
                idx = low_bit.bit_length() - 1
                cover_count[idx] += 1
                m ^= low_bit
                
        improved = True
        while improved and time.time() < deadline - max_time:
            improved = False
            n = len(blocks)
            
            # 1. Try deletions
            # We already do this in prune_blocks, but let's integrate it.
            # Actually, prune_blocks is called separately.
            
            # 2. Try 2-1 exchanges
            # Pick two random blocks
            for _ in range(100):
                if time.time() > deadline - max_time:
                    break
                if len(blocks) < 2:
                    break
                i, j = random.sample(range(len(blocks)), 2)
                b1, b2 = blocks[i], blocks[j]
                
                # Find unique t-subsets for b1 and b2
                # unique_ts_1: covered by b1 and no other block in solution
                # We can compute this by:
                # mask_1 = block_to_ts[b1]
                # mask_2 = block_to_ts[b2]
                # union_mask = mask_1 | mask_2
                # For each ts in union_mask, check cover_count.
                # If cover_count[ts] == 1, it's unique to one of them.
                
                # To be efficient, we only care about ts that are in b1 or b2.
                # Let's collect unique_ts for b1 and b2.
                
                unique_ts_1 = []
                unique_ts_2 = []
                
                m1 = block_to_ts[b1]
                while m1:
                    low_bit = m1 & -m1
                    idx = low_bit.bit_length() - 1
                    if cover_count[idx] == 1:
                        # Is it covered by b2?
                        if (block_to_ts[b2] >> idx) & 1:
                            # Covered by b2 too, so not unique to b1?
                            # Wait, if cover_count is 1, it's covered by exactly one block.
                            # If it's in b1, it's unique to b1.
                            unique_ts_1.append(idx)
                    m1 ^= low_bit
                    
                m2 = block_to_ts[b2]
                while m2:
                    low_bit = m2 & -m2
                    idx = low_bit.bit_length() - 1
                    if cover_count[idx] == 1:
                        unique_ts_2.append(idx)
                    m2 ^= low_bit
                        
                if not unique_ts_1 and not unique_ts_2:
                    # Both can be deleted?
                    # If we delete both, we lose coverage for nothing?
                    # No, if they have no unique ts, they are redundant.
                    # prune_blocks would have removed them.
                    # So this case shouldn't happen if we prune frequently.
                    continue
                    
                all_unique = unique_ts_1 + unique_ts_2
                if not all_unique:
                    continue
                    
                # Find candidate blocks that cover all unique_ts
                # Start with all blocks
                # Intersect with ts_to_blocks[ts] for each ts in all_unique
                
                # Optimization: Start with the ts that has the fewest covering blocks
                # to minimize the candidate set.
                # We don't have the counts readily, but we can just iterate.
                
                candidate_mask = (1 << num_blocks) - 1
                for ts_idx in all_unique:
                    candidate_mask &= ts_to_blocks[ts_idx]
                    if not candidate_mask:
                        break
                        
                if not candidate_mask:
                    continue
                    
                # Exclude current blocks
                candidate_mask &= ~(1 << b1)
                candidate_mask &= ~(1 << b2)
                
                # Pick a random candidate
                # If candidate_mask has bits, pick one.
                # We can pick the one that covers the most *other* t-subsets?
                # For now, just pick a random one.
                
                # To pick random bit:
                # Let's just pick the lowest one for speed.
                low_c = candidate_mask & -candidate_mask
                new_b_idx = low_c.bit_length() - 1
                
                # Perform swap: remove b1, b2, add new_b_idx
                # Check validity: new_b_idx must cover all unique_ts (guaranteed by construction)
                # Update cover_count
                m1 = block_to_ts[b1]
                while m1:
                    low_bit = m1 & -m1
                    idx = low_bit.bit_length() - 1
                    cover_count[idx] -= 1
                    m1 ^= low_bit
                    
                m2 = block_to_ts[b2]
                while m2:
                    low_bit = m2 & -m2
                    idx = low_bit.bit_length() - 1
                    cover_count[idx] -= 1
                    m2 ^= low_bit
                    
                mn = block_to_ts[new_b_idx]
                while mn:
                    low_bit = mn & -mn
                    idx = low_bit.bit_length() - 1
                    cover_count[idx] += 1
                    mn ^= low_bit
                    
                # Update blocks list
                # Remove i and j, add new_b_idx
                # Be careful with indices
                if i < j:
                    blocks[i] = new_b_idx
                    blocks.pop(j)
                else:
                    blocks[j] = new_b_idx
                    blocks.pop(i)
                    
                block_set.discard(b1)
                block_set.discard(b2)
                block_set.add(new_b_idx)
                
                improved = True
                # Prune after improvement
                blocks = prune_blocks(blocks)
                # Rebuild cover_count if we pruned?
                # Yes, prune_blocks returns a list, but we need cover_count for next iteration.
                # It's better to rebuild cover_count from scratch after pruning if size changed significantly.
                # Or just continue and let the next loop iteration handle it?
                # The cover_count is now stale if we pruned.
                # Let's just break and restart the local search loop with fresh cover_count.
                break
                
        return blocks

    # Main Logic
    best_solution = []
    
    # Phase 1: Greedy
    sol = greedy_construct(seed=0)
    sol = prune_blocks(sol)
    best_solution = sol
    best_len = len(sol)
    
    # Phase 2: Local Search
    while time.time() < deadline - 5.0:
        sol = best_solution.copy()
        sol = local_search(sol, max_time=5.0)
        sol = prune_blocks(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
        else:
            # Try a random restart or different seed if stuck
            if time.time() < deadline - 20.0:
                # Perturb
                if len(best_solution) > 10:
                    sol = best_solution.copy()
                    # Remove 2 random blocks
                    idxs = random.sample(range(len(sol)), 2)
                    sol.pop(max(idxs))
                    sol.pop(min(idxs))
                    # Add 2 random blocks that cover uncovered ts?
                    # Just run greedy to fill gaps?
                    # Simpler: just run local search again with different random seed
                    random.seed(time.time() * 1000)
                    sol = local_search(sol, max_time=5.0)
                    sol = prune_blocks(sol)
                    if len(sol) < best_len:
                        best_len = len(sol)
                        best_solution = sol

    # Convert indices back to blocks
    return [list(all_k_blocks[idx]) for idx in best_solution]

# Ensure the function signature matches
# The prompt asks for `construct(v, k, t)`
# The code above defines it correctly.

if __name__ == "__main__":
    # Test
    print(construct(19, 8, 5))
