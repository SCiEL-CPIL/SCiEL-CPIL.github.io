import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Helper for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # If parameters are not the specific one, use a simple fallback
    if v != 19 or k != 8 or t != 5:
        # Fallback: Standard Greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            best_block = None
            best_gain = -1
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    # Specific optimization for v=19, k=8, t=5
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage structures
    # block_to_ts[j]: list of t-subset indices covered by block j
    block_to_ts = []
    # ts_to_blocks[i]: list of block indices covering t-subset i
    ts_to_blocks = [[] for _ in range(n_t)]
    
    for j, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)
        
    # Convert to numpy arrays for speed
    block_ts_arrays = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    # ts_to_blocks remains as list of lists for easier iteration in gain updates if needed, 
    # but we will use a matrix or sparse representation if possible. 
    # For now, list of lists is fine for 11628 subsets.

    def run_packing_greedy(seed):
        """
        1. Build a maximal t-packing (blocks with disjoint t-subsets).
        2. Complete the cover greedily, prioritizing uncovered t-subsets.
        """
        random.seed(seed)
        is_selected = np.full(n_blocks, False, dtype=bool)
        is_uncovered = np.full(n_t, True, dtype=bool)
        sol = []
        
        # Phase 1: Maximal t-packing
        # We want to select blocks that do not share any t-subsets with previously selected blocks.
        # This ensures that each selected block covers C(k, t) *unique* t-subsets.
        # We pick blocks randomly from those that are still "valid" (don't conflict).
        
        # To speed up conflict checking, we can maintain a set of covered t-subsets.
        # But C(19,5) is 11628, so a boolean array is fine.
        
        # Randomize block order to get different packings
        block_indices = list(range(n_blocks))
        random.shuffle(block_indices)
        
        for b_idx in block_indices:
            if time.time() > deadline - 20.0:
                break
            # Check if this block conflicts with any already selected block in the packing
            # i.e., if any of its t-subsets are already covered by the packing
            conflict = False
            for ts_idx in block_to_ts[b_idx]:
                if not is_uncovered[ts_idx]:
                    conflict = True
                    break
            
            if not conflict:
                # Add to packing
                sol.append(list(all_blocks[b_idx]))
                is_selected[b_idx] = True
                for ts_idx in block_to_ts[b_idx]:
                    is_uncovered[ts_idx] = False
        
        # Phase 2: Greedy Completion
        # Now we have a set of blocks (the packing) and some uncovered t-subsets.
        # We continue greedily selecting blocks that cover the most uncovered t-subsets.
        
        # Calculate initial gains
        gain = np.zeros(n_blocks, dtype=np.int32)
        for j in range(n_blocks):
            if is_selected[j]:
                continue
            # Count how many uncovered t-subsets this block covers
            # We can do this by iterating block_to_ts[j] and checking is_uncovered
            # This is O(C(k,t)) per block, which is 56.
            # Total O(n_blocks * 56) ~ 4M operations. Acceptable.
            count = 0
            for ts_idx in block_to_ts[j]:
                if is_uncovered[ts_idx]:
                    count += 1
            gain[j] = count
        
        # Greedy loop
        while True:
            if time.time() > deadline - 5.0:
                return sol
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Randomly pick one to diversify
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1 # Mark as selected
            
            # Identify newly covered t-subsets
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            # Update gains for other blocks
            # A block's gain decreases by the number of its t-subsets that are in newly_covered_ts
            # We need to decrement gain[j] for all j such that block j contains any ts in newly_covered_ts
            
            # Efficient update:
            # For each ts in newly_covered_ts, iterate through ts_to_blocks[ts] and decrement gain
            # But we must avoid double counting if a block covers multiple newly_covered_ts.
            # Since newly_covered_ts are disjoint for a single block addition? No, a block adds 56 ts.
            # We need to know how many of the 56 ts of block j are in newly_covered_ts.
            
            # Let's use a temporary array or set for newly_covered_ts for O(1) lookup
            new_ts_set = set(newly_covered_ts)
            
            # Iterate through all blocks? No, only those that might be affected.
            # The affected blocks are those that contain at least one ts in newly_covered_ts.
            # We can collect these block indices.
            
            affected_blocks = set()
            for ts_idx in newly_covered_ts:
                affected_blocks.update(ts_to_blocks[ts_idx])
            
            # For each affected block, recalculate its gain contribution from newly_covered_ts
            # Actually, it's easier to just decrement gain[j] by the count of intersection.
            # Since we don't have a precomputed intersection matrix, we can do:
            # For each affected block j:
            #   decrement = sum(1 for ts in block_to_ts[j] if ts in new_ts_set)
            #   gain[j] -= decrement
            
            # To speed this up, we can precompute the list of blocks for each ts.
            # But checking intersection is still costly if done naively.
            
            # Alternative: 
            # We know gain[j] was the count of uncovered ts in block j.
            # Now some of those are covered.
            # We can just recalculate gain[j] for affected blocks if the number of affected blocks is small.
            # How many blocks cover a specific ts? C(19-5, 8-5) = C(14,3) = 364.
            # So each ts is in 364 blocks.
            # 56 newly covered ts * 364 blocks ~ 20k blocks affected (with duplicates).
            # Recalculating gain for 20k blocks * 56 ts each = 1.1M ops. This is fine.
            
            for j in affected_blocks:
                if is_selected[j]:
                    continue
                # Recalculate gain for block j
                # Count uncovered ts in block j
                cnt = 0
                for ts_idx in block_to_ts[j]:
                    if is_uncovered[ts_idx]:
                        cnt += 1
                gain[j] = cnt

        return sol

    def optimize_redundancy(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
            
        # Map block tuple to index in all_blocks for quick lookup
        # Or just compute t-subsets directly
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_indices = []
        
        # First pass: count coverage
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            if idx >= 0:
                ts_list = block_to_ts[idx]
            else:
                ts_list = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_ts_indices.append(ts_list)
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                    
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break
                    
        return blocks

    def local_search_swap(blocks, max_iters=100):
        """
        Try to swap blocks to improve the solution.
        A block is "bad" if it covers few unique subsets.
        We try to replace a block with another that covers more unique subsets, 
        or allows removing another block.
        
        Simple heuristic: 
        1. Identify blocks with low "unique coverage" (subsets covered only by this block).
        2. Try to replace one such block with a random other block.
        3. If the new block covers all the unique subsets of the old block (and maybe more), 
           and doesn't break the cover, accept.
        4. Then run redundancy removal.
        """
        if not blocks:
            return blocks
        
        # Precompute unique coverage for each block
        # unique_count[i] = number of t-subsets covered ONLY by block i
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            if idx >= 0:
                ts_list = block_to_ts[idx]
            else:
                ts_list = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_ts_indices.append(ts_list)
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
        
        # Calculate unique counts
        unique_counts = []
        for i in range(len(blocks)):
            cnt = 0
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    cnt += 1
            unique_counts.append(cnt)
        
        # Sort blocks by unique count (ascending) to target "bad" blocks
        # We don't want to disrupt the order too much, but we can pick candidates
        
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a block with low unique count
            # Filter out blocks with 0 unique count? No, they are redundant. 
            # optimize_redundancy should have removed them. 
            # So we look for blocks with small unique count > 0.
            
            candidates = [i for i, c in enumerate(unique_counts) if c > 0 and c < 10]
            if not candidates:
                break
                
            i = random.choice(candidates)
            old_block = blocks[i]
            old_ts = block_ts_indices[i]
            
            # Find a replacement block
            # We want a block that covers all ts in old_ts where cover_count[ts] == 1
            # Let critical_ts = [ts for ts in old_ts if cover_count[ts] == 1]
            critical_ts = set(ts for ts in old_ts if cover_count[ts] == 1)
            
            if not critical_ts:
                # This block is redundant?
                # Just remove it?
                # optimize_redundancy should handle this.
                continue
                
            # Try random blocks
            # We can't try all 75k blocks.
            # Sample 100 random blocks.
            found_replacement = False
            for _ in range(100):
                new_idx = random.randint(0, n_blocks - 1)
                if new_idx == block_to_idx.get(tuple(old_block), -1):
                    continue
                
                new_block = all_blocks[new_idx]
                new_ts = block_to_ts[new_idx]
                
                # Check if new_block covers all critical_ts
                # Convert critical_ts to set of tuples for fast lookup? 
                # Or just check if critical_ts is subset of set(new_ts)
                # new_ts is a list of indices.
                new_ts_set = set(new_ts)
                
                if critical_ts.issubset(new_ts_set):
                    # Accept swap
                    blocks[i] = list(new_block)
                    block_ts_indices[i] = new_ts
                    # Update cover_count
                    for ts_idx in old_ts:
                        cover_count[ts_idx] -= 1
                    for ts_idx in new_ts:
                        cover_count[ts_idx] += 1
                    
                    # Recalculate unique counts for affected blocks?
                    # For now, just break and run redundancy removal
                    improved = True
                    found_replacement = True
                    break
            
            if not found_replacement:
                continue

        # Final redundancy removal
        blocks = optimize_redundancy(blocks)
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized packing-greedy restarts
    num_restarts = 100 # Reduced count to allow time for local search
    
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        sol = run_packing_greedy(i)
        if sol is None:
            continue
            
        # Post-process: remove redundant blocks
        sol = optimize_redundancy(sol)
        
        # Local search
        sol = local_search_swap(sol, max_iters=20)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = [list(b) for b in sol]
            
        # If we hit the theoretical lower bound, we can stop
        if best_len <= 208:
            break

    if best_blocks is None:
        # Fallback: deterministic greedy
        best_blocks = run_packing_greedy(0)
        if best_blocks is None:
            return []
        best_blocks = optimize_redundancy(best_blocks)

    return best_blocks
