import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 112  # Safety margin (120s limit, 8s buffer)
    
    all_points = list(range(v))
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the indices of t-subsets it covers
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [t_sub_to_idx[tsub] for tsub in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # For each t-subset, precompute the list of blocks that cover it
    # We will use this for quick lookups in local search if needed, but mainly for greedy init?
    # Actually, for greedy we just need the block's tsubs.
    
    def time_ok():
        return time.time() < deadline

    def greedy_construct(seed):
        """
        Greedy construction with random tie-breaking.
        Returns list of block IDs.
        """
        rng = random.Random(seed)
        covered_count = [0] * n_t
        uncovered_count = n_t
        blocks_in_solution = []
        used_block_ids = set()
        
        # Heap of (-gain, random_tiebreaker, block_id)
        # Using random tiebreaker helps diversify runs
        heap = []
        for i in range(n_blocks):
            gain = len(block_tsubs[i]) # C(k, t)
            # Add a small random noise to break ties and encourage diversity
            # But gain is integer, so we can just use (-(gain), rng.random(), i)
            # However, standard greedy is deterministic. Let's use a "noisy" gain.
            # gain + rng.random() * 0.1 ? No, gain is count.
            # Let's just use random tie-breaking for equal gains.
            heapq.heappush(heap, (-gain, rng.random(), i))
            
        while uncovered_count > 0 and time_ok():
            valid_block = None
            while heap:
                neg_gain, _, b_id = heapq.heappop(heap)
                # Recalculate gain
                current_gain = 0
                for tsub_idx in block_tsubs[b_id]:
                    if covered_count[tsub_idx] == 0:
                        current_gain += 1
                if current_gain == -neg_gain:
                    valid_block = b_id
                    break
                else:
                    if current_gain > 0:
                        heapq.heappush(heap, (-current_gain, rng.random(), b_id))
            
            if valid_block is None:
                break
            
            blocks_in_solution.append(valid_block)
            used_block_ids.add(valid_block)
            
            for tsub_idx in block_tsubs[valid_block]:
                if covered_count[tsub_idx] == 0:
                    uncovered_count -= 1
                covered_count[tsub_idx] += 1
                
        return blocks_in_solution

    def local_search(block_ids, seed):
        """
        Local search to minimize the number of blocks.
        1. Remove redundant blocks.
        2. Swap blocks to enable further removals.
        """
        rng = random.Random(seed)
        sol_blocks = block_ids[:]
        
        # Build cover_map: tsub_idx -> list of block_ids
        cover_map = defaultdict(list)
        for b_id in sol_blocks:
            for tsub_idx in block_tsubs[b_id]:
                cover_map[tsub_idx].append(b_id)
                
        # Helper to check redundancy
        def is_redundant(b_id):
            for tsub_idx in block_tsubs[b_id]:
                if len(cover_map[tsub_idx]) == 1:
                    return False
            return True
            
        def remove_block(b_id):
            sol_blocks.remove(b_id)
            for tsub_idx in block_tsubs[b_id]:
                lst = cover_map[tsub_idx]
                # Remove b_id from lst
                try:
                    lst.remove(b_id)
                except ValueError:
                    pass
                if not lst:
                    del cover_map[tsub_idx]
                    
        def add_block(b_id):
            sol_blocks.append(b_id)
            for tsub_idx in block_tsubs[b_id]:
                if tsub_idx not in cover_map:
                    cover_map[tsub_idx] = []
                cover_map[tsub_idx].append(b_id)

        # Pass 1: Remove redundant blocks
        changed = True
        while changed and time_ok():
            changed = False
            # Iterate backwards to handle removals safely
            for i in range(len(sol_blocks) - 1, -1, -1):
                if is_redundant(sol_blocks[i]):
                    remove_block(sol_blocks[i])
                    changed = True

        # Pass 2: Swap-based local search
        # We try to replace a block with another that covers its unique t-subsets
        # and potentially helps remove other blocks.
        
        improved = True
        while improved and time_ok():
            improved = False
            # Shuffle order to avoid bias?
            indices = list(range(len(sol_blocks)))
            rng.shuffle(indices)
            
            for idx in indices:
                if not time_ok(): break
                if idx >= len(sol_blocks): continue
                
                b_id = sol_blocks[idx]
                
                # Find unique t-subsets for b_id
                unique_tsubs = []
                for tsub_idx in block_tsubs[b_id]:
                    if len(cover_map[tsub_idx]) == 1:
                        unique_tsubs.append(tsub_idx)
                        
                if not unique_tsubs:
                    # Should be removed in Pass 1, but maybe state changed
                    remove_block(b_id)
                    improved = True
                    break # Restart loop
                
                # Find a replacement block
                # The replacement must cover all unique_tsubs.
                # To speed up, we can intersect the candidate sets for each unique tsub.
                # But we don't have tsub_blocks precomputed as a list of lists of block IDs?
                # We can compute it on the fly or precompute.
                # Let's precompute tsub_blocks for efficiency in this local search.
                # Wait, I didn't precompute tsub_blocks in the main scope.
                # Let's create it here if not exists? Or just iterate?
                # n_blocks is 50k. unique_tsubs is small.
                # Iterating 50k blocks and checking membership in block_tsubs[new_b_id] is O(50k * len(unique_tsubs)).
                # This might be slow if done many times.
                
                # Optimization: Precompute a set for each block's tsubs for fast lookup?
                # block_tsubs is a list of lists. Converting to set is expensive.
                # But we can check `tsub_idx in block_tsubs[new_b_id]`.
                
                best_new_block = -1
                best_score = -1
                
                # Candidate generation:
                # A valid new block must contain the points of all unique t-subsets.
                # Let U be the union of points in unique_tsubs.
                # If len(U) > k, no replacement exists.
                
                # To find candidates, we can look at the blocks covering the first unique tsub.
                # We need a mapping tsub_idx -> list of block_ids.
                # Let's build it once outside the loop?
                
                # For now, let's just iterate all blocks.
                # If this is too slow, we'll optimize.
                
                for new_b_id in range(n_blocks):
                    if new_b_id == b_id:
                        continue
                    if new_b_id in sol_blocks:
                        continue
                        
                    # Check if new_b_id covers all unique_tsubs
                    valid = True
                    for tsub_idx in unique_tsubs:
                        if tsub_idx not in block_tsubs[new_b_id]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: How many other blocks' unique t-subsets does new_b_id cover?
                    # Or simply: how many t-subsets does it cover that are currently covered by exactly 2 blocks?
                    # If we add new_b_id, those t-subsets will have coverage 3.
                    # If we remove b_id, the unique_tsubs of b_id are covered by new_b_id.
                    # What about other t-subsets covered by b_id?
                    # If a tsub is covered by b_id and one other block, removing b_id leaves it covered by 1.
                    # So b_id is "critical" for those too.
                    # Actually, the condition for removal is that ALL tsubs of b_id are covered by others.
                    # So if we replace b_id with new_b_id, we need new_b_id to cover ALL tsubs of b_id?
                    # No.
                    # If we replace b_id with new_b_id:
                    # 1. Remove b_id.
                    # 2. Add new_b_id.
                    # The coverage of any tsub T:
                    #   count(T) = old_count(T) - (1 if T in b_id else 0) + (1 if T in new_b_id else 0)
                    # We need count(T) >= 1 for all T.
                    # For T in unique_tsubs(b_id): old_count=1. T in b_id.
                    #   new_count = 1 - 1 + (1 if T in new_b_id else 0).
                    #   So we MUST have T in new_b_id.
                    # For T in b_id but NOT in unique_tsubs: old_count >= 2.
                    #   If T not in new_b_id: new_count = old_count - 1 >= 1. OK.
                    #   If T in new_b_id: new_count = old_count. OK.
                    # For T not in b_id:
                    #   If T in new_b_id: new_count = old_count + 1. OK.
                    #   If T not in new_b_id: new_count = old_count. OK.
                    
                    # So the ONLY constraint is that new_b_id must cover all unique_tsubs of b_id.
                    # The "score" should be how much this swap helps future removals.
                    # A good heuristic: new_b_id should cover many t-subsets that are currently "weakly covered" (count=2) by other blocks?
                    # Or simply: pick the new_b_id that maximizes the number of t-subsets it shares with other blocks in the solution?
                    # Let's just pick the first valid one to save time.
                    # Or pick one that is not in the solution (already checked).
                    
                    best_new_block = new_b_id
                    break # Take the first one to save time
                    
                if best_new_block != -1:
                    remove_block(b_id)
                    add_block(best_new_block)
                    improved = True
                    break # Restart loop to check for new redundancies
        
        return sol_blocks

    # Main Loop: Run multiple iterations of Greedy + Local Search
    best_solution = None
    best_size = float('inf')
    
    # Run a few seeds
    # Time budget allocation:
    # Each greedy is fast. Local search is the bottleneck.
    # Let's run 5-10 iterations.
    
    num_iterations = 10
    for i in range(num_iterations):
        if not time_ok():
            break
            
        seed = 12345 + i
        greedy_blocks = greedy_construct(seed)
        
        if not greedy_blocks:
            continue
            
        # Local search
        ls_blocks = local_search(greedy_blocks, seed)
        
        if len(ls_blocks) < best_size:
            best_size = len(ls_blocks)
            best_solution = ls_blocks
            
        # Optional: If we find a very good solution, we can try more aggressive local search?
        # For now, just keep the best.
        
    if best_solution is None:
        # Fallback: Return a single greedy result
        greedy_blocks = greedy_construct(0)
        best_solution = greedy_blocks
        
    # Convert block IDs to lists of points
    return [list(all_blocks[i]) for i in best_solution]
