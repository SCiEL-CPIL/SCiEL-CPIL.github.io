import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    # Parameters for C(19, 8, 5)
    # v=19, k=8, t=5
    
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    num_ts = len(all_ts)
    
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    block_to_ts_indices = []
    # Map each t-subset index to list of block indices that cover it
    ts_to_block_indices = [[] for _ in range(num_ts)]
    
    for b_idx, block in enumerate(all_blocks):
        block_ts = []
        for ts in itertools.combinations(block, t):
            ts_idx = ts_index[ts]
            block_ts.append(ts_idx)
            ts_to_block_indices[ts_idx].append(b_idx)
        block_to_ts_indices.append(block_ts)
        
    # 2. Greedy Initialization
    def greedy_init(rng):
        uncovered = np.ones(num_ts, dtype=bool)
        counts = np.zeros(num_ts, dtype=np.int32)
        selected_blocks = []
        
        while np.any(uncovered) and time.time() < deadline - 5:
            # Find block with max coverage of uncovered
            best_block = -1
            best_gain = -1
            
            # Optimization: Instead of checking all blocks, we can pick a random uncovered ts
            # and look at blocks covering it. But standard greedy is fine for init.
            # To speed up, we can sample or use a heuristic.
            # For v=19, num_blocks=75k, num_ts=11k.
            # Checking all blocks every iteration is 75k * 11k ops? No, we can precompute gains?
            # Gains change as uncovered set shrinks.
            
            # Let's pick a random uncovered t-subset to anchor the search
            # This is a randomized greedy variant
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            anchor_ts_idx = rng.choice(uncovered_indices)
            
            candidates = ts_to_block_indices[anchor_ts_idx]
            
            for b_idx in candidates:
                # Calculate gain for this block
                gain = 0
                for ts_idx in block_to_ts_indices[b_idx]:
                    if uncovered[ts_idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                # Should not happen if we have a valid cover
                break
                
            selected_blocks.append(best_block)
            
            # Update uncovered and counts
            for ts_idx in block_to_ts_indices[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                counts[ts_idx] += 1
                
        return selected_blocks, counts

    # 3. Local Search
    def local_search(blocks, counts, rng):
        # blocks: list of block indices
        # counts: np array of coverage counts
        # Returns improved blocks and counts
        
        blocks = list(blocks)
        counts = counts.copy()
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            n = len(blocks)
            for i in range(n):
                if time.time() >= deadline - 2:
                    break
                b_idx = blocks[i]
                redundant = True
                for ts_idx in block_to_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for ts_idx in block_to_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break # Restart loop as indices changed
            
            if improved:
                continue
                
            # Phase 2: Swap/Replace
            # Try to replace a block with another that allows further reduction
            # We iterate through blocks and try to find a replacement
            # To avoid O(N^2 * M) complexity, we limit the search depth or candidates
            
            # Shuffle blocks to randomize order
            # rng.shuffle(blocks) is not enough, we need to try swaps
            
            for i in range(len(blocks)):
                if time.time() >= deadline - 2:
                    break
                    
                b_idx = blocks[i]
                
                # Find t-subsets uniquely covered by b_idx
                unique_ts = []
                for ts_idx in block_to_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                        
                if not unique_ts:
                    # This block is redundant, should have been caught in Phase 1
                    # But maybe counts changed? No, Phase 1 ran to completion.
                    # If unique_ts is empty, it IS redundant.
                    # Remove it.
                    for ts_idx in block_to_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                    
                # We must cover unique_ts
                # Candidate blocks must cover ALL unique_ts
                # Pick the first unique_ts to filter candidates
                anchor_ts = unique_ts[0]
                candidates = ts_to_block_indices[anchor_ts]
                
                best_new_block = -1
                best_score = -1
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    
                    # Check if cand_idx covers all unique_ts
                    cand_ts_set = set(block_to_ts_indices[cand_idx])
                    if all(ts in cand_ts_set for ts in unique_ts):
                        # Calculate score: how many other ts does it cover that are count==1?
                        # Or just prefer blocks that are "different"
                        # Heuristic: Count how many ts in cand_idx have count == 1
                        # This indicates potential for making other blocks redundant?
                        # Actually, if we replace b_idx with cand_idx, the counts of ts in cand_idx \ b_idx increase.
                        # The counts of ts in b_idx \ cand_idx decrease.
                        
                        # Let's just try the swap if it's valid (covers unique_ts)
                        # and see if it leads to redundancy removal.
                        # To avoid trying all candidates, we can just pick the first valid one
                        # or a random one.
                        
                        # Let's score by the number of ts in cand_idx that are currently count==1
                        # (excluding those in b_idx which are already count==1)
                        score = 0
                        b_ts_set = set(block_to_ts_indices[b_idx])
                        for ts_idx in block_to_ts_indices[cand_idx]:
                            if ts_idx not in b_ts_set and counts[ts_idx] == 1:
                                score += 1
                                
                        if score > best_score:
                            best_score = score
                            best_new_block = cand_idx
                            
                if best_new_block != -1:
                    # Perform swap
                    # 1. Remove old block
                    for ts_idx in block_to_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    # 2. Add new block
                    for ts_idx in block_to_ts_indices[best_new_block]:
                        counts[ts_idx] += 1
                    # 3. Update blocks list
                    blocks[i] = best_new_block
                    improved = True
                    break # Restart loop to check for new redundancies
                    
        return blocks, counts

    # 4. Main Loop
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple restarts
    # Estimate time per greedy + local search
    # Greedy is fast. Local search is the bottleneck.
    # Let's do 1 restart with long local search, or multiple with shorter?
    # Given 115s budget, we can do a few restarts.
    
    num_restarts = 3
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Greedy Init
        blocks, counts = greedy_init(rng)
        
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            
        # Local Search
        # Allocate remaining time to local search
        ls_start = time.time()
        ls_deadline = deadline - 5
        # If we have multiple restarts, we should split time.
        # For simplicity, let's just run local search until time runs out for this restart
        # or until no improvement.
        
        improved_blocks, _ = local_search(blocks, counts, rng)
        
        if len(improved_blocks) < best_count:
            best_blocks = improved_blocks
            best_count = len(improved_blocks)
            
        # If time is tight, break
        if time.time() > deadline - 20:
             break
             
    # Convert block indices back to lists of points
    if best_blocks is None:
        return []
        
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
