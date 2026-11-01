import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    # Parameters for C(19, 8, 5)
    # v=19, k=8, t=5
    
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    num_ts = len(all_ts)
    
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers (as numpy arrays for speed)
    block_to_ts_indices = []
    # Map each t-subset index to list of block indices that cover it
    ts_to_block_indices = [[] for _ in range(num_ts)]
    
    for b_idx, block in enumerate(all_blocks):
        block_ts = []
        for ts in itertools.combinations(block, t):
            ts_idx = ts_index[ts]
            block_ts.append(ts_idx)
            ts_to_block_indices[ts_idx].append(b_idx)
        block_to_ts_indices.append(np.array(block_ts, dtype=np.int32))
    
    # Convert ts_to_block_indices to numpy arrays for faster iteration
    ts_to_block_indices_np = [np.array(lst, dtype=np.int32) for lst in ts_to_block_indices]
    
    # 2. Greedy Initialization
    def greedy_init(rng):
        uncovered = np.ones(num_ts, dtype=bool)
        counts = np.zeros(num_ts, dtype=np.int32)
        selected_blocks = []
        
        while np.any(uncovered) and time.time() < deadline - 3:
            # Pick a random uncovered t-subset to anchor the search
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            anchor_ts_idx = int(rng.choice(uncovered_indices))
            
            candidates = ts_to_block_indices_np[anchor_ts_idx]
            
            best_block = -1
            best_gain = -1
            
            for b_idx in candidates:
                b_idx = int(b_idx)
                # Calculate gain using numpy
                gain = np.sum(uncovered[block_to_ts_indices[b_idx]])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected_blocks.append(best_block)
            
            # Update uncovered and counts using numpy
            block_ts = block_to_ts_indices[best_block]
            uncovered[block_ts] = False
            counts[block_ts] += 1
                
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
                # Check if block is redundant using numpy
                if np.all(counts[block_to_ts_indices[b_idx]] > 1):
                    # Remove block i
                    counts[block_to_ts_indices[b_idx]] -= 1
                    blocks.pop(i)
                    improved = True
                    break # Restart loop as indices changed
            
            if improved:
                continue
                
            # Phase 2: Swap/Replace
            # Try to replace a block with another that allows further reduction
            for i in range(len(blocks)):
                if time.time() >= deadline - 2:
                    break
                    
                b_idx = blocks[i]
                
                # Find t-subsets uniquely covered by b_idx using numpy
                block_ts = block_to_ts_indices[b_idx]
                unique_mask = counts[block_ts] == 1
                unique_ts = block_ts[unique_mask]
                        
                if len(unique_ts) == 0:
                    # This block is redundant
                    counts[block_ts] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                    
                # We must cover all unique_ts
                # Candidate blocks must cover ALL unique_ts
                # Pick the first unique_ts to filter candidates
                anchor_ts = int(unique_ts[0])
                candidates = ts_to_block_indices_np[anchor_ts]
                
                best_new_block = -1
                best_score = -1
                
                # Precompute the set of ts in current block for fast lookup
                b_ts_set = set(block_ts.tolist())
                
                for cand_idx in candidates:
                    cand_idx = int(cand_idx)
                    if cand_idx == b_idx:
                        continue
                    
                    # Check if cand_idx covers all unique_ts
                    cand_ts = block_to_ts_indices[cand_idx]
                    cand_ts_set = set(cand_ts.tolist())
                    if all(int(ts) in cand_ts_set for ts in unique_ts):
                        # Score: number of ts in cand_idx not in b_idx that have count==1
                        # This indicates potential for making other blocks redundant
                        diff_mask = ~np.isin(cand_ts, block_ts)
                        score = np.sum(counts[cand_ts[diff_mask]] == 1)
                                
                        if score > best_score:
                            best_score = score
                            best_new_block = cand_idx
                            
                if best_new_block != -1:
                    # Perform swap
                    counts[block_to_ts_indices[b_idx]] -= 1
                    counts[block_to_ts_indices[best_new_block]] += 1
                    blocks[i] = best_new_block
                    improved = True
                    break # Restart loop to check for new redundancies
                    
            # Phase 3: Try to remove 2 blocks and replace with 1 (2-for-1 swap)
            if not improved and time.time() < deadline - 5:
                n = len(blocks)
                if n >= 2:
                    # Sample a few pairs to try
                    num_pairs_to_try = min(20, n * (n - 1) // 2)
                    pairs_tried = set()
                    found_improvement = False
                    
                    for _ in range(num_pairs_to_try):
                        if time.time() >= deadline - 3:
                            break
                        i, j = rng.sample(range(n), 2)
                        if (min(i,j), max(i,j)) in pairs_tried:
                            continue
                        pairs_tried.add((min(i,j), max(i,j)))
                        
                        b1_idx = blocks[i]
                        b2_idx = blocks[j]
                        
                        # Find ts uniquely covered by b1 or b2
                        ts1 = block_to_ts_indices[b1_idx]
                        ts2 = block_to_ts_indices[b2_idx]
                        
                        # Union of ts covered by b1 and b2
                        all_ts_in_pair = np.union1d(ts1, ts2)
                        
                        # Ts that would become uncovered if we remove both
                        # These are ts where count==1 AND in b1 or b2
                        # But we need to be careful: if a ts is in both b1 and b2, count==2, removing both makes count 0
                        # So: ts is "at risk" if it's in (b1 ∪ b2) and count[ts] <= 2
                        # More precisely: after removing both, count becomes count[ts] - (1 if ts in b1) - (1 if ts in b2)
                        # We need this to be >= 0 for all ts, and > 0 for all ts not covered by replacement
                        # Actually, we need: for all ts, new_count[ts] >= 1
                        # new_count[ts] = count[ts] - (1 if ts in b1) - (1 if ts in b2) + (1 if ts in new_block)
                        
                        # The ts that are "critical" are those where count[ts] - (1 if ts in b1) - (1 if ts in b2) <= 0
                        # i.e., ts must be covered by the new block
                        in_b1 = np.isin(np.arange(num_ts), ts1)
                        in_b2 = np.isin(np.arange(num_ts), ts2)
                        
                        # Critical ts: those where count <= 1 + in_b1 + in_b2 but would drop to 0
                        # Simpler: critical_ts = ts where count[ts] == 1 and ts in (b1∪b2) OR count[ts] == 2 and ts in b1∩b2
                        # Let's just compute: after removal, which ts have count 0?
                        temp_counts = counts.copy()
                        temp_counts[ts1] -= 1
                        temp_counts[ts2] -= 1
                        critical_ts = np.where(temp_counts == 0)[0]
                        
                        if len(critical_ts) == 0:
                            # Both blocks are redundant! Remove both
                            counts[ts1] -= 1
                            counts[ts2] -= 1
                            # Remove j first (higher index) to keep i valid
                            blocks.pop(max(i,j))
                            blocks.pop(min(i,j))
                            improved = True
                            found_improvement = True
                            break
                        
                        if len(critical_ts) > 5:  # Too many to cover with one block
                            continue
                        
                        # Find a block that covers all critical_ts
                        # Use the first critical ts as anchor
                        anchor = int(critical_ts[0])
                        candidates = ts_to_block_indices_np[anchor]
                        
                        best_new_block = -1
                        best_score = -1
                        
                        for cand_idx in candidates:
                            cand_idx = int(cand_idx)
                            if cand_idx == b1_idx or cand_idx == b2_idx:
                                continue
                            
                            cand_ts = block_to_ts_indices[cand_idx]
                            # Check if covers all critical_ts
                            if np.all(np.isin(critical_ts, cand_ts)):
                                # Score: prefer blocks that add coverage to ts with low counts
                                score = np.sum(counts[cand_ts] == 1)
                                if score > best_score:
                                    best_score = score
                                    best_new_block = cand_idx
                        
                        if best_new_block != -1:
                            # Perform 2-for-1 swap
                            counts[ts1] -= 1
                            counts[ts2] -= 1
                            counts[block_to_ts_indices[best_new_block]] += 1
                            # Remove the two blocks and add the new one
                            blocks.pop(max(i,j))
                            blocks.pop(min(i,j))
                            blocks.append(best_new_block)
                            improved = True
                            found_improvement = True
                            break
                    
                    if found_improvement:
                        continue
        
        return blocks, counts

    # 4. Main Loop
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Greedy Init
        blocks, counts = greedy_init(rng)
        
        if len(blocks) < best_count:
            best_blocks = list(blocks)
            best_count = len(blocks)
            
        # Local Search
        improved_blocks, _ = local_search(blocks, counts, rng)
        
        if len(improved_blocks) < best_count:
            best_blocks = list(improved_blocks)
            best_count = len(improved_blocks)
            
        # If time is tight, break
        if time.time() > deadline - 25:
             break
             
    # Convert block indices back to lists of points
    if best_blocks is None:
        return []
        
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
