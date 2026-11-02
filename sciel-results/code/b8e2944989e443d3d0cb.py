import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 150s limit, stop at 140s to allow cleanup
    deadline = start_time + 140.0

    # 1. Precomputation
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(all_points, k))
    num_k_blocks = len(all_k_blocks)
    
    # block_to_ts: list of lists of t-subset indices covered by block i
    block_to_ts = []
    # ts_to_blocks: list of sets of block indices covering t-subset i
    ts_to_blocks = [set() for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        ts_indices = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_indices.append(idx)
            ts_to_blocks[idx].add(b_idx)
        block_to_ts.append(ts_indices)
        
    # Convert ts_to_blocks to lists for faster iteration if needed, or keep as sets
    # For gain updates, we need to iterate over blocks covering a ts.
    ts_to_blocks_lists = [list(s) for s in ts_to_blocks]
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(range(num_t_subsets))
        # gain[b] = number of uncovered ts covered by block b
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        selected_blocks = []
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
                
            # Find block with max gain among unselected
            # To handle ties randomly, we can pick from top few or just max
            # np.argmax is fast.
            max_gain_val = np.max(gain[~is_selected])
            if max_gain_val <= 0:
                break
            
            # Find all indices with max gain
            candidates = np.where((gain == max_gain_val) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Pick one randomly from candidates to break symmetry
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            selected_blocks.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1 # Mark as selected
            
            # Update uncovered and gains
            b_ts = block_to_ts[b_idx]
            newly_covered = []
            for ts_idx in b_ts:
                if ts_idx in uncovered:
                    newly_covered.append(ts_idx)
                    uncovered.discard(ts_idx)
            
            # Update gains for blocks covering newly covered ts's
            if newly_covered:
                # We need to decrement gain for all blocks covering these ts's
                # To avoid double counting if a block covers multiple newly covered ts's,
                # we can use a temporary dict or just iterate and check if ts was in uncovered (it was).
                # Actually, gain[b] is count of ts in block_to_ts[b] that are in uncovered.
                # If we remove ts from uncovered, gain[b] should decrease by 1 for each such ts.
                
                # Efficient way:
                # For each ts in newly_covered:
                #   For each block b in ts_to_blocks[ts]:
                #     gain[b] -= 1
                
                # This is O(sum(|ts_to_blocks[ts]| for ts in newly_covered))
                # |ts_to_blocks[ts]| is roughly constant (~2000-3000 for v=19,k=8,t=5?)
                # C(19,5) = 11628. C(19,8) = 75582.
                # Each ts is in C(19-5, 8-5) = C(14,3) = 364 blocks? No.
                # Number of blocks containing a specific 5-subset:
                # We need to choose 8-5=3 more points from 19-5=14.
                # C(14,3) = 364.
                # So each ts is in ~364 blocks.
                # newly_covered size is up to 56.
                # So ~56 * 364 = 20k updates per step.
                # 300 steps * 20k = 6M updates. Very fast.
                
                for ts_idx in newly_covered:
                    for b in ts_to_blocks_lists[ts_idx]:
                        gain[b] -= 1
        
        return selected_blocks

    def delete_blocks(blocks):
        if not blocks:
            return blocks, np.zeros(num_t_subsets, dtype=np.int32)
            
        # Initialize cover counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        
        for b in blocks:
            # Find index of b in all_k_blocks
            # This is a bottleneck if done naively. 
            # Pre-compute a map from tuple to index? 
            # all_k_blocks is a list of tuples.
            # We can create a dict: block_tuple_to_idx
            pass
        
        # Let's create the map outside or pass it in
        # We'll assume we have a way to get indices.
        # For speed, let's pre-compute `block_tuple_to_idx` in construct scope.
        
        # Re-doing setup for deletion to be self-contained and fast
        # Actually, we can just use the indices if we track them.
        # But greedy returns lists.
        
        # Let's create a helper to get index
        # Since all_k_blocks is a list of tuples, and blocks are lists, 
        # we can convert to tuple and lookup.
        
        # To make this fast, we should have pre-computed:
        # block_to_idx = {b: i for i, b in enumerate(all_k_blocks)}
        
        indices = [block_to_idx[b] for b in blocks]
        
        for idx in indices:
            ts_arr = block_to_ts[idx]
            for ts_idx in ts_arr:
                cover_count[ts_idx] += 1
        
        removed_any = True
        while removed_any:
            removed_any = False
            # Iterate in random order to break symmetry
            idx_list = list(range(len(blocks)))
            random.shuffle(idx_list)
            
            for i in idx_list:
                if time.time() > deadline - 10.0:
                    return [blocks[j] for j in range(len(blocks)) if j != -1], cover_count
                
                b_idx = indices[i]
                ts_arr = block_to_ts[b_idx]
                
                # Check if any ts has count == 1
                # If all counts > 1, we can remove
                can_remove = True
                for ts_idx in ts_arr:
                    if cover_count[ts_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for ts_idx in ts_arr:
                        cover_count[ts_idx] -= 1
                    blocks[i] = None
                    indices[i] = -1
                    removed_any = True
                    if time.time() > deadline - 10.0:
                        break
            
            if removed_any:
                # Compact lists
                new_blocks = []
                new_indices = []
                for b, idx in zip(blocks, indices):
                    if b is not None:
                        new_blocks.append(b)
                        new_indices.append(idx)
                blocks = new_blocks
                indices = new_indices
        
        return blocks, cover_count

    def local_search(blocks, cover_count):
        if not blocks:
            return blocks, cover_count
            
        # Reconstruct indices
        indices = [block_to_idx[b] for b in blocks]
        
        improved = True
        max_iters = 5000 # Cap iterations
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 15.0:
                break
                
            # Pick a random block to try to swap/remove
            i = random.randint(0, len(blocks) - 1)
            b_idx = indices[i]
            ts_arr = block_to_ts[b_idx]
            
            # Find unique ts's
            unique_ts = []
            for ts_idx in ts_arr:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            
            if not unique_ts:
                # Can remove directly
                for ts_idx in ts_arr:
                    cover_count[ts_idx] -= 1
                blocks.pop(i)
                indices.pop(i)
                improved = True
                continue
            
            # Need to find a replacement block that covers all unique_ts
            # Candidate blocks must be in intersection of ts_to_blocks[ts] for ts in unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            # Exclude current block and already selected blocks?
            # If we swap with an already selected block, we are just reordering or doing nothing.
            # We want a block NOT in the current solution to potentially enable new deletions.
            # Or a block in the solution that allows removing *another* block?
            # Standard 1-swap: remove b_i, add b_j.
            # If b_j is already in solution, we just removed b_i. But b_i had unique ts's, so removing it breaks coverage unless b_j covers them.
            # If b_j is already in solution, it already covers its ts's.
            # If b_j covers unique_ts of b_i, and b_j is already in solution, then those ts's were covered by b_j too?
            # No, cover_count is global. If cover_count[ts] == 1 for ts in unique_ts, then NO other block covers them.
            # So b_j (if already in solution) cannot cover them, because if it did, cover_count would be >= 2.
            # Therefore, b_j MUST be a block NOT in the current solution.
            
            current_block_indices = set(indices)
            valid_candidates = [c for c in candidates if c not in current_block_indices]
            
            if not valid_candidates:
                continue
            
            # Pick a candidate. Heuristic: pick one that covers the most "vulnerable" (low count) other ts's?
            # Or just random?
            # Let's pick the one that maximizes the number of ts's with count == 1 it covers (excluding unique_ts which are forced).
            # Actually, simpler: pick one that minimizes the "cost" or just random.
            # Let's try picking one that covers the most additional ts's with count == 1.
            
            best_cand = None
            best_score = -1
            
            # Limit candidates to avoid O(N) scan if too many
            if len(valid_candidates) > 20:
                valid_candidates = random.sample(valid_candidates, 20)
                
            for c_idx in valid_candidates:
                c_ts = block_to_ts[c_idx]
                score = 0
                for ts_idx in c_ts:
                    if cover_count[ts_idx] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c_idx
            
            if best_cand is None:
                continue
            
            # Perform swap
            # 1. Remove b_i
            for ts_idx in ts_arr:
                cover_count[ts_idx] -= 1
            
            # 2. Add best_cand
            c_ts_arr = block_to_ts[best_cand]
            for ts_idx in c_ts_arr:
                cover_count[ts_idx] += 1
            
            # Check if this enables deletions
            # Run a quick deletion pass
            # To keep it simple, just try to delete any block that is now redundant.
            # If size doesn't decrease, revert.
            
            # We need to check if ANY block can be deleted.
            # Since we just swapped, maybe b_i's unique ts's are now covered by best_cand (they are, by construction).
            # But maybe best_cand made another block redundant?
            
            # Let's try to delete the block we just added? No, it covers unique ts's.
            # Let's try to delete other blocks.
            
            # Efficient check: 
            # For each block j != i (in the new list), check if it has unique ts's.
            # If not, remove it.
            
            # We modified cover_count.
            # Let's iterate through blocks (excluding the one we just added at position i)
            # Actually, we replaced b_i with best_cand at index i.
            
            blocks[i] = list(all_k_blocks[best_cand])
            indices[i] = best_cand
            
            # Try to remove redundant blocks
            any_removed = False
            for j in range(len(blocks)):
                if j == i:
                    continue
                b_j_idx = indices[j]
                ts_j = block_to_ts[b_j_idx]
                can_remove = True
                for ts_idx in ts_j:
                    if cover_count[ts_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts_idx in ts_j:
                        cover_count[ts_idx] -= 1
                    blocks.pop(j)
                    indices.pop(j)
                    any_removed = True
                    # Adjust index i if j < i
                    if j < i:
                        i -= 1
                    improved = True
                    # Restart loop to pick a new random block
                    break
            
            if not any_removed:
                # Revert swap
                for ts_idx in c_ts_arr:
                    cover_count[ts_idx] -= 1
                for ts_idx in ts_arr:
                    cover_count[ts_idx] += 1
                blocks[i] = list(all_k_blocks[b_idx])
                indices[i] = b_idx
        
        return blocks, cover_count

    # Pre-compute block to index map for O(1) lookup
    block_to_idx = {b: i for i, b in enumerate(all_k_blocks)}
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple instances
    num_runs = 10
    for i in range(num_runs):
        if time.time() > deadline - 30.0:
            break
            
        random.seed(42 + i)
        blocks = greedy_cover(42 + i)
        
        if not blocks:
            continue
            
        blocks, cover_count = delete_blocks(blocks)
        
        # Local search
        if time.time() < deadline - 20.0:
            blocks, cover_count = local_search(blocks, cover_count)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we get a very good result, we can stop early?
        # C(19,8,5) best known is 335.
        # If we get < 335, it's a breakthrough.
        # If we get close, maybe continue.
        
    if best_blocks is None:
        # Fallback
        blocks = greedy_cover(42)
        blocks, _ = delete_blocks(blocks)
        return blocks
        
    return best_blocks

# Helper function for combinatorics
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
