import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute triplet IDs for each block
    # block_triplets[block_idx] = list of triplet_ids
    block_triplets = []
    # Also precompute which blocks contain which triplet
    # triplet_to_blocks[triplet_id] = list of block_indices
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    
    for b_idx, block in enumerate(all_blocks):
        t_ids = []
        for trip in itertools.combinations(block, t):
            tid = triplet_to_id[trip]
            t_ids.append(tid)
            triplet_to_blocks[tid].append(b_idx)
        block_triplets.append(t_ids)
        
    def get_block_mask_from_ids(t_ids):
        # Returns a set of triplet ids
        return set(t_ids)

    # Greedy Construction
    def greedy_construct():
        counts = np.zeros(n_triplets, dtype=np.int32)
        blocks = []
        
        # We need to efficiently find the best block.
        # Strategy: Pick a random uncovered triplet, find best block containing it.
        # "Best" defined by gain (number of newly covered triplets).
        
        uncovered_count = n_triplets
        # To speed up finding uncovered triplet, we can just pick a random index and check, 
        # or maintain a list of uncovered. 
        # Given n_triplets is ~2000, a linear scan or random check is fine.
        
        while uncovered_count > 0:
            # Find an uncovered triplet
            # Randomly sample indices until we find an uncovered one
            # Or just iterate. 2000 is small.
            target_tid = -1
            # Use a random start to break symmetry
            start_idx = random.randint(0, n_triplets - 1)
            for i in range(n_triplets):
                idx = (start_idx + i) % n_triplets
                if counts[idx] == 0:
                    target_tid = idx
                    break
            
            if target_tid == -1:
                break # Should not happen if uncovered_count > 0
                
            # Find best block containing target_tid
            best_block_idx = -1
            best_gain = -1
            
            # Iterate through all blocks containing target_tid
            for b_idx in triplet_to_blocks[target_tid]:
                gain = 0
                # Calculate gain: number of triplets in this block that are currently uncovered
                for tid in block_triplets[b_idx]:
                    if counts[tid] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            # Add the best block
            if best_block_idx != -1:
                blocks.append(best_block_idx)
                # Update counts
                for tid in block_triplets[best_block_idx]:
                    if counts[tid] == 0:
                        uncovered_count -= 1
                    counts[tid] += 1
            else:
                break # No block found? Should not happen.
                
        return [all_blocks[i] for i in blocks]

    # Local Search Optimizer
    def optimize(block_indices):
        # block_indices: list of indices into all_blocks
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        # Initialize counts
        for b_idx in block_indices:
            for tid in block_triplets[b_idx]:
                counts[tid] += 1
                
        improved = True
        no_improve_iter = 0
        max_no_improve = 500 # Break after 500 iterations without improvement to allow restarts
        
        while improved:
            if time.time() > deadline - 10:
                break
                
            improved = False
            n = len(block_indices)
            
            # 1. Remove Redundant Blocks
            # Check if any block can be removed
            # A block is redundant if all its triplets have count > 1
            i = 0
            while i < n:
                b_idx = block_indices[i]
                redundant = True
                for tid in block_triplets[b_idx]:
                    if counts[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for tid in block_triplets[b_idx]:
                        counts[tid] -= 1
                    block_indices.pop(i)
                    n -= 1
                    improved = True
                    # Do not increment i, check the new block at index i
                else:
                    i += 1
                    
            if improved:
                no_improve_iter = 0
                continue # Restart loop to catch new redundancies or swaps
            
            # 2. Try 2-for-1 Swaps
            # Randomly shuffle pairs to avoid bias
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline - 10:
                    break
                    
                b1_idx = block_indices[i]
                b2_idx = block_indices[j]
                
                # Find triplets uniquely covered by b1 or b2
                # unique_tids: triplets with count 1 that are in b1 or b2
                unique_tids = set()
                
                # Check b1
                for tid in block_triplets[b1_idx]:
                    if counts[tid] == 1:
                        unique_tids.add(tid)
                        
                # Check b2
                for tid in block_triplets[b2_idx]:
                    if counts[tid] == 1:
                        unique_tids.add(tid)
                        
                if not unique_tids:
                    # Both are redundant (should have been caught by step 1)
                    # Remove both
                    for tid in block_triplets[b2_idx]:
                        counts[tid] -= 1
                    for tid in block_triplets[b1_idx]:
                        counts[tid] -= 1
                    # Remove j first (higher index)
                    block_indices.pop(j)
                    block_indices.pop(i)
                    improved = True
                    break # Restart loop
                    
                # Find candidate blocks that cover all unique_tids
                # A candidate block must contain all triplets in unique_tids
                # This means it must contain all points in the union of these triplets
                # But more efficiently: it must be in the intersection of triplet_to_blocks[tid] for all tid in unique_tids
                
                # Start with the list of blocks for the first triplet
                first_tid = next(iter(unique_tids))
                candidates = triplet_to_blocks[first_tid]
                
                # Intersect with lists for other triplets
                # To speed up, sort unique_tids by length of their block lists?
                # For now, just intersect. The lists are ~6000 long.
                # If unique_tids has size m, we do m-1 intersections.
                # We can optimize by picking the triplet with the smallest list first.
                
                # Let's refine: pick tid with min len(triplet_to_blocks[tid])
                sorted_tids = sorted(unique_tids, key=lambda tid: len(triplet_to_blocks[tid]))
                
                candidate_set = set(triplet_to_blocks[sorted_tids[0]])
                
                for tid in sorted_tids[1:]:
                    candidate_set.intersection_update(triplet_to_blocks[tid])
                    if not candidate_set:
                        break
                        
                if not candidate_set:
                    continue
                    
                # Find a candidate that is not b1 or b2
                replacement_found = False
                for cand_idx in candidate_set:
                    if cand_idx == b1_idx or cand_idx == b2_idx:
                        continue
                        
                    # Since cand_idx is in the intersection of lists for all unique_tids,
                    # it covers all unique_tids.
                    # We just need to perform the swap.
                    
                    # Remove b1, b2
                    for tid in block_triplets[b2_idx]:
                        counts[tid] -= 1
                    for tid in block_triplets[b1_idx]:
                        counts[tid] -= 1
                    block_indices.pop(j)
                    block_indices.pop(i)
                    
                    # Add candidate
                    block_indices.append(cand_idx)
                    for tid in block_triplets[cand_idx]:
                        counts[tid] += 1
                        
                    improved = True
                    replacement_found = True
                    break
                    
                if improved:
                    break # Restart loop
                    
            if improved:
                no_improve_iter = 0
                continue
            
            # 3. Try 1-for-1 Swaps
            # Randomly pick a block
            if n > 0:
                i = random.randint(0, n - 1)
                b1_idx = block_indices[i]
                
                unique_tids = set()
                for tid in block_triplets[b1_idx]:
                    if counts[tid] == 1:
                        unique_tids.add(tid)
                        
                if not unique_tids:
                    continue # Redundant, handled by step 1 (or missed?)
                
                # Find replacement
                sorted_tids = sorted(unique_tids, key=lambda tid: len(triplet_to_blocks[tid]))
                candidate_set = set(triplet_to_blocks[sorted_tids[0]])
                
                for tid in sorted_tids[1:]:
                    candidate_set.intersection_update(triplet_to_blocks[tid])
                    if not candidate_set:
                        break
                        
                if not candidate_set:
                    continue
                
                for cand_idx in candidate_set:
                    if cand_idx == b1_idx:
                        continue
                        
                    # Perform swap
                    for tid in block_triplets[b1_idx]:
                        counts[tid] -= 1
                    block_indices.pop(i)
                    
                    block_indices.append(cand_idx)
                    for tid in block_triplets[cand_idx]:
                        counts[tid] += 1
                        
                    improved = True
                    break
                    
            if not improved:
                no_improve_iter += 1
                if no_improve_iter > max_no_improve:
                    break
                    
        return [all_blocks[i] for i in block_indices]

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple times with different random seeds
    # We can run more seeds if the greedy is fast.
    # Greedy is ~15 million ops. Optimize is fast due to precomputed intersections.
    
    for seed in range(50): # Increased seeds
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        
        # Greedy init
        init_block_indices = []
        counts = np.zeros(n_triplets, dtype=np.int32)
        uncovered_count = n_triplets
        
        while uncovered_count > 0:
            target_tid = -1
            start_idx = random.randint(0, n_triplets - 1)
            for i in range(n_triplets):
                idx = (start_idx + i) % n_triplets
                if counts[idx] == 0:
                    target_tid = idx
                    break
                    
            if target_tid == -1:
                break
                
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in triplet_to_blocks[target_tid]:
                gain = 0
                for tid in block_triplets[b_idx]:
                    if counts[tid] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            if best_block_idx != -1:
                init_block_indices.append(best_block_idx)
                for tid in block_triplets[best_block_idx]:
                    if counts[tid] == 0:
                        uncovered_count -= 1
                    counts[tid] += 1
            else:
                break
        
        # Optimize
        opt_block_indices = optimize(init_block_indices)
        opt_blocks = [all_blocks[i] for i in opt_block_indices]
        
        # Verify validity
        counts_check = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts_check[triplet_to_id[triplet]] += 1
                
        if np.min(counts_check) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
                
    if best_blocks is None:
        # Fallback: Just return the last greedy result if no valid optimized one found
        # This shouldn't happen, but just in case
        init_block_indices = []
        counts = np.zeros(n_triplets, dtype=np.int32)
        uncovered_count = n_triplets
        while uncovered_count > 0:
            target_tid = -1
            for i in range(n_triplets):
                if counts[i] == 0:
                    target_tid = i
                    break
            if target_tid == -1: break
            best_block_idx = -1
            best_gain = -1
            for b_idx in triplet_to_blocks[target_tid]:
                gain = 0
                for tid in block_triplets[b_idx]:
                    if counts[tid] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            if best_block_idx != -1:
                init_block_indices.append(best_block_idx)
                for tid in block_triplets[best_block_idx]:
                    if counts[tid] == 0:
                        uncovered_count -= 1
                    counts[tid] += 1
            else: break
        best_blocks = [all_blocks[i] for i in init_block_indices]
        
    return best_blocks

if __name__ == "__main__":
    pass
