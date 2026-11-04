import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 140  # Safety margin
    
    # 1. Generate all blocks and subsets
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute which subsets each block covers
    # block_subsets[i] is a list of subset indices
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # Precompute which blocks cover each subset
    # blocks_covering_subset[j] is a list of block indices
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    # 2. Greedy Construction
    cover_count = np.zeros(N_subsets, dtype=np.int32)
    # potential[i] = number of uncovered subsets covered by block i
    potential = np.array([len(s) for s in block_subsets], dtype=np.int32)
    
    selected_blocks = []
    
    while np.any(cover_count == 0):
        if time.time() - start_time > timeout:
            break
            
        # Find block with max potential
        # If tie, pick one randomly or first? First is deterministic.
        best_idx = np.argmax(potential)
        if potential[best_idx] == 0:
            break # Should not happen if cover_count has 0s, unless bug
            
        selected_blocks.append(best_idx)
        
        # Update cover_count and potentials
        for s in block_subsets[best_idx]:
            old_count = cover_count[s]
            cover_count[s] += 1
            if old_count == 0:
                # This subset was uncovered, now covered.
                # Decrease potential for all blocks that cover this subset
                for b in blocks_covering_subset[s]:
                    potential[b] -= 1
                    
    # 3. Local Search: Redundancy Removal
    # We have a valid covering (assuming we didn't time out mid-way, 
    # but if we did, we might have an invalid one. 
    # The task requires a valid covering. 
    # If we time out in the greedy phase, we must ensure validity or return best effort?
    # "Valid when every t-subset ... lies in at least one block".
    # If we break early, it's invalid. 
    # So we must ensure we finish the greedy phase. 
    # 11628 subsets / 56 per block ~ 208 blocks min. 
    # Greedy might take ~300-400 steps. 
    # Each step updates ~56 subsets. 
    # Each subset update touches ~364 blocks. 
    # 56 * 364 ~ 20k operations per step. 
    # 400 steps * 20k = 8 million operations. 
    # Python can do this in < 1 second. 
    # So timeout is unlikely in the greedy phase.
    
    # Now, try to remove redundant blocks.
    # A block is redundant if for every subset s in it, cover_count[s] > 1.
    
    # We need to efficiently check redundancy.
    # Let's maintain a list of selected block indices.
    
    # To speed up redundancy check, we can just iterate.
    # Number of blocks ~ 300. Subsets per block ~ 56.
    # 300 * 56 = 16800 checks. Very fast.
    
    changed = True
    while changed and (time.time() - start_time < timeout):
        changed = False
        # Iterate through selected blocks in reverse? Or random order?
        # Random order might help escape local minima if we were doing swaps, 
        # but for pure removal, order doesn't matter much for the final set of non-redundant blocks.
        for i in range(len(selected_blocks) - 1, -1, -1):
            b_idx = selected_blocks[i]
            is_redundant = True
            for s in block_subsets[b_idx]:
                if cover_count[s] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block
                for s in block_subsets[b_idx]:
                    cover_count[s] -= 1
                selected_blocks.pop(i)
                changed = True
                
    # 4. Local Search: Block Replacement (Swap)
    # Try to replace a block with another that allows removal of more blocks?
    # Or just try to find a smaller set.
    # Given time constraints, simple redundancy removal is often sufficient to get close to optimal.
    # However, greedy + removal might leave a "core" that is not minimal.
    
    # Let's try a simple randomized local search:
    # Pick a random block from selected, remove it.
    # If valid, keep.
    # If invalid, try to fix it by adding one block?
    # This is complex.
    
    # Alternative: Restart with different random seeds?
    # The greedy step is deterministic (argmax). 
    # We can randomize the tie-breaking in argmax.
    
    # Let's implement a few restarts with randomized tie-breaking.
    
    best_blocks = selected_blocks
    best_len = len(selected_blocks)
    
    # We have used some time. Let's see if we can do a few restarts.
    # Current time usage is likely < 5s.
    # We have 135s left.
    
    # Let's do a loop of restarts.
    # In each restart, we randomize the selection among ties.
    
    # To allow randomization, we need to modify the greedy loop.
    # Instead of np.argmax, we find all max values and pick one randomly.
    
    # But re-running the whole greedy from scratch is cheap.
    
    # Let's restructure:
    
    def run_greedy(randomize_ties=False):
        cover_count_local = np.zeros(N_subsets, dtype=np.int32)
        potential_local = np.array([len(s) for s in block_subsets], dtype=np.int32)
        selected_local = []
        
        while np.any(cover_count_local == 0):
            if time.time() - start_time > timeout:
                return None # Timeout
                
            max_pot = np.max(potential_local)
            if max_pot == 0:
                break
                
            if randomize_ties:
                candidates = np.where(potential_local == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates)-1)]
            else:
                best_idx = np.argmax(potential_local)
                
            selected_local.append(best_idx)
            
            for s in block_subsets[best_idx]:
                old_count = cover_count_local[s]
                cover_count_local[s] += 1
                if old_count == 0:
                    for b in blocks_covering_subset[s]:
                        potential_local[b] -= 1
                        
        # Redundancy removal
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_local) - 1, -1, -1):
                b_idx = selected_local[i]
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count_local[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in block_subsets[b_idx]:
                        cover_count_local[s] -= 1
                    selected_local.pop(i)
                    changed = True
        return selected_local

    # We already ran the first one (deterministic).
    # Let's run a few randomized ones.
    
    # Estimate time per run.
    # If first run took T, we can do (timeout - T) / T runs.
    
    # For safety, let's just do a fixed number of extra runs, say 10-20, if time permits.
    
    # Actually, the first run is the baseline.
    # Let's check time.
    
    t_elapsed = time.time() - start_time
    if t_elapsed < timeout - 5:
        # Do some randomized restarts
        # Limit to e.g. 20 restarts or until time runs out
        for i in range(20):
            if time.time() - start_time > timeout - 2:
                break
            res = run_greedy(randomize_ties=True)
            if res is not None and len(res) < best_len:
                best_len = len(res)
                best_blocks = res
                
    # Convert indices back to blocks (lists of ints)
    # The problem asks for a list of blocks, each a list of k distinct integers.
    # all_blocks is a list of tuples.
    
    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    
    return final_blocks
