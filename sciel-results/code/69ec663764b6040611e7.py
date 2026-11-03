import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Fallback for non-target parameters
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            best_count = -1
            for block_tuple in itertools.combinations(range(v), k):
                cnt = 0
                for tsub in itertools.combinations(block_tuple, t):
                    if tsub in uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_block = list(block_tuple)
            if best_block is None:
                break
            blocks.append(best_block)
            for tsub in itertools.combinations(best_block, t):
                uncovered.discard(tsub)
            if time.time() > deadline:
                break
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute all blocks and their t-subset IDs
    all_blocks = []
    block_tsubs = [] # list of lists of tsub IDs
    block_tsub_sets = [] # list of sets of tsub IDs for fast intersection
    
    for block_tuple in itertools.combinations(points, k):
        tsubs_in_block = [tsub_to_id[tsub] for tsub in itertools.combinations(block_tuple, t)]
        all_blocks.append(block_tuple)
        block_tsubs.append(tsubs_in_block)
        block_tsub_sets.append(set(tsubs_in_block))
        
    num_blocks_total = len(all_blocks)
    
    # --- Phase 1: Greedy Init ---
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    active_blocks = []
    active_set = set()
    uncovered_mask = np.ones(num_tsubs, dtype=bool)
    
    while np.any(uncovered_mask):
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in range(num_blocks_total):
            tsubs_b = block_tsubs[b_idx]
            gain = 0
            for tid in tsubs_b:
                if uncovered_mask[tid]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1 or best_gain == 0:
            break
            
        active_blocks.append(best_block_idx)
        active_set.add(best_block_idx)
        
        for tid in block_tsubs[best_block_idx]:
            cover_count[tid] += 1
            if cover_count[tid] == 1:
                uncovered_mask[tid] = False
        
        if time.time() > deadline - 30:
            break
            
    if np.any(uncovered_mask):
        while np.any(uncovered_mask):
            best_block_idx = -1
            best_gain = -1
            for b_idx in range(num_blocks_total):
                if b_idx in active_set:
                    continue
                tsubs_b = block_tsubs[b_idx]
                gain = 0
                for tid in tsubs_b:
                    if uncovered_mask[tid]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            if best_block_idx == -1:
                break
            active_blocks.append(best_block_idx)
            active_set.add(best_block_idx)
            for tid in block_tsubs[best_block_idx]:
                cover_count[tid] += 1
                if cover_count[tid] == 1:
                    uncovered_mask[tid] = False
            if time.time() > deadline - 10:
                break

    # --- Phase 2: Local Search ---
    current_size = len(active_blocks)
    best_size = current_size
    best_solution = list(active_blocks)
    
    rng = random.Random(42)
    ls_deadline = deadline - 5
    
    # Helper to try deletions
    def do_deletions():
        nonlocal active_blocks, active_set, cover_count, current_size, best_size, best_solution
        changed = True
        while changed and time.time() < ls_deadline:
            changed = False
            for i in range(len(active_blocks) - 1, -1, -1):
                b_idx = active_blocks[i]
                removable = True
                for tid in block_tsubs[b_idx]:
                    if cover_count[tid] == 1:
                        removable = False
                        break
                if removable:
                    active_blocks.pop(i)
                    active_set.remove(b_idx)
                    for tid in block_tsubs[b_idx]:
                        cover_count[tid] -= 1
                    current_size -= 1
                    changed = True
                    if current_size < best_size:
                        best_size = current_size
                        best_solution = list(active_blocks)
                    break # Restart after deletion

    do_deletions()
    
    # Swap loop
    swaps_done = 0
    max_swaps = 10000
    
    while time.time() < ls_deadline and swaps_done < max_swaps:
        # Try deletions first
        deleted = False
        for i in range(len(active_blocks) - 1, -1, -1):
            b_idx = active_blocks[i]
            removable = True
            for tid in block_tsubs[b_idx]:
                if cover_count[tid] == 1:
                    removable = False
                    break
            if removable:
                active_blocks.pop(i)
                active_set.remove(b_idx)
                for tid in block_tsubs[b_idx]:
                    cover_count[tid] -= 1
                current_size -= 1
                deleted = True
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(active_blocks)
                break
        
        if deleted:
            continue
        
        # Try 1-swap
        b_out_idx = rng.randint(0, len(active_blocks) - 1)
        b_out = active_blocks[b_out_idx]
        critical_tids = [tid for tid in block_tsubs[b_out] if cover_count[tid] == 1]
        
        valid_swap_found = False
        for _ in range(5): # Try 5 random candidates
            b_in = rng.randint(0, num_blocks_total - 1)
            if b_in in active_set:
                continue
            
            # Check if b_in covers all critical_tids
            if all(tid in block_tsub_sets[b_in] for tid in critical_tids):
                # Perform swap
                active_blocks[b_out_idx] = b_in
                active_set.remove(b_out)
                active_set.add(b_in)
                
                # Update counts
                for tid in block_tsubs[b_out]:
                    cover_count[tid] -= 1
                for tid in block_tsubs[b_in]:
                    cover_count[tid] += 1
                    
                valid_swap_found = True
                break
        
        swaps_done += 1
        # If no swap found, we might be stuck. 
        # Could try 2-swap, but it's expensive.
        # For now, we just continue, hoping a random b_out later works.

    return [list(all_blocks[i]) for i in best_solution]
