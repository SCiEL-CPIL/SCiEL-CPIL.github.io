import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using heap-based greedy + enhanced local search.
    v=19, k=7, t=4
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            tsubs.append(tsub_to_idx[ts])
        block_tsubs.append(np.array(tsubs, dtype=np.int32))
    
    # Precompute tsub_to_blocks: list of block indices for each t-subset
    tsub_to_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_to_blocks[ts_idx].append(bi)
    
    # Convert to sets for faster membership testing in swap validation
    # Actually, for swap validation we check if ts_idx is in block_tsubs[bi_new]
    # Let's precompute a set version for each block's tsubsets for O(1) lookup
    block_tsub_sets = [set(tsubs.tolist()) for tsubs in block_tsubs]
    
    # --- Heap-based Greedy Construction ---
    def greedy_construct_heap(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # gain[bi] = number of uncovered t-subsets in block bi
        gain = np.zeros(n_blocks, dtype=np.int32)
        gain.fill(len(block_tsubs[0]))  # C(k,t) for all blocks initially
        
        # Heap: (negative_gain, block_index) for max-heap behavior
        # We'll use lazy deletion: push updated gains, skip stale entries
        heap = []
        for bi in range(n_blocks):
            heapq.heappush(heap, (-gain[bi], bi))
        
        n_uncovered = n_tsubsets
        
        while n_uncovered > 0:
            if time.time() > deadline - 80:
                break
                
            # Pop until we find a valid (non-stale) entry
            chosen_idx = None
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if g == gain[bi] and g > 0:
                    chosen_idx = bi
                    break
                # If gain[bi] changed, this entry is stale; discard
                elif g > gain[bi]:
                    # Stale: old gain was higher than current
                    continue
                elif g < gain[bi]:
                    # This shouldn't happen if we always push new gains
                    continue
                else:
                    # g == gain[bi] but gain[bi] == 0, skip
                    continue
            
            if chosen_idx is None:
                break
                
            blocks_used.append(chosen_idx)
            
            # Update uncovered and gains
            tsubs = block_tsubs[chosen_idx]
            newly_covered = tsubs[uncovered[tsubs]]
            
            if len(newly_covered) > 0:
                n_uncovered -= len(newly_covered)
                uncovered[newly_covered] = False
                
                # Decrement gain for all blocks that cover any of the newly_covered t-subsets
                affected_blocks = []
                for ts_idx in newly_covered:
                    affected_blocks.extend(tsub_to_blocks[ts_idx])
                
                if affected_blocks:
                    affected_blocks = np.array(affected_blocks, dtype=np.int32)
                    # Count occurrences to properly decrement
                    # Use np.unique with return_counts for efficiency
                    unique_affected, counts = np.unique(affected_blocks, return_counts=True)
                    gain[unique_affected] -= counts
                    
                    # Push updated gains to heap
                    for bi, new_g in zip(unique_affected, gain[unique_affected]):
                        if new_g > 0:
                            heapq.heappush(heap, (-new_g, bi))
            
            # Mark chosen block as used (gain = 0) to prevent re-selection
            gain[chosen_idx] = 0
            
        return blocks_used
    
    # Get initial solution with multiple seeds, keep the best
    best_initial = None
    for seed in [123, 456, 789, 101, 202]:
        if time.time() > deadline - 75:
            break
        init = greedy_construct_heap(seed)
        if init and (best_initial is None or len(init) < len(best_initial)):
            best_initial = init
    
    initial_blocks = best_initial
    if not initial_blocks:
        return [list(b) for b in all_blocks[:100]]
    
    # --- Enhanced Local Search ---
    blocks = list(initial_blocks)
    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
    
    for bi in blocks:
        for ts_idx in block_tsubs[bi]:
            cover_count[ts_idx] += 1
    
    current_cost = len(blocks)
    best_cost = current_cost
    best_blocks = blocks.copy()
    
    # Precompute for each block, the set of t-subsets it covers (for fast membership)
    # Already have block_tsub_sets
    
    # Helper: get unique t-subsets for a block (those with cover_count == 1)
    def get_unique_tsubs(bi):
        tsubs = block_tsubs[bi]
        return [ts_idx for ts_idx in tsubs if cover_count[ts_idx] == 1]
    
    # SA Parameters
    T = 50.0
    T_min = 0.5
    alpha = 0.9995
    
    rng = random.Random(42)
    
    # Track best swap candidates: for each block, which other blocks could replace it
    # Precompute candidate pools for swap moves
    # For a block bi_old with unique t-subsets U, bi_new must cover all of U
    # We can precompute for each subset of t-subsets, which blocks cover them
    # But that's too expensive. Instead, use random sampling with validation.
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while T > T_min and time.time() < deadline - 10:
        iteration += 1
        
        if time.time() > deadline - 10:
            break
            
        move_type = rng.random()
        
        if move_type < 0.35:
            # Try to remove a redundant block
            if not blocks:
                continue
                
            i = rng.randint(0, len(blocks) - 1)
            bi = blocks[i]
            
            # Check if redundant
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
                    
            if redundant:
                blocks.pop(i)
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] -= 1
                current_cost -= 1
                no_improve_count = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
            else:
                no_improve_count += 1
                
        elif move_type < 0.85:
            # Try a swap: replace bi_old with bi_new
            if not blocks:
                continue
                
            i = rng.randint(0, len(blocks) - 1)
            bi_old = blocks[i]
            
            unique_tsubs = get_unique_tsubs(bi_old)
            
            if not unique_tsubs:
                # Redundant block; try to remove it instead
                blocks.pop(i)
                for ts_idx in block_tsubs[bi_old]:
                    cover_count[ts_idx] -= 1
                current_cost -= 1
                no_improve_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
                continue
            
            # Find candidate blocks that cover all unique_tsubs
            # Strategy: sample random blocks and check
            # Better: for the first unique t-subset, get all blocks covering it,
            # then filter for those covering the rest
            first_ts = unique_tsubs[0]
            candidates = tsub_to_blocks[first_ts]
            
            if not candidates:
                continue
            
            # Shuffle and try up to 20 candidates
            rng.shuffle(candidates)
            swapped = False
            for bi_new in candidates[:20]:
                if bi_new == bi_old:
                    continue
                    
                # Check if bi_new covers all unique_tsubs
                valid = True
                bts_set = block_tsub_sets[bi_new]
                for ts_idx in unique_tsubs:
                    if ts_idx not in bts_set:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Perform swap
                for ts_idx in block_tsubs[bi_old]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_tsubs[bi_new]:
                    cover_count[ts_idx] += 1
                    
                blocks[i] = bi_new
                swapped = True
                break
            
            if not swapped:
                no_improve_count += 1
            else:
                # Swap succeeded; check if it enables removals
                # Try to remove redundant blocks after swap
                for j in range(len(blocks)):
                    if time.time() > deadline - 10:
                        break
                    bj = blocks[j]
                    redundant = True
                    for ts_idx in block_tsubs[bj]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(j)
                        for ts_idx in block_tsubs[bj]:
                            cover_count[ts_idx] -= 1
                        current_cost -= 1
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = blocks.copy()
                        no_improve_count = 0
                        break  # Restart the removal scan
        
        else:
            # Add a random block (only when hot)
            if T < 5.0:
                no_improve_count += 1
                continue
                
            bi_new = rng.randint(0, n_blocks - 1)
            if bi_new in blocks:
                continue
                
            blocks.append(bi_new)
            for ts_idx in block_tsubs[bi_new]:
                cover_count[ts_idx] += 1
            current_cost += 1
            no_improve_count += 1
        
        # Update temperature
        T *= alpha
        
        # If stuck too long, do a small restart: perturb the solution
        if no_improve_count > max_no_improve and T > T_min * 10:
            # Remove 3-5 random non-essential blocks and re-add
            n_remove = min(3, len(blocks) // 10)
            for _ in range(n_remove):
                if not blocks:
                    break
                i = rng.randint(0, len(blocks) - 1)
                bi = blocks[i]
                # Only remove if not critical
                critical = False
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        critical = True
                        break
                if not critical:
                    blocks.pop(i)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    current_cost -= 1
            no_improve_count = 0
        
        # Early termination if we've reached a very good solution
        if best_cost <= 135:
            break
    
    # Final Polish: aggressive reduction
    final_blocks = best_blocks.copy()
    cover_count_final = np.zeros(n_tsubsets, dtype=np.int32)
    for bi in final_blocks:
        for ts_idx in block_tsubs[bi]:
            cover_count_final[ts_idx] += 1
    
    # Multiple passes of removal
    for _ in range(3):
        i = 0
        while i < len(final_blocks):
            bi = final_blocks[i]
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count_final[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                final_blocks.pop(i)
                for ts_idx in block_tsubs[bi]:
                    cover_count_final[ts_idx] -= 1
            else:
                i += 1
    
    # Try swap-based reduction on final
    for _ in range(500):
        if time.time() > deadline - 2:
            break
        if not final_blocks:
            break
        i = rng.randint(0, len(final_blocks) - 1)
        bi_old = final_blocks[i]
        unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi_old] if cover_count_final[ts_idx] == 1]
        if not unique_tsubs:
            final_blocks.pop(i)
            for ts_idx in block_tsubs[bi_old]:
                cover_count_final[ts_idx] -= 1
            continue
        first_ts = unique_tsubs[0]
        candidates = tsub_to_blocks[first_ts]
        if not candidates:
            continue
        rng.shuffle(candidates)
        swapped = False
        for bi_new in candidates[:10]:
            if bi_new == bi_old:
                continue
            bts_set = block_tsub_sets[bi_new]
            valid = all(ts_idx in bts_set for ts_idx in unique_tsubs)
            if not valid:
                continue
            for ts_idx in block_tsubs[bi_old]:
                cover_count_final[ts_idx] -= 1
            for ts_idx in block_tsubs[bi_new]:
                cover_count_final[ts_idx] += 1
            final_blocks[i] = bi_new
            swapped = True
            break
        if swapped:
            # Try removal again
            j = 0
            while j < len(final_blocks):
                bj = final_blocks[j]
                redundant = all(cover_count_final[ts_idx] > 1 for ts_idx in block_tsubs[bj])
                if redundant:
                    final_blocks.pop(j)
                    for ts_idx in block_tsubs[bj]:
                        cover_count_final[ts_idx] -= 1
                else:
                    j += 1
    
    if len(final_blocks) < best_cost:
        best_blocks = final_blocks
        best_cost = len(final_blocks)
    
    return [list(all_blocks[bi]) for bi in best_blocks]
