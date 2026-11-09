import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to its index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs_in_block = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs_in_block)
        
    # For fast validity checks, we need cover_count
    # Greedy Construction
    cover_count = [0] * n_tsubs
    selected_indices = []
    remaining = set(range(n_blocks))
    
    # Initial Greedy
    # To speed up, we can use a simple heuristic: pick block covering most uncovered tsubs
    # We maintain a list of uncovered tsubs? Or just check gain.
    # Since all blocks have same initial gain, we can just pick randomly or sequentially?
    # No, we need to update gains.
    
    # Optimized Greedy with NumPy or just efficient lists
    # Let's stick to a standard greedy but implemented carefully.
    
    # Actually, a random start + local search is often better than pure greedy.
    # Let's generate a random set of blocks that covers everything? 
    # That's hard.
    
    # Let's do Greedy first.
    # Gain for each block: number of uncovered tsubs it covers.
    # Initially all are C(k,t).
    
    # To avoid O(N^2) in greedy, we can use a priority queue or just scan.
    # With N=50388, scanning is 50k ops. 136 blocks * 50k = 6.8M ops. Fast enough.
    
    block_gain = [len(ts) for ts in block_tsubs]
    
    while True:
        if all(c > 0 for c in cover_count):
            break
        
        best_block = -1
        best_gain = -1
        # Random tie-breaking might help?
        for b in remaining:
            if block_gain[b] > best_gain:
                best_gain = block_gain[b]
                best_block = b
        
        if best_block == -1 or best_gain <= 0:
            break
        
        selected_indices.append(best_block)
        remaining.discard(best_block)
        
        for s_idx in block_tsubs[best_block]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] = 1
                # Update gains for other blocks
                # This is the expensive part.
                # We need to decrement gain for all blocks covering s_idx that are still remaining.
                # We can precompute tsub_to_blocks
                pass # We didn't precompute tsub_to_blocks in the new block yet.
                
    # Let's re-implement the greedy with precomputed tsub_to_blocks for speed
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    cover_count = [0] * n_tsubs
    selected_indices = []
    remaining = set(range(n_blocks))
    block_gain = [len(ts) for ts in block_tsubs]
    
    while True:
        if all(c > 0 for c in cover_count):
            break
        
        best_block = -1
        best_gain = -1
        for b in remaining:
            if block_gain[b] > best_gain:
                best_gain = block_gain[b]
                best_block = b
        
        if best_block == -1 or best_gain <= 0:
            break
        
        selected_indices.append(best_block)
        remaining.discard(best_block)
        
        for s_idx in block_tsubs[best_block]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] = 1
                for b in tsub_to_blocks[s_idx]:
                    if b in remaining:
                        block_gain[b] -= 1
                        
    current_set = set(selected_indices)
    best_set = set(current_set)
    
    # Local Search: Simulated Annealing
    # State: set of block indices
    # Move: Remove redundant, or Swap
    # We maintain cover_count for the current set
    
    # Rebuild cover_count for current_set
    cover_count = [0] * n_tsubs
    for b in current_set:
        for s_idx in block_tsubs[b]:
            cover_count[s_idx] += 1
            
    T = 1000.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if T < 1.0:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a random block
            if not current_set:
                continue
            b_remove = random.choice(list(current_set))
            
            # Check if valid
            valid = True
            for s_idx in block_tsubs[b_remove]:
                if cover_count[s_idx] == 1:
                    valid = False
                    break
            
            if valid:
                # Apply removal
                for s_idx in block_tsubs[b_remove]:
                    cover_count[s_idx] -= 1
                current_set.discard(b_remove)
                
                if len(current_set) < len(best_set):
                    best_set = set(current_set)
                T = cooling * T
            else:
                T = cooling * T
        else:
            # Try to swap
            if not current_set:
                continue
            b_old = random.choice(list(current_set))
            
            # Pick a random block not in current_set
            # To avoid O(N) selection, we can just pick a random index and check
            # Or maintain a list of available blocks?
            # N is ~50k. Picking random index is O(1).
            b_new = random.randint(0, n_blocks - 1)
            if b_new in current_set:
                continue
                
            # Check if swap is valid
            # 1. Remove b_old: check if any tsub in b_old has count 1
            valid = True
            for s_idx in block_tsubs[b_old]:
                if cover_count[s_idx] == 1:
                    valid = False
                    break
            
            if valid:
                # 2. Add b_new: check if any tsub in b_new that was NOT in b_old has count 0
                # Note: tsubs in intersection will have count >= 1 (since b_old covered them, and we are removing b_old, so they drop by 1, but if they were in b_new, they will be added back).
                # Actually, simpler:
                # Tentatively remove b_old
                for s_idx in block_tsubs[b_old]:
                    cover_count[s_idx] -= 1
                    
                # Check if adding b_new restores coverage
                for s_idx in block_tsubs[b_new]:
                    if cover_count[s_idx] == 0:
                        valid = False
                        break
                        
                if valid:
                    # Commit swap
                    for s_idx in block_tsubs[b_new]:
                        cover_count[s_idx] += 1
                    current_set.discard(b_old)
                    current_set.add(b_new)
                    
                    if len(current_set) < len(best_set):
                        best_set = set(current_set)
                    T = cooling * T
                else:
                    # Rollback
                    for s_idx in block_tsubs[b_old]:
                        cover_count[s_idx] += 1
                    T = cooling * T
            else:
                T = cooling * T
                
    return [list(all_blocks[i]) for i in best_set]
