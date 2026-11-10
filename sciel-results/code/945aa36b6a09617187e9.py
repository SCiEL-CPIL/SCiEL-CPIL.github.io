import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    points = range(v)
    t_subs = list(itertools.combinations(points, t))
    t_sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    
    blocks = list(itertools.combinations(points, k))
    block_to_idx = {b: i for i, b in enumerate(blocks)}
    
    # block_covered_ids[i] = list of t-subset ids covered by blocks[i]
    block_covered_ids = []
    # blocks_by_t_sub[j] = list of block indices covering t_sub[j]
    blocks_by_t_sub = [[] for _ in t_subs]
    
    for b_idx, b in enumerate(blocks):
        ids = []
        for t_sub in itertools.combinations(b, t):
            tid = t_sub_to_id[t_sub]
            ids.append(tid)
            blocks_by_t_sub[tid].append(b_idx)
        block_covered_ids.append(ids)
        
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    timeout = 110 # Safety margin for 120s limit
    
    restart = 0
    while time.time() - start_time < timeout:
        restart += 1
        random.seed(restart)
        
        uncovered = set(range(len(t_subs)))
        cover_count = [0] * len(t_subs)
        current_blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            u_id = random.choice(list(uncovered))
            
            candidates = blocks_by_t_sub[u_id]
            
            best_gain = -1
            best_block_idx = -1
            best_gain_val = -1
            
            for b_idx in candidates:
                gain = 0
                for tid in block_covered_ids[b_idx]:
                    if tid in uncovered:
                        gain += 1
                if gain > best_gain_val:
                    best_gain_val = gain
                    best_block_idx = b_idx
            
            # Add best block
            current_blocks.append(blocks[best_block_idx])
            for tid in block_covered_ids[best_block_idx]:
                if tid in uncovered:
                    uncovered.remove(tid)
                cover_count[tid] += 1
            
        # Local search: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            # Recalculate cover_count from current_blocks to be safe? 
            # Or update incrementally? 
            # Given small size, recalc is safer and fast enough.
            cover_count = [0] * len(t_subs)
            for b in current_blocks:
                b_idx = block_to_idx[b]
                for tid in block_covered_ids[b_idx]:
                    cover_count[tid] += 1
            
            for i in range(len(current_blocks) - 1, -1, -1):
                b = current_blocks[i]
                b_idx = block_to_idx[b]
                redundant = True
                for tid in block_covered_ids[b_idx]:
                    if cover_count[tid] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for tid in block_covered_ids[b_idx]:
                        cover_count[tid] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break # Restart scan from end or beginning? Break to restart loop
            
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks[:]
            
    return best_blocks
