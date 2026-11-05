import itertools
import random
import time

# Precomputation for C(24,7,3)
V_PRE = 24
K_PRE = 7
T_PRE = 3

all_t_subsets_pre = list(itertools.combinations(range(V_PRE), T_PRE))
subset_to_id_pre = {s: i for i, s in enumerate(all_t_subsets_pre)}
num_subsets_pre = len(all_t_subsets_pre)

all_k_blocks_pre = list(itertools.combinations(range(V_PRE), K_PRE))
block_to_subsets_pre = {}
for b in all_k_blocks_pre:
    subs = [subset_to_id_pre[s] for s in itertools.combinations(b, T_PRE)]
    block_to_subsets_pre[b] = subs

subset_to_blocks_pre = [[] for _ in range(num_subsets_pre)]
for b, subs in block_to_subsets_pre.items():
    for s in subs:
        subset_to_blocks_pre[s].append(b)

def construct(v, k, t):
    if (v, k, t) == (V_PRE, K_PRE, T_PRE):
        return _construct_precomputed()
    else:
        return _construct_generic(v, k, t)

def _construct_precomputed():
    start_time = time.time()
    time_limit = 110.0
    
    best_blocks = None
    best_size = float('inf')
    
    # Use MRV Greedy + Local Search
    # Multiple restarts to escape local minima
    num_restarts = 30
    time_per_restart = (time_limit - 5) / num_restarts
    
    for restart in range(num_restarts):
        random.seed(1000 + restart)
        
        # 1. Greedy Initialization (MRV)
        blocks = _greedy_init_mrv()
        
        # 2. Local Search
        blocks = _local_search(blocks, time.time() + time_per_restart)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        if time.time() > time_limit - 5:
            break
            
    # Convert tuples to lists
    if best_blocks:
        return [list(b) for b in best_blocks]
    return []

def _greedy_init_mrv():
    uncovered = set(range(num_subsets_pre))
    blocks = []
    
    while uncovered:
        target = None
        min_count = float('inf')
        
        for s in uncovered:
            cnt = len(subset_to_blocks_pre[s])
            if cnt < min_count:
                min_count = cnt
                target = s
                if cnt == 1:
                    break
                    
        if target is None:
            break
            
        candidates = subset_to_blocks_pre[target]
        best_block = None
        max_gain = -1
        
        for b in candidates:
            gain = 0
            for s in block_to_subsets_pre[b]:
                if s in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        for s in block_to_subsets_pre[best_block]:
            if s in uncovered:
                uncovered.remove(s)
                
    return blocks

def _local_search(blocks, deadline):
    counts = [0] * num_subsets_pre
    block_subsets = []
    
    for b in blocks:
        subs = block_to_subsets_pre[b]
        block_subsets.append(subs)
        for s in subs:
            counts[s] += 1
            
    def try_remove():
        removed = True
        while removed:
            removed = False
            for i in range(len(block_subsets)):
                redundant = True
                for s in block_subsets[i]:
                    if counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[i]:
                        counts[s] -= 1
                    block_subsets.pop(i)
                    blocks.pop(i)
                    removed = True
                    break
                    
    try_remove()
    
    while time.time() < deadline:
        if not blocks:
            break
            
        i = random.randint(0, len(blocks) - 1)
        b = blocks[i]
        b_set = set(b)
        
        x = random.choice(b)
        outside = [p for p in range(V_PRE) if p not in b_set]
        if not outside:
            continue
        y = random.choice(outside)
        
        new_b = tuple(sorted(list(b_set - {x}) | {y}))
        new_subs = block_to_subsets_pre[new_b]
        old_subs = block_subsets[i]
        
        old_sub_set = set(old_subs)
        new_sub_set = set(new_subs)
        
        lost = old_sub_set - new_sub_set
        valid = True
        for s in lost:
            if counts[s] <= 1:
                valid = False
                break
                
        if valid:
            for s in lost:
                counts[s] -= 1
            gained = new_sub_set - old_sub_set
            for s in gained:
                counts[s] += 1
                
            blocks[i] = new_b
            block_subsets[i] = new_subs
            
            try_remove()
            
    return blocks

def _construct_generic(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        max_gain = -1
        for b in itertools.combinations(range(v), k):
            if set(target).issubset(set(b)):
                gain = sum(1 for s in itertools.combinations(b, t) if s in uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
        if best_block:
            blocks.append(best_block)
            uncovered.difference_update(itertools.combinations(best_block, t))
    return [list(b) for b in blocks]
