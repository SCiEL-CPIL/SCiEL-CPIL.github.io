import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precomputation
    t_sets = list(itertools.combinations(range(v), t))
    M = len(t_sets)
    t_set_to_id = {ts: i for i, ts in enumerate(t_sets)}
    
    k_sets = list(itertools.combinations(range(v), k))
    N_blocks = len(k_sets)
    
    block_tsets = []
    for b in k_sets:
        covered = []
        for ts in itertools.combinations(b, t):
            covered.append(t_set_to_id[ts])
        block_tsets.append(covered)
        
    tset_blocks = [[] for _ in range(M)]
    for b_idx, ts_list in enumerate(block_tsets):
        for ts_id in ts_list:
            tset_blocks[ts_id].append(b_idx)
            
    best_blocks = None
    best_len = float('inf')
    
    def greedy(seed):
        rng = random.Random(seed)
        coverage = np.zeros(M, dtype=np.int32)
        uncovered_count = M
        active_blocks = []
        
        while uncovered_count > 0:
            # Find a t-set with coverage 0
            while True:
                idx = rng.randint(0, M - 1)
                if coverage[idx] == 0:
                    break
            
            candidates = tset_blocks[idx]
            best_gain = -1
            best_cand = None
            
            for c in candidates:
                gain = 0
                for ts_id in block_tsets[c]:
                    if coverage[ts_id] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
            
            active_blocks.append(best_cand)
            for ts_id in block_tsets[best_cand]:
                if coverage[ts_id] == 0:
                    uncovered_count -= 1
                coverage[ts_id] += 1
                
        return active_blocks

    def reduce(blocks):
        coverage = np.zeros(M, dtype=np.int32)
        for b in blocks:
            for ts_id in block_tsets[b]:
                coverage[ts_id] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                for ts_id in block_tsets[b]:
                    coverage[ts_id] -= 1
                
                redundant = True
                for ts_id in block_tsets[b]:
                    if coverage[ts_id] == 0:
                        redundant = False
                        break
                
                if redundant:
                    blocks.pop(i)
                    changed = True
                else:
                    for ts_id in block_tsets[b]:
                        coverage[ts_id] += 1
                        
        return blocks

    seed = 0
    while time.time() < deadline:
        blocks = greedy(seed)
        blocks = reduce(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
        seed += 1
        
    if best_blocks is None:
        return []
        
    result = [list(k_sets[b]) for b in best_blocks]
    return result
