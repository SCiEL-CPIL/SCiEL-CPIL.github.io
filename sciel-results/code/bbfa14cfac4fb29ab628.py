"""Covering design C(v,k,t) — Optimized Greedy Set Cover."""
import itertools
import time
import random

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 100.0   # 100s safety margin
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    def run_greedy():
        block_gain = [len(subs) for subs in block_tsubs]
        is_covered = [0] * N_t
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            best_block_idx = -1
            max_gain = -1
            for i in range(N_blocks):
                if block_gain[i] > max_gain:
                    max_gain = block_gain[i]
                    best_block_idx = i
            
            if best_block_idx == -1 or max_gain == 0:
                break
                
            selected_blocks.append(all_blocks[best_block_idx])
            
            for si in block_tsubs[best_block_idx]:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    for b_idx in tsub_blocks[si]:
                        block_gain[b_idx] -= 1
                        
        return selected_blocks

    def try_reduce(blocks):
        if not blocks:
            return blocks
        
        block_subs = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs.append(subs)
            
        n = len(blocks)
        cov = [0] * N_t
        for subs in block_subs:
            for si in subs:
                cov[si] += 1
                
        removed = [False] * n
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                for si in block_subs[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in block_subs[i]:
                        cov[si] -= 1
                        
        return [b for i, b in enumerate(blocks) if not removed[i]]

    sol = run_greedy()
    sol = try_reduce(sol)
    
    return sol
