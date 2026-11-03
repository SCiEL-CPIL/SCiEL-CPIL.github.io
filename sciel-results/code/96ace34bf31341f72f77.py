import itertools
import random
import time
import numpy as np
from collections import defaultdict

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets (targets)
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # --- 1. Greedy Initialization ---
    def greedy_init(seed):
        local_rng = random.Random(seed)
        uncovered = np.ones(n_ts, dtype=bool)
        uncovered_count = n_ts
        
        gains = np.zeros(n_blocks, dtype=np.int32)
        for b in range(n_blocks):
            gains[b] = len(block_ts[b])
        
        selected = []
        
        while uncovered_count > 0:
            if time.time() - _start_time > _DEADLINE - 5.0:
                break
                
            best_b = -1
            best_gain = -1
            for b in range(n_blocks):
                if gains[b] > best_gain:
                    best_gain = gains[b]
                    best_b = b
            
            if best_gain <= 0:
                break
            
            selected.append(best_b)
            
            for s_idx in block_ts[best_b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    uncovered_count -= 1
                    for other_b in ts_to_blocks[s_idx]:
                        gains[other_b] -= 1
                        if gains[other_b] < 0:
                            gains[other_b] = 0
        
        return selected

    # --- 2. Simulated Annealing Optimization ---
    def sa_optimize(initial_blocks, seed, time_limit):
        local_rng = random.Random(seed)
        start_time = time.time()
        
        in_block = np.zeros(n_blocks, dtype=bool)
        coverage = np.zeros(n_ts, dtype=np.int32)
        
        for b in initial_blocks:
            in_block[b] = True
            for s in block_ts[b]:
                coverage[s] += 1
        
        n_uncovered = int(np.sum(coverage == 0))
        n_selected = len(initial_blocks)
        
        P = 1000
        
        def get_obj():
            return n_selected + P * n_uncovered
        
        current_obj = get_obj()
        best_obj = current_obj
        best_in_block = in_block.copy()
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        block_indices = np.arange(n_blocks)
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
                
            move_type = local_rng.random()
            
            if move_type < 0.6:
                if n_selected == 0:
                    continue
                b = local_rng.choice(block_indices)
                if not in_block[b]:
                    continue
                
                in_block[b] = False
                n_selected -= 1
                
                new_uncovered = 0
                for s in block_ts[b]:
                    coverage[s] -= 1
                    if coverage[s] == 0:
                        new_uncovered += 1
                
                delta_uncovered = new_uncovered
                delta_obj = -1 + P * delta_uncovered
                
                if local_rng.random() < np.exp(-delta_obj / T) if delta_obj > 0 else True:
                    n_uncovered += delta_uncovered
                    current_obj += delta_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    in_block[b] = True
                    n_selected += 1
                    for s in block_ts[b]:
                        coverage[s] += 1
                        
            elif move_type < 0.8:
                b = local_rng.choice(block_indices)
                if in_block[b]:
                    continue
                
                in_block[b] = True
                n_selected += 1
                
                old_uncovered = 0
                for s in block_ts[b]:
                    if coverage[s] == 0:
                        old_uncovered += 1
                    coverage[s] += 1
                
                delta_uncovered = -old_uncovered
                delta_obj = 1 + P * delta_uncovered
                
                if local_rng.random() < np.exp(-delta_obj / T) if delta_obj > 0 else True:
                    n_uncovered += delta_uncovered
                    current_obj += delta_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    in_block[b] = False
                    n_selected -= 1
                    for s in block_ts[b]:
                        coverage[s] -= 1
                        
            else:
                b1 = local_rng.choice(block_indices)
                if not in_block[b1]:
                    continue
                b2 = local_rng.choice(block_indices)
                if in_block[b2]:
                    continue
                if b1 == b2:
                    continue
                
                in_block[b1] = False
                in_block[b2] = True
                
                new_uncovered_delta = 0
                
                for s in block_ts[b1]:
                    if coverage[s] == 1:
                        new_uncovered_delta += 1
                    coverage[s] -= 1
                
                for s in block_ts[b2]:
                    if coverage[s] == 0:
                        new_uncovered_delta -= 1
                    coverage[s] += 1
                
                delta_obj = P * new_uncovered_delta
                
                if local_rng.random() < np.exp(-delta_obj / T) if delta_obj > 0 else True:
                    n_uncovered += new_uncovered_delta
                    current_obj += delta_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    for s in block_ts[b2]:
                        coverage[s] -= 1
                    for s in block_ts[b1]:
                        coverage[s] += 1
                    in_block[b1] = True
                    in_block[b2] = False
                    
            T *= cooling
            
        best_cov = np.zeros(n_ts, dtype=np.int32)
        best_blocks_list = []
        for b in range(n_blocks):
            if best_in_block[b]:
                best_blocks_list.append(b)
                for s in block_ts[b]:
                    best_cov[s] += 1
        
        if np.any(best_cov == 0):
             return initial_blocks
        else:
             return best_blocks_list

    # --- 3. Repair ---
    def repair(blocks):
        coverage = np.zeros(n_ts, dtype=np.int32)
        in_block = set(blocks)
        
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
        
        uncovered = np.where(coverage == 0)[0]
        
        while len(uncovered) > 0 and time.time() - _start_time < _DEADLINE - 2.0:
            s = uncovered[0]
            candidates = ts_to_blocks[s]
            
            best_cand = -1
            best_gain = 0
            for b in candidates:
                if b in in_block:
                    continue
                gain = 0
                for ss in block_ts[b]:
                    if coverage[ss] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = b
            
            if best_cand == -1:
                break
                
            blocks.append(best_cand)
            in_block.add(best_cand)
            
            for ss in block_ts[best_cand]:
                coverage[ss] += 1
            
            uncovered = np.where(coverage == 0)[0]
            
        return blocks

    # --- 4. Polish ---
    def polish(blocks):
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
                
        changed = True
        while changed and time.time() - _start_time < _DEADLINE - 1.0:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for s in block_ts[b]:
                    if coverage[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_ts[b]:
                        coverage[s] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # --- Main Execution ---
    
    # 1. Greedy Init
    initial_blocks = greedy_init(42)
    
    # 2. SA Optimization
    sa_time = _remaining() * 0.8
    if sa_time > 10:
        optimized_blocks = sa_optimize(initial_blocks, 12345, sa_time)
    else:
        optimized_blocks = initial_blocks
    
    # 3. Repair
    optimized_blocks = repair(optimized_blocks)
    
    # 4. Polish
    optimized_blocks = polish(optimized_blocks)
    
    # 5. Convert to list of lists of ints
    result = [list(all_blocks[b]) for b in optimized_blocks]
    
    return result
