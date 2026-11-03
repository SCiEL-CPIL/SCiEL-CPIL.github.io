import itertools
import numpy as np

def _run_single_greedy(v, k, t, all_blocks, M, P, tsub_points, n_blocks, n_tsubs, initial_point_count, rng):
    uncovered = np.ones(n_tsubs, dtype=bool)
    uncovered_int = uncovered.astype(np.int8)
    used = np.zeros(n_blocks, dtype=bool)
    point_tsub_count = initial_point_count.copy()
    
    blocks = []
    
    while uncovered.any():
        scores = M.dot(uncovered_int).astype(np.float64)
        
        max_pt = point_tsub_count.max()
        if max_pt > 0:
            under_cover = 1.0 - (point_tsub_count / max_pt).astype(np.float64)
        else:
            under_cover = np.ones(v, dtype=np.float64)
            
        bias_scores = P.dot(under_cover)
        bias_scale = 0.001
        total_scores = scores + bias_scale * bias_scores
        
        total_scores[used] = -1.0
        
        best_score = np.max(total_scores)
        if best_score <= 0:
            break
            
        candidates = np.where(total_scores == best_score)[0]
        
        if len(candidates) == 0:
             break

        best_idx = candidates[rng.integers(len(candidates))]
        
        used[best_idx] = True
        blk = all_blocks[best_idx]
        blocks.append(list(blk))
        
        covered_by_block = M[best_idx, :]
        newly_covered_idx = np.where(covered_by_block & uncovered)[0]
        
        uncovered[covered_by_block] = False
        uncovered_int[covered_by_block] = 0
        
        for s_idx in newly_covered_idx:
            pts = tsub_points[s_idx]
            point_tsub_count[pts] -= 1
            
    return blocks

def construct(v, k, t):
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    M = np.zeros((n_blocks, n_tsubs), dtype=bool)
    P = np.zeros((n_blocks, v), dtype=bool)
    
    for b, blk in enumerate(all_blocks):
        for p in blk:
            P[b, p] = True
        for ts in itertools.combinations(blk, t):
            s_idx = tsub_to_idx[ts]
            M[b, s_idx] = True
            
    tsub_points = [np.array(ts, dtype=np.int64) for ts in all_tsubs]
    
    initial_point_count = np.zeros(v, dtype=np.int64)
    for ts in all_tsubs:
        for p in ts:
            initial_point_count[p] += 1
            
    best_blocks = None
    best_len = float('inf')
    
    num_runs = 10
    
    for i in range(num_runs):
        rng = np.random.default_rng(i)
        blocks = _run_single_greedy(v, k, t, all_blocks, M, P, tsub_points, n_blocks, n_tsubs, initial_point_count, rng)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
