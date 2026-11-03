import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0
    t0 = time.time()
    
    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build bitset for each block: which t-subsets it covers
    # Each bitset is a Python int with bit i set if t-subset i is covered
    block_bitsets = [0] * n_blocks
    # Also build list of t-subset indices per block for iteration
    block_tsubs = [None] * n_blocks
    
    for bi, block in enumerate(all_blocks):
        ts_list = []
        bits = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            ts_list.append(idx)
            bits |= (1 << idx)
        block_tsubs[bi] = ts_list
        block_bitsets[bi] = bits
    
    # Build inverse: for each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi in range(n_blocks):
        for ts_idx in block_tsubs[bi]:
            tsub_blocks[ts_idx].append(bi)
    
    # 2. LP Relaxation to get dual weights
    dual_weights = None
    try:
        # Build sparse matrix
        rows = []
        cols = []
        data = []
        for bi in range(n_blocks):
            for ts_idx in block_tsubs[bi]:
                rows.append(ts_idx)
                cols.append(bi)
                data.append(1.0)
        
        M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
        
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            dual_weights = np.abs(res.ineqlin.marginals)
            if np.sum(dual_weights) == 0:
                dual_weights = np.ones(n_tsubsets)
    except Exception:
        dual_weights = np.ones(n_tsubsets)
    
    if dual_weights is None:
        dual_weights = np.ones(n_tsubsets)
    
    # Precompute block weights from dual
    block_dual_weights = np.array([np.sum(dual_weights[block_tsubs[bi]]) for bi in range(n_blocks)])
    
    # 3. Bitset-based greedy
    def bitset_greedy(seed=0):
        rng = random.Random(seed)
        full_mask = (1 << n_tsubsets) - 1
        uncovered = full_mask
        
        # Track which blocks are still candidates
        remaining = set(range(n_blocks))
        
        selected_blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            
            # Find block with maximum weight among those covering uncovered t-subsets
            best_block = -1
            best_weight = -1
            best_gain = -1
            
            # Sample from remaining blocks if too many
            if len(remaining) > 5000:
                sample = random.sample(list(remaining), 5000)
            else:
                sample = list(remaining)
            
            for bi in sample:
                if bi not in remaining:
                    continue
                gain = bin(block_bitsets[bi] & uncovered).count('1')
                if gain <= 0:
                    continue
                # Weighted score: dual weight sum for newly covered t-subsets
                # Approximate using precomputed weights scaled by gain ratio
                total_w = block_dual_weights[bi]
                if total_w <= 0:
                    w_score = gain
                else:
                    # Estimate fraction of block's t-subsets that are uncovered
                    frac = gain / len(block_tsubs[bi])
                    w_score = total_w * frac + gain * 0.1
                
                if w_score > best_weight or (w_score == best_weight and gain > best_gain):
                    best_weight = w_score
                    best_gain = gain
                    best_block = bi
            
            if best_block == -1:
                # Fallback: pick any block with positive gain
                for bi in remaining:
                    if bin(block_bitsets[bi] & uncovered).count('1') > 0:
                        best_block = bi
                        break
                if best_block == -1:
                    break
            
            selected_blocks.append(best_block)
            remaining.discard(best_block)
            uncovered &= ~block_bitsets[best_block]
        
        return selected_blocks
    
    # 4. Local search with bitset operations
    def local_search(block_indices, seed=0):
        rng = random.Random(seed)
        selected = set(block_indices)
        
        # Compute coverage count per t-subset
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
        
        best_selected = set(selected)
        best_count = len(selected)
        
        max_steps = 20000
        no_improve_limit = 2000
        step = 0
        no_improve = 0
        
        while step < max_steps and time.time() < deadline - 2:
            step += 1
            
            if len(selected) < best_count:
                best_selected = set(selected)
                best_count = len(selected)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > no_improve_limit:
                    break
            
            if not selected:
                break
            
            sel_list = list(selected)
            r = rng.random()
            
            if r < 0.5:
                # Try to remove a block
                bi = sel_list[rng.randint(0, len(sel_list) - 1)]
                ts_indices = block_tsubs[bi]
                if all(cover_count[ts] > 1 for ts in ts_indices):
                    selected.discard(bi)
                    for ts in ts_indices:
                        cover_count[ts] -= 1
            else:
                # Try swap: remove one, add one that covers its unique t-subsets
                bi_to_remove = sel_list[rng.randint(0, len(sel_list) - 1)]
                ts_indices = block_tsubs[bi_to_remove]
                unique_tsubs = [ts for ts in ts_indices if cover_count[ts] == 1]
                
                if not unique_tsubs:
                    selected.discard(bi_to_remove)
                    for ts in ts_indices:
                        cover_count[ts] -= 1
                    continue
                
                # Find a replacement block that covers all unique t-subsets
                found = False
                for _ in range(50):
                    ci = rng.randint(0, n_blocks - 1)
                    if ci in selected:
                        continue
                    
                    cand_ts = set(block_tsubs[ci])
                    if all(ts in cand_ts for ts in unique_tsubs):
                        # Perform swap
                        selected.discard(bi_to_remove)
                        for ts in ts_indices:
                            cover_count[ts] -= 1
                        selected.add(ci)
                        for ts in block_tsubs[ci]:
                            cover_count[ts] += 1
                        found = True
                        break
                
                if not found and rng.random() < 0.02:
                    # Random add to escape local optimum
                    ci = rng.randint(0, n_blocks - 1)
                    if ci not in selected:
                        selected.add(ci)
                        for ts in block_tsubs[ci]:
                            cover_count[ts] += 1
        
        return [all_blocks[bi] for bi in best_selected]
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts with different seeds
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        init_blocks = bitset_greedy(seed)
        if not init_blocks:
            continue
        
        refined = local_search(init_blocks, seed)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined
    
    if best_blocks is None:
        # Fallback: simple greedy without bitsets
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        while uncovered:
            max_gain = max(gain)
            if max_gain == 0:
                break
            candidates = [i for i in range(n_blocks) if gain[i] == max_gain]
            chosen = candidates[0]
            blocks_used.append(chosen)
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            gain[chosen] = 0
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks
