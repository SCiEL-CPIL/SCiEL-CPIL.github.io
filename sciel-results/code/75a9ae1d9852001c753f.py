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
    
    # block_tsubs: list of arrays of t-subset indices for each block
    block_tsubs = [None] * n_blocks
    block_bitsets = None  # We'll use cover_count instead for speed
    
    rows = []
    cols = []
    data = []
    
    Ckt = int(np.math.comb(k, t)) if hasattr(np, 'math') else 35  # C(7,4)=35
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
        for ts_idx in tsubs:
            rows.append(ts_idx)
            cols.append(bi)
            data.append(1.0)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
    
    # tsub_blocks: for each t-subset, list of block indices covering it
    tsub_blocks = [None] * n_tsubsets
    for i in range(n_tsubsets):
        tsub_blocks[i] = M[i, :].indices.astype(np.int32)
    
    # 2. LP Relaxation
    dual_weights = None
    try:
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            dual_weights = np.abs(res.ineqlin.marginals)
            if np.sum(dual_weights) < 1e-6:
                dual_weights = np.ones(n_tsubsets)
    except:
        dual_weights = np.ones(n_tsubsets)
    
    # 3. Greedy Initialization with LP weights
    def lp_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # Precompute block weights from dual weights
        block_weights = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            block_weights[bi] = np.sum(dual_weights[block_tsubs[bi]])
        
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
                
            max_w = block_weights.max()
            if max_w <= 1e-9:
                break
                
            chosen = int(np.argmax(block_weights))
            selected_blocks.append(chosen)
            
            ts_list = block_tsubs[chosen]
            active_mask = uncovered[ts_list]
            
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                uncovered[active_ts_indices] = False
                
                for ts_idx in active_ts_indices:
                    w_sub = dual_weights[ts_idx]
                    if w_sub == 0:
                        continue
                    blocks_to_update = tsub_blocks[ts_idx]
                    block_weights[blocks_to_update] -= w_sub
                    
            block_weights[chosen] = -1e10
            
        return selected_blocks

    # 4. Local Search with efficient essential-t-subset swaps
    def local_search(blocks, seed=0, max_iterations=500):
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        reduced = list(blocks)
        best_reduced = list(reduced)
        best_len = len(reduced)
        
        # State
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for b in reduced:
            for idx in block_tsubs[b]:
                cover_count[idx] += 1
        
        in_solution = np.zeros(n_blocks, dtype=bool)
        for b in reduced:
            in_solution[b] = True
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.995
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
                
            improved = False
            
            # Phase 1: Remove redundant blocks (cover_count >= 2 for all tsubs in block)
            changed = True
            while changed:
                changed = False
                for i in range(len(reduced)):
                    if time.time() > deadline - 3:
                        break
                    
                    b_idx = reduced[i]
                    ts_list = block_tsubs[b_idx]
                    # Check if all t-subsets in this block are covered by at least 2 blocks
                    if np.all(cover_count[ts_list] >= 2):
                        # Remove block i
                        for idx in ts_list:
                            cover_count[idx] -= 1
                        in_solution[b_idx] = False
                        reduced.pop(i)
                        changed = True
                        improved = True
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        break  # Restart scan
            
            if len(reduced) < 2:
                break
                
            # Phase 2: Swap using essential t-subsets (fast O(1) candidate check)
            if not improved:
                n = len(reduced)
                order = list(range(n))
                rng.shuffle(order)
                
                for i in order:
                    if time.time() > deadline - 3:
                        break
                        
                    b_idx = reduced[i]
                    ts_list = block_tsubs[b_idx]
                    
                    # Find essential t-subsets (covered only by this block)
                    essential_mask = (cover_count[ts_list] == 1)
                    essential_t_idxs = ts_list[essential_mask]
                    
                    if len(essential_t_idxs) == 0:
                        # This block is redundant - should have been caught in Phase 1
                        # Remove it
                        for idx in ts_list:
                            cover_count[idx] -= 1
                        in_solution[b_idx] = False
                        reduced.pop(i)
                        improved = True
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        break
                        
                    # Find candidates: blocks that cover ALL essential t-subsets
                    # Intersection of tsub_blocks for each essential t-subset
                    candidate_set = None
                    for t_idx in essential_t_idxs:
                        covering = tsub_blocks[t_idx]
                        if candidate_set is None:
                            candidate_set = set(covering.tolist())
                        else:
                            candidate_set.intersection_update(covering.tolist())
                        
                        if not candidate_set:
                            break
                    
                    if not candidate_set:
                        continue
                    
                    # Filter out blocks already in solution
                    candidates = [b for b in candidate_set if not in_solution[b]]
                    
                    if not candidates:
                        continue
                    
                    # Try a few random candidates
                    rng.shuffle(candidates)
                    found_swap = False
                    
                    for ci in candidates[:8]:
                        # Verify: after swap, all t-subsets still covered
                        # Essential t-subsets are covered by ci (by construction)
                        # Non-essential t-subsets in b_idx have cover_count >= 2,
                        # so removing b_idx leaves them with cover_count >= 1
                        # Adding ci only increases counts
                        # So the swap is valid!
                        
                        # Perform swap
                        old_b = reduced[i]
                        for idx in block_tsubs[old_b]:
                            cover_count[idx] -= 1
                        in_solution[old_b] = False
                        
                        for idx in block_tsubs[ci]:
                            cover_count[idx] += 1
                        in_solution[ci] = True
                        
                        reduced[i] = ci
                        improved = True
                        found_swap = True
                        break
                        
                    if found_swap:
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        break  # Restart Phase 1
            
            # Cooling
            T *= cooling_rate
            
            if not improved and T <= T_min:
                break
                
        return best_reduced

    # 5. Simulated Annealing Polish (as per directive: smooth surrogate / final polish)
    def sa_polish(blocks, seed=0, max_steps=2000):
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        current = list(blocks)
        best = list(current)
        best_len = len(current)
        
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for b in current:
            for idx in block_tsubs[b]:
                cover_count[idx] += 1
        
        in_solution = np.zeros(n_blocks, dtype=bool)
        for b in current:
            in_solution[b] = True
        
        T = 0.5
        T_min = 0.005
        cooling = 0.999
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            action = rng.random()
            
            if action < 0.3:
                # Try removal
                idx = rng.randint(0, len(current) - 1)
                b_idx = current[idx]
                ts_list = block_tsubs[b_idx]
                
                if np.all(cover_count[ts_list] >= 2):
                    for i in ts_list:
                        cover_count[i] -= 1
                    in_solution[b_idx] = False
                    current.pop(idx)
                    if len(current) < best_len:
                        best_len = len(current)
                        best = list(current)
                else:
                    # Reject (would break coverage)
                    pass
                    
            elif action < 0.8:
                # Try swap
                idx = rng.randint(0, len(current) - 1)
                b_idx = current[idx]
                ts_list = block_tsubs[b_idx]
                
                essential_mask = (cover_count[ts_list] == 1)
                essential_t_idxs = ts_list[essential_mask]
                
                if len(essential_t_idxs) == 0:
                    # Redundant, remove
                    for i in ts_list:
                        cover_count[i] -= 1
                    in_solution[b_idx] = False
                    current.pop(idx)
                    if len(current) < best_len:
                        best_len = len(current)
                        best = list(current)
                    continue
                    
                # Find candidates covering all essential t-subsets
                candidate_set = None
                for t_idx in essential_t_idxs:
                    covering = tsub_blocks[t_idx]
                    if candidate_set is None:
                        candidate_set = set(covering.tolist())
                    else:
                        candidate_set.intersection_update(covering.tolist())
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue
                
                candidates = [b for b in candidate_set if not in_solution[b]]
                if not candidates:
                    continue
                
                ci = candidates[rng.randint(0, len(candidates) - 1)]
                
                # Perform swap
                old_b = current[idx]
                for i in block_tsubs[old_b]:
                    cover_count[i] -= 1
                in_solution[old_b] = False
                
                for i in block_tsubs[ci]:
                    cover_count[i] += 1
                in_solution[ci] = True
                
                current[idx] = ci
                
                if len(current) < best_len:
                    best_len = len(current)
                    best = list(current)
                    
            else:
                # Try add a block (rare, for SA exploration)
                # Skip - we're minimizing, adding makes worse
                pass
            
            T *= cooling
            
            if T <= T_min and step > 100:
                break
                
        return best

    # Main loop: multiple trials
    best_blocks = None
    best_len = float('inf')
    
    num_trials = 25
    
    for seed in range(num_trials):
        if time.time() > deadline - 20:
            break
            
        blocks_idx = lp_greedy(seed)
        if not blocks_idx:
            continue
            
        # Local search
        blocks_idx = local_search(blocks_idx, seed, max_iterations=300)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = list(blocks_idx)
            
    # Final SA polish on best
    if best_blocks is not None and time.time() < deadline - 15:
        polished = sa_polish(best_blocks, 42, max_steps=3000)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(polished)
    
    # One more local search + SA on the polished result
    if best_blocks is not None and time.time() < deadline - 8:
        final_ls = local_search(best_blocks, 777, max_iterations=200)
        if len(final_ls) < best_len:
            best_blocks = final_ls
            best_len = len(best_blocks)
        
        if time.time() < deadline - 5:
            final_sa = sa_polish(best_blocks, 999, max_steps=1500)
            if len(final_sa) < best_len:
                best_blocks = final_sa
                best_len = len(best_blocks)
    
    if best_blocks is None:
        blocks_idx = lp_greedy(0)
        best_blocks = local_search(blocks_idx, 0, max_iterations=100)
    
    return [list(all_blocks[i]) for i in best_blocks]
