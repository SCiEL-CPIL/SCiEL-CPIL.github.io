import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # Build sparse incidence matrix A (n_t x n_k)
    # A[i, j] = 1 if t-subset i is in block j
    rows = []
    cols = []
    data = []
    for j, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            i = sub_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    
    # Also prepare bit masks for fast local search
    block_masks = [0] * n_k
    for j, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            i = sub_to_idx[t_sub]
            mask |= (1 << i)
        block_masks[j] = mask
        
    full_mask = (1 << n_t) - 1
    block_bit_counts = [m.bit_count() for m in block_masks]

    # 2. LP Relaxation (LASSO-like guidance)
    # Minimize sum(x) subject to A x >= 1, x >= 0
    # This is a standard LP.
    lp_x = None
    try:
        if time.time() < deadline - 80: # Allow 35s for LP
            c = np.ones(n_k)
            # A x >= 1  =>  -A x <= -1
            A_ub = -A_sparse
            b_ub = -np.ones(n_t)
            bounds = [(0, None)] * n_k
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='HiGHS')
            if res.success:
                lp_x = res.x
                # lp_x is a fractional solution
    except Exception:
        lp_x = None

    # 3. Greedy Construction with LP Bias
    def greedy_construct(use_lp_bias=True, rng=None):
        uncovered = full_mask
        selected = []
        remaining = [True] * n_k
        
        # If we have LP solution, we can use it to break ties or bias gain
        # Score = Gain + bias * lp_x[i]
        bias = 0.5 if use_lp_bias and lp_x is not None else 0.0
        
        while uncovered != 0:
            if time.time() > deadline - 20:
                break
            best_idx = -1
            best_score = -1.0
            unc = uncovered
            bm = block_masks
            for i in range(n_k):
                if not remaining[i]:
                    continue
                gain = (bm[i] & unc).bit_count()
                if gain == 0:
                    continue
                score = float(gain)
                if bias > 0:
                    score += bias * lp_x[i]
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~bm[best_idx]
            remaining[best_idx] = False
            
        return selected

    # 4. Pruning
    def prune_blocks(selected_indices):
        if not selected_indices:
            return selected_indices
        
        cover_count = [0] * n_t
        for idx in selected_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb
        
        current = selected_indices[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cover_count[i] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                
                if is_critical:
                    new_current.append(idx)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    improved = True
            current = new_current
        return current

    # 5. Local Search (Deletion + Swap)
    def local_search(selected_indices, max_time=10):
        if not selected_indices:
            return selected_indices
        
        current = prune_blocks(selected_indices)
        current_set = set(current)
        
        cov = [0] * n_t
        for idx in current:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb
        
        # Deletion Pass
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                removed_idx = current[i]
                mask = block_masks[removed_idx]
                can_remove = True
                m = mask
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    if cov[k2] == 1:
                        can_remove = False
                        break
                    m ^= lsb
                if can_remove:
                    m = mask
                    while m:
                        lsb = m & -m
                        k2 = lsb.bit_length() - 1
                        cov[k2] -= 1
                        m ^= lsb
                    current.pop(i)
                    current_set.discard(removed_idx)
                    changed = True
                    i -= 1
        
        # Swap Pass (1-for-1)
        # Try to replace a block with another that covers same or more, 
        # but allows further reduction? 
        # Simple swap: If removing i uncovers some, can j cover them?
        # Actually, standard 1-swap in set cover is:
        # Remove i. If uncovered, find j that covers max uncovered.
        # If j covers all uncovered, swap i->j.
        # This doesn't reduce size, but might allow future deletions.
        
        if time.time() < deadline - 10:
            for _ in range(5): # A few passes
                if time.time() > deadline - 10:
                    break
                improved = False
                for i in range(len(current)):
                    if time.time() > deadline - 10:
                        break
                    removed_idx = current[i]
                    mask = block_masks[removed_idx]
                    
                    # Temp remove
                    m = mask
                    while m:
                        lsb = m & -m
                        k2 = lsb.bit_length() - 1
                        cov[k2] -= 1
                        m ^= lsb
                    
                    # Find uncovered
                    uncovered_mask = 0
                    for k2 in range(n_t):
                        if cov[k2] == 0:
                            uncovered_mask |= (1 << k2)
                    
                    if uncovered_mask == 0:
                        # Just delete
                        current.pop(i)
                        current_set.discard(removed_idx)
                        improved = True
                        i -= 1
                        continue
                        
                    # Find best replacement
                    best_add = -1
                    best_gain = -1
                    for j in range(n_k):
                        if j in current_set:
                            continue
                        gain = (block_masks[j] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_add = j
                        if gain == uncovered_mask.bit_count():
                            break
                            
                    if best_add != -1 and best_gain == uncovered_mask.bit_count():
                        # Swap
                        current[i] = best_add
                        current_set.discard(removed_idx)
                        current_set.add(best_add)
                        m = block_masks[best_add]
                        while m:
                            lsb = m & -m
                            k2 = lsb.bit_length() - 1
                            cov[k2] += 1
                            m ^= lsb
                        improved = True
                    else:
                        # Restore
                        m = mask
                        while m:
                            lsb = m & -m
                            k2 = lsb.bit_length() - 1
                            cov[k2] += 1
                            m ^= lsb
                if not improved:
                    break
                    
        return prune_blocks(current)

    # 6. ILS
    rng = random.Random(42)
    best_solution = greedy_construct(use_lp_bias=True, rng=rng)
    best_solution = prune_blocks(best_solution)
    best_solution = local_search(best_solution, max_time=15)
    best_len = len(best_solution)
    
    # Try standard greedy if LP bias didn't help or as backup
    if time.time() < deadline - 40:
        sol = greedy_construct(use_lp_bias=False)
        sol = prune_blocks(sol)
        sol = local_search(sol, max_time=15)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)

    # ILS Loop
    ils_iter = 0
    while time.time() < deadline - 20 and ils_iter < 50:
        ils_iter += 1
        perturbed = best_solution[:]
        perturbed_set = set(perturbed)
        
        # Perturbation: Remove 1-2 blocks
        if len(perturbed) > 2:
            n_remove = rng.randint(1, 2)
            to_remove = rng.sample(range(len(perturbed)), min(n_remove, len(perturbed)-1))
            for i in sorted(to_remove, reverse=True):
                perturbed_set.discard(perturbed[i])
                perturbed.pop(i)
                
        # Repair: Add blocks to cover
        cov = [0] * n_t
        for idx in perturbed:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb
                
        uncovered = 0
        for k2 in range(n_t):
            if cov[k2] == 0:
                uncovered |= (1 << k2)
                
        while uncovered != 0 and time.time() < deadline - 15:
            best_idx = -1
            best_gain = -1
            for j in range(n_k):
                if j in perturbed_set:
                    continue
                gain = (block_masks[j] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            if best_idx == -1:
                break
            perturbed.append(best_idx)
            perturbed_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        perturbed = prune_blocks(perturbed)
        perturbed = local_search(perturbed, max_time=10)
        
        if len(perturbed) < best_len:
            best_solution = perturbed
            best_len = len(perturbed)
            
    # Final Validation
    final_cov = [0] * n_t
    for idx in best_solution:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            k2 = lsb.bit_length() - 1
            final_cov[k2] += 1
            m ^= lsb
            
    sol_set = set(best_solution)
    for k2 in range(n_t):
        if final_cov[k2] == 0:
            for j in range(n_k):
                if (block_masks[j] >> k2) & 1:
                    if j not in sol_set:
                        best_solution.append(j)
                        sol_set.add(j)
                    break
                    
    blocks = [list(all_k_blocks[i]) for i in best_solution]
    
    # Ensure distinct
    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)
            
    return unique_blocks
