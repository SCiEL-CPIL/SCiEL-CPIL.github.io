"""Covering design C(19,8,5) - LP-guided rounding + greedy + local search."""
import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog


def _precompute(v, k, t):
    """Precompute all t-subsets, all k-blocks, and their relationships."""
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    block_to_idx = {b: i for i, b in enumerate(all_kblocks)}
    
    # For each block, which t-sets does it cover?
    block_to_tsets = []
    for block in all_kblocks:
        tsets_in_block = [tset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_tsets.append(tsets_in_block)
    
    # For each t-set, which blocks cover it?
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, tsets_list in enumerate(block_to_tsets):
        for ts_idx in tsets_list:
            tset_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for fast operations
    block_to_tsets_np = [np.array(lst, dtype=np.int32) for lst in block_to_tsets]
    tset_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tset_to_blocks]
    
    # Build incidence matrix A (num_tsets x num_kblocks) where A[i,j]=1 if block j covers tset i
    # This is large: C(19,5)=11628 rows, C(19,8)=75582 cols. Use sparse representation.
    # For LP, we need A.T (num_kblocks x num_tsets) for the constraint matrix.
    # Actually linprog expects A_ub (m x n) where m=constraints, n=variables.
    # Variables: x_j for each block j. Constraints: sum_j A[i,j] x_j >= 1 for each tset i.
    # linprog minimizes c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq, bounds.
    # So we need: -A x <= -1  =>  A_ub = -A, b_ub = -1
    # A is num_tsets x num_kblocks.
    
    return all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks


def _lp_rounding(v, k, t, precomp, deadline, seed=42):
    """Solve LP relaxation and iteratively round using dual prices."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    rng = random.Random(seed)
    
    # Build sparse incidence matrix for LP
    # A is (num_tsets, num_kblocks) sparse
    # For scipy linprog, we need A_ub of shape (num_tsets, num_kblocks)
    # Constraint: for each tset i, sum over blocks j covering i of x_j >= 1
    # Rewrite as: -sum x_j <= -1
    
    from scipy.sparse import lil_matrix, csr_matrix
    
    A_lil = lil_matrix((num_tsets, num_kblocks), dtype=np.float64)
    for b_idx, tsets_list in enumerate(block_to_tsets):
        for ts_idx in tsets_list:
            A_lil[ts_idx, b_idx] = 1.0
    
    A_csr = A_lil.tocsr()
    
    # LP: minimize sum(x_j) s.t. A x >= 1, 0 <= x_j <= 1
    # linprog form: minimize c^T x s.t. A_ub x <= b_ub
    # So A_ub = -A, b_ub = -1
    c = np.ones(num_kblocks, dtype=np.float64)
    A_ub = -A_csr
    b_ub = -np.ones(num_tsets, dtype=np.float64)
    bounds = [(0, 1)] * num_kblocks
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback to greedy if LP fails
        return _greedy_cover(v, k, t, precomp, seed=seed, deadline=deadline)
    
    if not result.success:
        return _greedy_cover(v, k, t, precomp, seed=seed, deadline=deadline)
    
    x = result.x  # LP solution
    
    # Iterative rounding
    # Start with fractional solution, round variables greedily using dual prices
    selected = set()
    remaining_tsets = set(range(num_tsets))
    x_remaining = x.copy()
    
    # Compute dual variables (prices) from the LP solution
    # For linprog with HiGHS, duals are in result.ineqlin (for A_ub constraints)
    # The dual variable for constraint i (tset i) is related to the shadow price
    if hasattr(result, 'ineqlin') and result.ineqlin is not None:
        duals = np.abs(result.ineqlin)  # Prices for each tset constraint
    else:
        # Fallback: use uniform prices
        duals = np.ones(num_tsets, dtype=np.float64)
    
    # Iterative rounding: repeatedly pick the variable with best "reduced cost"
    # reduced cost = 1 - sum of duals for tsets it covers (if positive, it's worth adding)
    # Actually, in the LP context, we round by picking variables with highest x_j
    # but guided by duals: prefer variables that cover high-price (hard) tsets
    
    # Phase 1: Round variables with x_j > 0.5
    for j in range(num_kblocks):
        if time.time() > deadline:
            break
        if x[j] > 0.5:
            selected.add(j)
    
    # Check coverage after initial rounding
    covered = np.zeros(num_tsets, dtype=bool)
    for j in selected:
        for ts_idx in block_to_tsets[j]:
            covered[ts_idx] = True
    
    uncovered = np.where(~covered)[0]
    
    # Phase 2: Greedily add blocks to cover remaining uncovered tsets
    # Use dual prices to guide selection
    while len(uncovered) > 0 and time.time() < deadline:
        if len(uncovered) == 0:
            break
        
        # For each candidate block not yet selected, compute "weighted gain"
        # weighted gain = sum of duals[ts] for uncovered ts in block
        best_block = -1
        best_score = -1
        
        # Sample candidates for efficiency
        candidate_pool = [j for j in range(num_kblocks) if j not in selected]
        if len(candidate_pool) > 500:
            candidate_pool = rng.sample(candidate_pool, 500)
        
        for j in candidate_pool:
            score = 0.0
            for ts_idx in block_to_tsets[j]:
                if not covered[ts_idx]:
                    score += duals[ts_idx]
            if score > best_score:
                best_score = score
                best_block = j
        
        if best_block == -1 or best_score <= 0:
            # Fallback to plain greedy
            max_gain = 0
            for j in candidate_pool:
                gain = sum(1 for ts_idx in block_to_tsets[j] if not covered[ts_idx])
                if gain > max_gain:
                    max_gain = gain
                    best_block = j
            if best_block == -1:
                break
        
        selected.add(best_block)
        for ts_idx in block_to_tsets[best_block]:
            covered[ts_idx] = True
        
        uncovered = np.where(~covered)[0]
    
    # Convert to block list
    sol = [list(all_kblocks[j]) for j in sorted(selected)]
    return sol


def _greedy_cover(v, k, t, precomp, seed=0, explore_prob=0.0, deadline=None):
    """Greedy set-cover with vectorized gain computation."""
    if deadline is None:
        deadline = time.time() + 100
    
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    rng = random.Random(seed)
    
    # Initialize gain array: each block covers C(k,t) t-sets initially
    gain = np.full(num_kblocks, len(block_to_tsets[0]), dtype=np.int32)
    is_uncovered = np.full(num_tsets, True, dtype=bool)
    is_selected = np.full(num_kblocks, False, dtype=bool)
    sol = []
    
    while True:
        if time.time() > deadline:
            return None
        
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
        
        candidates = np.where((gain == max_gain) & (~is_selected))[0]
        if len(candidates) == 0:
            break
        
        if explore_prob > 0 and rng.random() < explore_prob:
            explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
            if len(explore_cands) > 0:
                b_idx = int(explore_cands[rng.randint(0, len(explore_cands) - 1)])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
        else:
            if len(candidates) > 1:
                sample_size = min(20, len(candidates))
                sample_idx = rng.sample(range(len(candidates)), sample_size)
                sample_blocks = [candidates[si] for si in sample_idx]
                
                best_b = None
                best_score = float('inf')
                for b in sample_blocks:
                    bts = block_to_tsets_np[b]
                    already = np.count_nonzero(~is_uncovered[bts])
                    score = already
                    if score < best_score:
                        best_score = score
                        best_b = b
                b_idx = int(best_b)
            else:
                b_idx = int(candidates[0])
        
        sol.append(list(all_kblocks[b_idx]))
        is_selected[b_idx] = True
        gain[b_idx] = -1
        
        bts = block_to_tsets_np[b_idx]
        newly_covered_mask = is_uncovered[bts]
        newly_covered = bts[newly_covered_mask]
        
        if len(newly_covered) == 0:
            continue
        
        is_uncovered[newly_covered] = False
        
        blocks_to_decrement = []
        for ts_idx in newly_covered:
            blocks_to_decrement.extend(tset_to_blocks[ts_idx])
        
        if blocks_to_decrement:
            counts = np.bincount(blocks_to_decrement, minlength=num_kblocks)
            gain -= counts
            gain[is_selected] = -1000
    
    return sol


def _local_search(blocks, precomp, deadline, seed=42):
    """Local search to reduce block count via redundancy removal, swaps, and 2-for-1 merges."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    blocks = [list(b) for b in blocks]
    rng = random.Random(seed)
    
    cover_count = np.zeros(num_tsets, dtype=np.int32)
    block_ts_indices = []
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_ts_indices.append(block_to_tsets[idx])
            block_indices.append(idx)
        else:
            ts_list = [tset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_ts_indices.append(ts_list)
            block_indices.append(-1)
        for ts_idx in block_ts_indices[-1]:
            cover_count[ts_idx] += 1
    
    def try_remove_redundant():
        for bi in range(len(blocks)):
            redundant = True
            for ts_idx in block_ts_indices[bi]:
                if cover_count[ts_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts_idx in block_ts_indices[bi]:
                    cover_count[ts_idx] -= 1
                blocks.pop(bi)
                block_ts_indices.pop(bi)
                block_indices.pop(bi)
                return True
        return False
    
    while time.time() < deadline:
        if not try_remove_redundant():
            break
    
    merge_improved = True
    while merge_improved and time.time() < deadline:
        merge_improved = False
        n = len(blocks)
        for i in range(n):
            if time.time() > deadline:
                return blocks
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return blocks
                
                unique_ts = []
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] <= 2:
                        unique_ts.append(ts_idx)
                for ts_idx in block_ts_indices[j]:
                    if cover_count[ts_idx] <= 2:
                        if ts_idx not in unique_ts:
                            unique_ts.append(ts_idx)
                
                if not unique_ts:
                    continue
                
                candidates = set(tset_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(tset_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                sol_idx_set = set(block_indices)
                valid_cands = [c for c in candidates if c not in sol_idx_set]
                if not valid_cands:
                    continue
                
                new_idx = rng.choice(valid_cands)
                
                for ts_idx in block_ts_indices[j]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_tsets[new_idx]:
                    cover_count[ts_idx] += 1
                
                blocks.pop(j)
                block_ts_indices.pop(j)
                block_indices.pop(j)
                blocks.pop(i)
                block_ts_indices.pop(i)
                block_indices.pop(i)
                
                blocks.append(list(all_kblocks[new_idx]))
                block_ts_indices.append(block_to_tsets[new_idx])
                block_indices.append(new_idx)
                
                merge_improved = True
                break
            if merge_improved:
                break
    
    swap_improved = True
    max_swap_iters = 30
    for _ in range(max_swap_iters):
        if time.time() > deadline:
            break
        if not swap_improved:
            break
        swap_improved = False
        
        sol_idx_set = set(block_indices)
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            
            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            
            if not unique_ts:
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                blocks.pop(i)
                block_ts_indices.pop(i)
                block_indices.pop(i)
                swap_improved = True
                break
            
            candidates = set(tset_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(tset_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_cands = [c for c in candidates if c not in sol_idx_set]
            if not valid_cands:
                continue
            
            new_idx = rng.choice(valid_cands)
            
            old_idx = block_indices[i]
            for ts_idx in block_to_tsets[old_idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_tsets[new_idx]:
                cover_count[ts_idx] += 1
            
            blocks[i] = list(all_kblocks[new_idx])
            block_ts_indices[i] = block_to_tsets[new_idx]
            block_indices[i] = new_idx
            sol_idx_set.discard(old_idx)
            sol_idx_set.add(new_idx)
            swap_improved = True
            break
    
    while time.time() < deadline:
        if not try_remove_redundant():
            break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    precomp = _precompute(v, k, t)
    
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: LP-guided rounding
    lp_deadline = time.time() + 30
    lp_sol = _lp_rounding(v, k, t, precomp, lp_deadline, seed=42)
    if lp_sol is not None and len(lp_sol) < best_len:
        best_len = len(lp_sol)
        best_solution = lp_sol
    
    # Phase 2: Multiple greedy restarts
    greedy_deadline = time.time() + 20
    for i in range(30):
        if time.time() > greedy_deadline:
            break
        explore = 0.0 if i < 10 else (0.1 if i < 20 else 0.2)
        sol = _greedy_cover(v, k, t, precomp, seed=i, explore_prob=explore, deadline=greedy_deadline)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Phase 3: Local search
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_solution is not None:
        num_ls_restarts = max(1, int(remaining_time / 15))
        num_ls_restarts = min(num_ls_restarts, 8)
        ls_time_per = (remaining_time - 3) / num_ls_restarts
        
        for ls_i in range(num_ls_restarts):
            if time.time() > deadline - 3:
                break
            ls_deadline = time.time() + ls_time_per
            ls_sol = _local_search(best_solution, precomp, ls_deadline, seed=100 + ls_i)
            if len(ls_sol) < best_len:
                best_len = len(ls_sol)
                best_solution = ls_sol
    
    if best_solution is None:
        # Fallback
        best_solution = _greedy_cover(v, k, t, precomp, seed=0, deadline=deadline)
        if best_solution is None:
            best_solution = [list(precomp[1][0])]
    
    return best_solution
