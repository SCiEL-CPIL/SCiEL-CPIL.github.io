import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq
import time

def construct(v, k, t):
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Build sparse matrix A where A[i, j] = 1 if t_subsets[i] is a subset of k_subsets[j]
    rows = []
    cols = []
    data = []
    
    # Also build subsets_of_block and blocks_of_subset for the dynamic greedy
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    
    # LP Formulation:
    # Minimize sum(x_j)
    # Subject to: M.T @ x >= ones(n_t)
    # Which is: -M.T @ x <= -ones(n_t)
    
    c = np.ones(n_k)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_t)
    bounds = [(0, 1)] * n_k
    
    duals = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return _fallback_greedy(v, k, t)
        
        # Extract dual variables
        # res.ineqlin.marginals corresponds to A_ub constraints
        # The constraint is -M.T @ x <= -1.
        # The marginal is the change in objective per unit change in b_ub.
        # For minimization with <= constraints, marginals are <= 0.
        # The "value" or dual variable y_i for the original constraint >= 1 is -marginal[i].
        duals = -res.ineqlin.marginals
        duals = np.maximum(duals, 0)
        
    except Exception:
        return _fallback_greedy(v, k, t)
    
    # Dynamic LP-Greedy
    # 1. Initialize block weights
    block_weights = np.zeros(n_k)
    for j in range(n_k):
        # Sum of duals for subsets in block j
        # subsets_of_block[j] is a list of indices
        if subsets_of_block[j]:
            block_weights[j] = np.sum(duals[subsets_of_block[j]])
            
    # 2. Heap initialization
    # (negative_weight, block_index)
    heap = [(-block_weights[j], j) for j in range(n_k)]
    heapq.heapify(heap)
    
    covered = np.zeros(n_t, dtype=bool)
    picked = np.zeros(n_k, dtype=bool)
    blocks = []
    
    while heap:
        neg_w, j = heapq.heappop(heap)
        w = -neg_w
        
        # Lazy deletion: check if this entry is still valid
        # It's valid if the current weight matches the stored weight (approx) and not picked
        # Since we push updates, old entries remain.
        # We need to check if the actual current weight is still w?
        # It's hard to track exact current weight without an array.
        # We have `block_weights` array.
        # If `block_weights[j]` is not close to `w`, it's an old entry.
        # Also if `picked[j]` is True, skip.
        
        if picked[j]:
            continue
            
        # Check if weight changed significantly
        # Due to floating point, use a small epsilon or just compare
        # Actually, we update `block_weights` array in place.
        # The heap contains stale entries.
        # We must verify if `w` is still the correct weight.
        # If `block_weights[j] != w`, it's stale.
        # However, `block_weights[j]` decreases. So `w` (from heap) might be larger than `block_weights[j]`.
        # If `w > block_weights[j] + 1e-9`, it's stale.
        if w > block_weights[j] + 1e-9:
            continue
            
        # Check if this block covers any *uncovered* subsets
        # If not, its effective weight for new coverage is 0.
        # We should skip it if it doesn't add new coverage.
        # But the weight `w` is sum of duals of *all* subsets in B.
        # If all are covered, `w` should have been reduced to 0?
        # Yes, because when a subset S is covered, we subtract y[S] from all blocks containing S.
        # So if a block only contains covered subsets, its weight should be 0.
        # So if w > 0, it implies it has some uncovered subsets with positive dual.
        
        if w <= 1e-9:
            continue
            
        # Select block j
        picked[j] = True
        blocks.append(list(k_subsets[j]))
        
        # Update weights for blocks that share subsets with j
        # For each subset S in block j:
        #   If S is not covered:
        #     Mark S as covered
        #     For each block B' containing S:
        #       block_weights[B'] -= duals[S]
        #       Push (-block_weights[B'], B') into heap
        
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                y_s = duals[s_idx]
                if y_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        if not picked[b_prime]:
                            block_weights[b_prime] -= y_s
                            heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                            
    # Check if all covered
    if not np.all(covered):
        # Fallback for any remaining
        # This should rarely happen if LP is valid, but just in case
        remaining_indices = np.where(~covered)[0]
        # Simple greedy for remaining
        rem_blocks = _fallback_greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets)
        blocks.extend(rem_blocks)
        
    # --- POST-PROCESSING: REDUNDANCY REMOVAL ---
    # Convert blocks list to indices for easier manipulation
    # We need to map block tuple to index
    block_index_map = {tuple(b): i for i, b in enumerate(k_subsets)}
    
    # Get indices of selected blocks
    selected_indices = [block_index_map[tuple(b)] for b in blocks]
    
    # Calculate coverage counts
    cover_count = np.zeros(n_t, dtype=int)
    for idx in selected_indices:
        for s_idx in subsets_of_block[idx]:
            cover_count[s_idx] += 1
            
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to allow removal without index shifting issues in a list
        for i in range(len(selected_indices) - 1, -1, -1):
            idx = selected_indices[i]
            # Check if removing this block makes any subset uncovered
            is_essential = False
            for s_idx in subsets_of_block[idx]:
                if cover_count[s_idx] == 1:
                    is_essential = True
                    break
            
            if not is_essential:
                # Remove block
                for s_idx in subsets_of_block[idx]:
                    cover_count[s_idx] -= 1
                selected_indices.pop(i)
                changed = True
                # We could break here to restart the loop, but continuing is also fine
                # Restarting is safer to find new redundancies created by removal
                # However, removing a block only decreases counts, so it cannot make 
                # another block *more* redundant if it wasn't already? 
                # Wait. If I remove B1, counts decrease. 
                # A block B2 that was essential (count 1 for some S) might become 
                # non-essential if B1 also covered S? No, if B1 covered S, count[S] was >= 2.
                # If B2 was essential, count[S]=1 for some S in B2. 
                # Removing B1 (which doesn't cover S, otherwise count[S]>=2) doesn't affect S.
                # So removing a block never makes another block *less* essential.
                # It might make another block *more* redundant? 
                # If B3 had count[S]=2 (B2 and B3 cover S), and we remove B2, count[S]=1. 
                # B3 becomes essential for S. 
                # So we MUST restart the loop or iterate carefully.
                # Since we are iterating backwards and popping, and removals only decrease counts,
                # a block that was non-essential remains non-essential? 
                # No. If B3 was non-essential (all counts >= 2), and we remove B2 (which covered S in B3), 
                # count[S] drops to 1. B3 becomes essential.
                # So we MUST restart the scan.
                break
                
    # Reconstruct blocks from selected_indices
    final_blocks = [list(k_subsets[idx]) for idx in selected_indices]
    
    return final_blocks

def _fallback_greedy(v, k, t):
    # Standard max-coverage greedy
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    # Precompute subsets for each block? No, generate on fly or use precomputed if available.
    # For simplicity in fallback, just use a slow but robust method.
    # Actually, we can reuse the main logic's data structures if we pass them, but signature is fixed.
    # So we must be self-contained or fast.
    # Given v=19, k=7, t=4, this is small.
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    
    while uncovered:
        best_block = None
        best_count = -1
        for B in k_subsets:
            # Count how many uncovered t-subsets are in B
            count = 0
            for S in itertools.combinations(B, t):
                if S in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = B
        if best_count == 0:
            break
        blocks.append(list(best_block))
        # Remove covered
        for S in itertools.combinations(best_block, t):
            if S in uncovered:
                uncovered.remove(S)
    return blocks

def _fallback_greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets):
    # Convert indices back to tuples
    inv_t_index = {i: s for s, i in t_index.items()}
    remaining_sets = set(inv_t_index[i] for i in remaining_indices)
    
    if not remaining_sets:
        return []
        
    blocks = []
    # Use standard greedy on the remaining sets
    # We only need to cover these specific sets
    # We can still use any k-subset, but we prioritize those that cover remaining sets
    
    while remaining_sets:
        best_block = None
        best_count = -1
        for B in k_subsets:
            count = 0
            for S in itertools.combinations(B, t):
                if S in remaining_sets:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = B
        if best_count == 0:
            break
        blocks.append(list(best_block))
        for S in itertools.combinations(best_block, t):
            if S in remaining_sets:
                remaining_sets.remove(S)
    return blocks
