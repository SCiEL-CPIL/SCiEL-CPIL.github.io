import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build block_tsubs: for each block, list of t-subset indices it covers
    # Also build the incidence matrix A (sparse)
    # A is n_tsubsets x n_blocks
    row_indices = []
    col_indices = []
    block_tsubs = [[] for _ in range(n_blocks)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            tsubs.append(idx)
            row_indices.append(idx)
            col_indices.append(bi)
        block_tsubs[bi] = tsubs
        
    data = np.ones(len(row_indices), dtype=np.float64)
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_tsubsets, n_blocks))
    
    # Build tsub_blocks for efficient gain updates
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    best_blocks = None
    best_indices = None
    best_len = float('inf')
    
    def greedy_cover_fast(seed=0):
        """Optimized Greedy set cover with NumPy vectorized gain updates."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Initial gain: number of t-subsets each block covers
        gain = np.array([len(tsubs) for tsubs in block_tsubs], dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline - 2:
                break
            
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(chosen)
            
            # Vectorized update: for each t-subset covered by chosen,
            # decrement gain for all blocks that cover it
            for ts_idx in block_tsubs[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Update gains for all blocks covering this t-subset
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            
            gain[chosen] = -1
        
        return blocks_used

    def remove_redundant_blocks(block_indices):
        """Remove redundant blocks from the solution."""
        if not block_indices:
            return block_indices
        
        # Count coverage of each t-subset by current blocks
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_indices:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        kept = [True] * len(block_indices)
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(block_indices):
                if not kept[i]:
                    continue
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    changed = True
                    
        return [block_indices[i] for i in range(len(block_indices)) if kept[i]]

    def local_search(block_indices, seed=0, max_swap_rounds=30):
        """
        Local search with multiple 1-for-1 swap rounds.
        Each round:
        1. Remove redundant blocks
        2. Try swaps to enable further reductions
        """
        if not block_indices:
            return block_indices
        
        current_blocks = list(block_indices)
        rng = random.Random(seed)
        
        for round_num in range(max_swap_rounds):
            if time.time() > deadline - 10:
                break
            
            # Phase 1: Remove redundant blocks
            current_blocks = remove_redundant_blocks(current_blocks)
            
            # Phase 2: Try 1-for-1 swaps
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for bi in current_blocks:
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] += 1
            
            # Create a set for O(1) lookup
            current_set = set(current_blocks)
            
            improved = False
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 10:
                    break
                
                # Find t-subsets uniquely covered by bi
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi] if cover_count[ts_idx] == 1]
                
                if not unique_tsubs:
                    # bi is redundant (shouldn't happen after removal, but just in case)
                    continue
                
                # Find candidate blocks that cover all unique_tsubs
                # Start with the rarest unique t-subset
                unique_tsubs_sorted = sorted(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                
                candidates = set(tsub_blocks[unique_tsubs_sorted[0]])
                for ts_idx in unique_tsubs_sorted[1:]:
                    candidates &= set(tsub_blocks[ts_idx])
                    if not candidates:
                        break
                
                # Remove current blocks from candidates
                candidates -= current_set
                
                if not candidates:
                    continue
                
                # Try a few random candidates
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for new_bi in cand_list[:3]:  # Try up to 3 candidates
                    # Perform the swap
                    current_blocks[i] = new_bi
                    current_set.discard(bi)
                    current_set.add(new_bi)
                    improved = True
                    break
                
                if improved:
                    break  # Restart the round with updated state
            
            if not improved:
                break  # No more improvements possible
        
        # Final redundancy removal
        current_blocks = remove_redundant_blocks(current_blocks)
        
        return current_blocks

    # Strategy: Multiple Greedy + Local Search
    num_trials = 30  # Increased from 20 due to faster greedy
    for seed in range(num_trials):
        if time.time() > deadline - 25:
            break
            
        blocks_idx = greedy_cover_fast(seed)
        if not blocks_idx:
            continue
            
        # Apply local search
        blocks_idx = local_search(blocks_idx, seed, max_swap_rounds=20)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_indices = blocks_idx
            best_blocks = [all_blocks[i] for i in blocks_idx]
            
            # If we've found a very good solution, do an intensive polish
            if best_len <= 145 and time.time() < deadline - 20:
                # Additional polish with more swap rounds
                polished = local_search(blocks_idx, seed + 1000, max_swap_rounds=50)
                if len(polished) < best_len:
                    best_len = len(polished)
                    best_indices = polished
                    best_blocks = [all_blocks[i] for i in polished]
            
        if best_len <= 142:  # Tightened threshold for early exit
            break

    # Try MILP with warm start if we have time
    if best_indices and time.time() < deadline - 20:
        c = np.ones(n_blocks, dtype=np.float64)
        lb = np.ones(n_tsubsets, dtype=np.float64)
        ub = np.full(n_tsubsets, np.inf, dtype=np.float64)
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n_blocks, dtype=np.int8)
        bounds = Bounds(0, 1)
        
        x0 = np.zeros(n_blocks)
        for i in best_indices:
            x0[i] = 1.0
            
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                x0=x0,
                options={'time_limit': int(deadline - time.time() - 8), 'mip_gap': 0.01}
            )
            
            if result.status in (0, 1):
                x = result.x
                selected_idx = [i for i in range(n_blocks) if x[i] > 0.5]
                if selected_idx and len(selected_idx) < best_len:
                    best_len = len(selected_idx)
                    best_indices = selected_idx
                    best_blocks = [all_blocks[i] for i in selected_idx]
        except Exception:
            pass

    if best_blocks is None:
        # Fallback
        blocks_idx = greedy_cover_fast(0)
        blocks_idx = local_search(blocks_idx, 0, max_swap_rounds=10)
        best_blocks = [all_blocks[i] for i in blocks_idx]
        
    return best_blocks
