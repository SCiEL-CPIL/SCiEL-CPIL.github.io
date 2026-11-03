import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Precomputation
    # All t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    # All k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # block_to_tsets: list of lists of tset indices
    # tset_to_blocks: list of lists of block indices
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # Helper to get a valid solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def run_local_search(initial_indices, max_time_per_run=10.0, seed=0):
        """
        Takes a list of block indices, applies local search improvements.
        Returns the best list of indices found.
        """
        current = list(initial_indices)
        if not current:
            return []
        
        rng = random.Random(seed)
        
        # Calculate initial cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        improved = True
        local_start = time.time()
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks (multiple passes until stable)
            while True:
                found_redundant = False
                order = list(range(len(current)))
                rng.shuffle(order)
                
                for i in order:
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    if i >= len(current):
                        continue
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break
                
                if not found_redundant or (time.time() - local_start) > max_time_per_run:
                    break
            
            if not current:
                break
                
            # Phase 2: 1-for-1 Swaps
            for strategy in range(3):
                if (time.time() - local_start) > max_time_per_run:
                    break
                    
                rng.shuffle(current)
                
                for i in range(len(current)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                        
                    b_out = current[i]
                    
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                            
                    if not unique_tsets:
                        continue

                    candidates = set(tset_to_blocks[unique_tsets[0]])
                    candidates.discard(b_out)
                    
                    for ts_idx in unique_tsets[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts_idx])
                    
                    if not candidates:
                        continue
                    
                    best_candidate = -1
                    best_score = -1
                    b_out_tsets_set = set(block_to_tsets[b_out])
                    
                    candidate_list = list(candidates)
                    limit = [50, 100, 200][strategy]
                    if len(candidate_list) > limit:
                        rng.shuffle(candidate_list)
                        candidate_list = candidate_list[:limit]
                    
                    for b_in in candidate_list:
                        score = 0
                        for ts_idx in block_to_tsets[b_in]:
                            if ts_idx not in b_out_tsets_set:
                                if strategy == 0:
                                    score += 1
                                elif strategy == 1:
                                    if cover_counts[ts_idx] == 1:
                                        score += 10
                                    else:
                                        score += 1
                                else:
                                    if cover_counts[ts_idx] <= 2:
                                        score += 5
                                    elif cover_counts[ts_idx] == 3:
                                        score += 2
                                    else:
                                        score += 1
                        if score > best_score:
                            best_score = score
                            best_candidate = b_in
                        
                    if best_candidate != -1:
                        for ts_idx in block_to_tsets[b_out]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[best_candidate]:
                            cover_counts[ts_idx] += 1
                        
                        current[i] = best_candidate
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                    
            # Loop back to Phase 1 if swaps were made
        
        return best_solution

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts
        
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])
                
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        while uncovered_count > 0 and time.time() < deadline - 15:
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                break
                
            selected.append(int(best_idx))
            selected_mask[best_idx] = True
            
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w
                            
        return selected

    def primal_dual_cover(seed, time_budget=20.0):
        """
        Primal-dual style construction:
        - Maintain dual prices y[ts] for each t-set
        - Iteratively raise y for uncovered t-sets
        - Select blocks with highest sum(y[ts] for ts in block) / cost
        - This mimics the primal-dual approximation for set cover
        """
        rng = random.Random(seed)
        pd_start = time.time()
        
        # Dual prices start at 0
        y = np.zeros(num_tsets, dtype=np.float64)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        selected_set = set()
        
        # Precompute block sizes for cost normalization
        # All blocks have the same size (k elements), so cost is uniform = 1
        # But we can use the number of t-sets a block covers as a proxy for "cost"
        # Actually in set cover, each set has cost 1, so ratio = sum(y) / 1 = sum(y)
        # Let's just use sum(y[ts] for ts in block) as the score
        
        while uncovered_count > 0 and (time.time() - pd_start) < time_budget:
            # Find uncovered t-sets
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            # Raise dual prices for uncovered t-sets
            # Use a small increment, possibly weighted by how "hard" they are to cover
            # t-sets with fewer covering blocks are harder
            for ts_idx in unc_idx:
                # Harder t-sets (fewer blocks cover them) get higher price increases
                n_covering = len(tset_to_blocks[ts_idx])
                # Inverse frequency weighting
                increment = 1.0 / (n_covering ** 0.5)
                y[ts_idx] += increment
            
            # Find best block: maximize sum(y[ts] for ts in block) among blocks covering at least one uncovered t-set
            # For efficiency, compute scores only for blocks that cover uncovered t-sets
            # But we need to check all blocks potentially
            
            # Compute scores vectorized-ish
            # score[b] = sum(y[ts] for ts in block_to_tsets[b])
            # Only consider blocks that cover at least one uncovered t-set
            # To speed up, maintain a list of "active" blocks (those covering uncovered t-sets)
            
            # Simple approach: scan all blocks, compute score
            best_score = -1.0
            best_idx = -1
            
            # Optimization: only consider blocks that have at least one uncovered t-set
            # Build this incrementally or just filter
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                # Check if this block covers any uncovered t-set
                has_uncovered = False
                score = 0.0
                for ts_idx in block_to_tsets[b_idx]:
                    if uncovered[ts_idx]:
                        has_uncovered = True
                        score += y[ts_idx]
                if has_uncovered and score > best_score:
                    best_score = score
                    best_idx = b_idx
            
            if best_idx == -1 or best_score <= 0:
                # Fallback: pick randomly from remaining
                remaining = [b for b in range(num_blocks) if b not in selected_set]
                if not remaining:
                    break
                best_idx = rng.choice(remaining)
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
            
            if uncovered_count == 0:
                break
        
        return selected

    def combined_construction(seed, time_budget=30.0):
        """
        Hybrid: start with primal-dual for a few steps, then switch to greedy.
        The dual prices from primal-dual guide the initial greedy weights.
        """
        rng = random.Random(seed)
        combo_start = time.time()
        
        # Phase 1: Primal-dual for a short time to establish dual prices
        pd_time = min(5.0, time_budget * 0.2)
        y = np.zeros(num_tsets, dtype=np.float64)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        selected_set = set()
        
        pd_deadline = combo_start + pd_time
        while uncovered_count > 0 and (time.time() - combo_start) < pd_deadline:
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            for ts_idx in unc_idx:
                n_covering = len(tset_to_blocks[ts_idx])
                increment = 1.0 / (n_covering ** 0.5)
                y[ts_idx] += increment
            
            best_score = -1.0
            best_idx = -1
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                has_uncovered = False
                score = 0.0
                for ts_idx in block_to_tsets[b_idx]:
                    if uncovered[ts_idx]:
                        has_uncovered = True
                        score += y[ts_idx]
                if has_uncovered and score > best_score:
                    best_score = score
                    best_idx = b_idx
            
            if best_idx == -1 or best_score <= 0:
                remaining = [b for b in range(num_blocks) if b not in selected_set]
                if not remaining:
                    break
                best_idx = rng.choice(remaining)
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
            
            if uncovered_count == 0:
                break
        
        # Phase 2: Continue with greedy using dual prices as weights
        if uncovered_count > 0 and (time.time() - combo_start) < time_budget:
            # Use y as weights for the remaining greedy phase
            # y[ts] > 0 for all t-sets (they were all raised), so use y directly
            # But normalize to avoid numerical issues
            y_norm = y / (np.max(y) + 1e-10)
            
            gains = np.zeros(num_blocks, dtype=np.float64)
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                if block_to_tsets[b_idx]:
                    gains[b_idx] = np.sum(y_norm[block_to_tsets[b_idx]])
            
            selected_mask = np.zeros(num_blocks, dtype=np.bool_)
            for b_idx in selected:
                selected_mask[b_idx] = True
            
            while uncovered_count > 0 and (time.time() - combo_start) < time_budget:
                best_idx = np.argmax(gains)
                if gains[best_idx] <= 0:
                    break
                    
                selected.append(int(best_idx))
                selected_mask[best_idx] = True
                
                for ts_idx in block_to_tsets[best_idx]:
                    if uncovered[ts_idx]:
                        uncovered[ts_idx] = False
                        uncovered_count -= 1
                        w = y_norm[ts_idx]
                        for b in tset_to_blocks[ts_idx]:
                            if not selected_mask[b]:
                                gains[b] -= w
            
            if uncovered_count == 0:
                break
        
        return selected

    best_global_solution = None
    best_global_size = float('inf')
    
    # Run 1: Standard greedy with extended local search
    t_start = time.time()
    init1 = greedy_cover(42)
    if init1:
        ls_time = min(40.0, deadline - t_start - 10)
        sol1 = run_local_search(init1, max_time_per_run=ls_time, seed=42)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1
            
    # Run 2: Primal-dual construction + local search
    t_start = time.time()
    if time.time() < deadline - 40:
        init2 = primal_dual_cover(123, time_budget=10.0)
        if init2:
            ls_time = min(25.0, deadline - t_start - 10)
            sol2 = run_local_search(init2, max_time_per_run=ls_time, seed=123)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2
    
    # Run 3: Combined primal-dual + greedy + local search
    t_start = time.time()
    if time.time() < deadline - 25:
        init3 = combined_construction(777, time_budget=15.0)
        if init3:
            ls_time = min(15.0, deadline - t_start - 5)
            sol3 = run_local_search(init3, max_time_per_run=ls_time, seed=777)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3
    
    # Final polish: Re-run local search on the best solution with remaining time
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time, seed=999)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol
            
    if not best_global_solution:
        if init1:
            return get_solution(init1)
        return []
        
    return get_solution(best_global_solution)

if __name__ == "__main__":
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
    covered = set()
    for block in res:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    total_tsets = len(list(itertools.combinations(range(v), t)))
    print(f"Covered: {len(covered)}/{total_tsets}")
