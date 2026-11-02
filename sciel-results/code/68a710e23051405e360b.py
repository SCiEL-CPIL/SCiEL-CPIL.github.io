import itertools
import random
import time
import heapq
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets and their indices
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

    def greedy_solve(seed):
        """Run a fast greedy pass to get an initial valid covering."""
        local_rng = random.Random(seed)
        local_start = time.time()
        
        # Budget: 5% of total time or max 5 seconds
        time_budget = min(5.0, _DEADLINE * 0.05)

        uncovered = [True] * n_ts
        uncovered_count = n_ts

        # Current gain for each block
        block_gain = [len(ts_list) for ts_list in block_ts]
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        # Heap: (-gain, block_index, version)
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            # Pop until valid
            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            # Update coverage
            for s_idx in block_ts[b_idx]:
                if not uncovered[s_idx]:
                    continue
                uncovered[s_idx] = False
                uncovered_count -= 1

                # Update gains of other blocks containing this t-subset
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        result = [list(all_blocks[b]) for b in selected]

        # Check if complete and fix if needed
        if uncovered_count > 0:
            unc_set = set(i for i in range(n_ts) if not uncovered[i])
            # Use heap for efficient fix-up
            fixup_heap = []
            for b_idx in range(n_blocks):
                if in_block[b_idx]:
                    continue
                gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                if gain > 0:
                    heapq.heappush(fixup_heap, (-gain, b_idx))
            
            while unc_set and fixup_heap and time.time() - local_start < _DEADLINE - 2.0:
                neg_gain, best_b = heapq.heappop(fixup_heap)
                if in_block[best_b]:
                    continue
                gain = -neg_gain
                if gain <= 0:
                    break
                result.append(list(all_blocks[best_b]))
                in_block[best_b] = True
                newly_covered = []
                for s in block_ts[best_b]:
                    if s in unc_set:
                        unc_set.discard(s)
                        newly_covered.append(s)
                # Update heap for blocks that might have changed gain
                for s in newly_covered:
                    for other_b in ts_to_blocks[s]:
                        if not in_block[other_b]:
                            new_gain = sum(1 for ss in block_ts[other_b] if ss in unc_set)
                            if new_gain > 0:
                                heapq.heappush(fixup_heap, (-new_gain, other_b))

        return result

    def local_search_improve(blocks, time_limit):
        """Improve a covering by removing redundant blocks and swapping."""
        local_start = time.time()
        
        if not blocks:
            return blocks

        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        current_indices = [block_to_idx[tuple(b)] for b in blocks]
        current_set = set(current_indices)
        
        # Rebuild coverage based on current_indices
        coverage = [0] * n_ts
        for idx in current_indices:
            for s_idx in block_ts[idx]:
                coverage[s_idx] += 1
        
        # Precompute block membership as frozensets for fast intersection
        block_ts_sets = [frozenset(ts_list) for ts_list in block_ts]
        
        # Helper to remove a block index
        def remove_block(idx):
            nonlocal coverage, current_set
            for s_idx in block_ts[idx]:
                coverage[s_idx] -= 1
            current_indices.remove(idx)
            current_set.discard(idx)

        # Helper to add a block index
        def add_block(idx):
            nonlocal coverage, current_set
            for s_idx in block_ts[idx]:
                coverage[s_idx] += 1
            current_indices.append(idx)
            current_set.add(idx)

        changed = True
        while changed and time.time() - local_start < time_limit:
            changed = False
            
            # Phase 1: Remove redundant blocks
            for i in range(len(current_indices) - 1, -1, -1):
                if time.time() - local_start > time_limit:
                    break
                idx = current_indices[i]
                redundant = True
                for s_idx in block_ts[idx]:
                    if coverage[s_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(idx)
                    changed = True
                    break # Restart loop as indices/coverage changed
            
            if time.time() - local_start > time_limit:
                break

            # Phase 2: Try swaps (1-for-1) to enable further reductions
            for i in range(len(current_indices)):
                if time.time() - local_start > time_limit:
                    break
                
                idx_to_remove = current_indices[i]
                
                # Tentatively remove
                for s_idx in block_ts[idx_to_remove]:
                    coverage[s_idx] -= 1
                
                # Check for uncovered
                uncovered_list = []
                for s_idx in block_ts[idx_to_remove]:
                    if coverage[s_idx] == 0:
                        uncovered_list.append(s_idx)
                
                if not uncovered_list:
                    # It was redundant
                    changed = True
                    break # Restart loop
                
                # Try to find a block that covers all uncovered_set
                best_new_idx = -1
                best_gain = 0
                uncovered_frozen = frozenset(uncovered_list)
                
                # Optimization: Only consider blocks that cover at least one uncovered t-subset
                candidate_blocks = set()
                for s_idx in uncovered_list:
                    for b_idx in ts_to_blocks[s_idx]:
                        if b_idx != idx_to_remove:
                            candidate_blocks.add(b_idx)
                
                for b_idx in candidate_blocks:
                    if b_idx in current_set:
                        continue
                    # Count how many of uncovered_set it covers using set intersection
                    gain = len(block_ts_sets[b_idx] & uncovered_frozen)
                    if gain == len(uncovered_list):
                        # Perfect cover
                        best_new_idx = b_idx
                        best_gain = gain
                        break
                    elif gain > best_gain:
                        best_gain = gain
                        best_new_idx = b_idx
                
                if best_new_idx != -1 and best_gain == len(uncovered_list):
                    # Add the new block
                    add_block(best_new_idx)
                    
                    # Now, check if any other block is redundant
                    redundant_found = False
                    for j in range(len(current_indices) - 1, -1, -1):
                        if j == len(current_indices) - 1:
                            continue
                        
                        idx_check = current_indices[j]
                        if idx_check == best_new_idx:
                            continue
                        
                        redundant = True
                        for s_idx in block_ts[idx_check]:
                            if coverage[s_idx] <= 1:
                                redundant = False
                                break
                        
                        if redundant:
                            remove_block(idx_check)
                            redundant_found = True
                            break
                    
                    if redundant_found:
                        changed = True
                        break # Restart main loop
                    else:
                        # No further reduction, revert the swap
                        remove_block(best_new_idx)
                        add_block(idx_to_remove)

        # Convert back to list of lists
        return [list(all_blocks[idx]) for idx in current_indices]

    # 1. Get initial greedy solutions
    # Run a few seeds to diversify
    best_result = []
    best_count = float('inf')
    
    seeds = [12345, 54321, 99999]
    for seed in seeds:
        if _remaining() < 10.0:
            break
        res = greedy_solve(seed)
        if len(res) < best_count:
            best_count = len(res)
            best_result = res

    # 2. Local Search on the best greedy result
    if _remaining() > 10.0:
        ls_time = min(_remaining() * 0.35, 25.0) # Use 35% of remaining time, max 25s
        improved = local_search_improve(best_result, ls_time)
        if len(improved) < best_count:
            best_count = len(improved)
            best_result = improved

    # 3. Try MILP optimization
    if _remaining() > 12.0:
        try:
            c = np.ones(n_blocks)
            
            row_indices = []
            col_indices = []
            data = []
            
            for b_idx, ts_list in enumerate(block_ts):
                for s_idx in ts_list:
                    row_indices.append(s_idx)
                    col_indices.append(b_idx)
                    data.append(1)
            
            A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_ts, n_blocks))
            
            constraints = [
                LinearConstraint(A, 1, np.inf)
            ]
            
            bounds = Bounds(0, 1)
            integrality = np.ones(n_blocks)
            
            # Warm start
            x0 = np.zeros(n_blocks)
            selected_set = set(tuple(blk) for blk in best_result)
            for b_idx, blk in enumerate(all_blocks):
                if tuple(blk) in selected_set:
                    x0[b_idx] = 1.0
            
            time_limit_s = _remaining() - 5.0
            if time_limit_s < 5:
                time_limit_s = 5
                
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                x0=x0,
                options={
                    "time_limit": time_limit_s,
                    "presolve": "default",
                    "mip_rel_gap": 0.0,
                    "mip_abs_gap": 0.0
                }
            )
            
            if res.status == 0 or res.status == 1:
                if res.x is not None:
                    new_blocks = []
                    for i in range(n_blocks):
                        if res.x[i] > 0.5:
                            new_blocks.append(list(all_blocks[i]))
                    
                    if new_blocks:
                        # Verify validity
                        cov_check_milp = [0] * n_ts
                        for blk in new_blocks:
                            for ts in itertools.combinations(blk, t):
                                cov_check_milp[ts_index[ts]] += 1
                        
                        if all(c > 0 for c in cov_check_milp):
                            if len(new_blocks) < best_count:
                                best_count = len(new_blocks)
                                best_result = new_blocks
                                
        except Exception as e:
            pass

    # 4. Final local search polish
    if _remaining() > 5.0 and best_result:
        ls_time = min(_remaining() - 2.0, 12.0)
        if ls_time > 1.0:
            improved = local_search_improve(best_result, ls_time)
            if len(improved) < best_count:
                best_count = len(improved)
                best_result = improved

    return best_result
