import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

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
    ts_to_blocks = [[] for _ in range(n_ts)]
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # --- 1. Greedy Initialization ---
    def greedy_init(seed):
        local_rng = random.Random(seed)
        uncovered = np.ones(n_ts, dtype=bool)
        uncovered_count = n_ts
        gains = np.array([len(ts) for ts in block_ts], dtype=np.int32)
        selected = []
        
        while uncovered_count > 0:
            if time.time() - _start_time > _DEADLINE - 5.0:
                break
            best_b = np.argmax(gains)
            best_gain = gains[best_b]
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

    # --- 2. Polish: Remove redundant blocks ---
    def polish(blocks):
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
        changed = True
        while changed and time.time() - _start_time < _DEADLINE - 5.0:
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

    # --- 3. Local Search: Swap optimization ---
    def local_search(blocks, max_iter=200):
        """Try to improve by swapping one block for another, or adding one block."""
        if time.time() - _start_time > _DEADLINE - 10.0:
            return blocks
        
        blocks = list(blocks)
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() - _start_time < _DEADLINE - 10.0:
            improved = False
            iter_count += 1
            
            # Try removing a block and see if coverage is still valid
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                # Check if any set is covered ONLY by b
                can_remove = True
                for s in block_ts[b]:
                    if coverage[s] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for s in block_ts[b]:
                        coverage[s] -= 1
                    blocks.pop(i)
                    improved = True
                    break
            
            if not improved:
                # Try adding a block that covers the most uncovered-or-singly-covered sets
                # Actually, try to find a block that when added allows removal of 2+ existing blocks
                # Simpler: try swapping - remove block i, add block j, check validity
                for i in range(len(blocks)):
                    if time.time() - _start_time > _DEADLINE - 10.0:
                        return blocks
                    b_remove = blocks[i]
                    # Temporarily remove
                    for s in block_ts[b_remove]:
                        coverage[s] -= 1
                    
                    # Find best replacement
                    best_j = -1
                    best_gain = 0
                    for j in range(n_blocks):
                        if j == b_remove:
                            continue
                        if j in blocks:
                            continue
                        # Gain = number of sets that would go from coverage 0 to >0
                        gain = 0
                        for s in block_ts[j]:
                            if coverage[s] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                    
                    # Restore
                    for s in block_ts[b_remove]:
                        coverage[s] += 1
                    
                    if best_j >= 0 and best_gain > 0:
                        # Perform swap
                        blocks[i] = best_j
                        for s in block_ts[b_remove]:
                            coverage[s] -= 1
                        for s in block_ts[best_j]:
                            coverage[s] += 1
                        improved = True
                        break
        
        return blocks

    # --- 4. ILP Formulation ---
    def solve_ilp(initial_ub, time_limit, warm_start=None):
        c = np.ones(n_blocks)
        A = lil_matrix((n_ts, n_blocks), dtype=np.int8)
        for b_idx, ts_list in enumerate(block_ts):
            for s_idx in ts_list:
                A[s_idx, b_idx] = 1
        A = A.tocsr()
        lb = np.ones(n_ts)
        ub = np.inf * np.ones(n_ts)
        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
        integrality = np.ones(n_blocks)
        
        # Add objective bound to help solver
        # If we have a warm start, we can add a constraint that sum(x) <= initial_ub
        if warm_start is not None and initial_ub < n_blocks:
            # Add constraint: sum(x_b) <= initial_ub
            obj_row = np.ones(n_blocks, dtype=np.int8)
            A2 = lil_matrix((1, n_blocks), dtype=np.int8)
            A2[0, :] = 1
            A2 = A2.tocsr()
            A_all = lil_matrix((n_ts + 1, n_blocks), dtype=np.int8)
            A_all[:n_ts, :] = A
            A_all[n_ts, :] = A2
            A_all = A_all.tocsr()
            lb_all = np.concatenate([lb, [0]])
            ub_all = np.concatenate([ub, [initial_ub]])
            constraints = LinearConstraint(A_all, lb_all, ub_all)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                       options={'time_limit': time_limit, 'mip_rel_gap': 0.005})
            if res.status in (0, 1):
                x = res.x
                selected = [i for i in range(n_blocks) if x[i] > 0.5]
                return selected
        except Exception as e:
            pass
        return None

    # --- Main Execution ---
    # 1. Quick greedy + polish
    initial_blocks = greedy_init(42)
    initial_blocks = polish(initial_blocks)
    current_best = len(initial_blocks)
    best_solution = list(initial_blocks)
    
    # 2. Local search on greedy solution
    if time.time() - _start_time < _DEADLINE - 15.0:
        ls_solution = local_search(initial_blocks, max_iter=100)
        if len(ls_solution) < current_best:
            current_best = len(ls_solution)
            best_solution = ls_solution
    
    # 3. Try ILP with warm start
    if time.time() - _start_time < _DEADLINE - 5.0:
        ilp_time = _remaining() - 5.0
        if ilp_time > 10.0:
            ilp_solution = solve_ilp(current_best, ilp_time, warm_start=best_solution)
            if ilp_solution is not None:
                ilp_solution = polish(ilp_solution)
                if len(ilp_solution) < current_best:
                    current_best = len(ilp_solution)
                    best_solution = ilp_solution
    
    # 4. Final polish
    best_solution = polish(best_solution)
    
    # 5. Convert to list of lists
    result = [list(all_blocks[b]) for b in best_solution]
    return result
