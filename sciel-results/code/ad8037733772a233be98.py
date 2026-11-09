import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Map block -> list of t-subset indices
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
    
    def run_greedy(seed):
        random.seed(seed)
        # Initial gains: number of t-subsets covered by each block
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE - 5.0:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            # Find all blocks with max gain
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            
            selected_indices.append(best_block_idx)
            
            # Update coverage and gains
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    block_gain[tsub_blocks[si]] -= 1
                        
        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []
        
        n_sel = len(selected_indices)
        global_to_local = {g: i for i, g in enumerate(selected_indices)}
        
        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)
        
        # Compute total_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1
                
        # Compute crit_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1
                    
        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel
        
        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue
                
            removed[i] = True
            g = selected_indices[i]
            
            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]
                
                if old_c == 2 and new_c == 1:
                    # si is now critical. Update crit_count for all other blocks containing si.
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        if g_other in global_to_local:
                            i_other = global_to_local[g_other]
                            if not removed[i_other]:
                                crit_count[i_other] += 1
                
                elif old_c == 1:
                    pass
        
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def is_valid_cover(selected_indices):
        """Check if selected_indices form a valid cover."""
        covered = np.zeros(N_t, dtype=np.int8)
        for g in selected_indices:
            for si in block_tsubs[g]:
                covered[si] = 1
        return np.all(covered == 1)

    def solve_ilp(initial_solution=None, time_limit=30.0):
        """
        Solve the ILP formulation of the covering problem.
        If initial_solution is provided, use it as a warm start.
        Returns a list of block indices or None if no solution found.
        """
        # Objective: minimize sum of x_i (number of blocks selected)
        c = np.ones(N_blocks, dtype=np.float64)
        
        # Constraints: for each t-subset, at least one block containing it must be selected
        # A[i, j] = 1 if block j covers t-subset i
        A = np.zeros((N_t, N_blocks), dtype=np.int8)
        for i, blocks in enumerate(tsub_blocks):
            for j in blocks:
                A[i, j] = 1
        
        # Linear constraints: A @ x >= 1
        constraint = LinearConstraint(A, 1, np.inf)
        
        # Bounds: x_i in {0, 1}
        bounds = Bounds(0, 1)
        
        # Integrate variables
        integrality = np.ones(N_blocks, dtype=np.int8)
        
        # Initial point from greedy if available
        x0 = None
        if initial_solution is not None and len(initial_solution) > 0:
            x0 = np.zeros(N_blocks, dtype=np.float64)
            for idx in initial_solution:
                if idx < N_blocks:
                    x0[idx] = 1.0
        
        # Set time limit
        options = {'time_limit': time_limit}
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, 
                      bounds=bounds, x0=x0, options=options)
            
            if res.status == 0 and res.x is not None:
                # Extract selected blocks
                selected = np.where(res.x > 0.5)[0].tolist()
                return selected
            elif res.status == 1:  # Optimal
                selected = np.where(res.x > 0.5)[0].tolist()
                return selected
            else:
                return None
        except Exception as e:
            print(f"ILP solver error: {e}")
            return None

    def local_search(selected_indices, max_iter=100):
        """
        Local search: try to remove blocks or swap blocks to improve the solution.
        Returns an improved list of global block indices.
        """
        if not selected_indices:
            return []
        
        current = list(selected_indices)
        current_set = set(current)
        n_sel = len(current)
        
        # Compute coverage counts
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        # Try removals first
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < DEADLINE - 5.0:
            improved = False
            iter_count += 1
            for i in range(len(current)):
                g = current[i]
                # Check if removing g keeps the cover valid
                removable = True
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    # Remove g
                    current.pop(i)
                    current_set.discard(g)
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                    improved = True
                    break  # Restart loop after removal
        
        # Try swaps: replace one block with another that covers more uncovered t-subsets
        if time.time() < DEADLINE - 5.0:
            random.shuffle(current)
            for i in range(len(current)):
                if time.time() >= DEADLINE - 5.0:
                    break
                g_old = current[i]
                # Find blocks that could replace g_old
                candidates = []
                # Sample some random blocks
                for _ in range(50):
                    g_new = random.randint(0, N_blocks - 1)
                    if g_new not in current_set:
                        candidates.append(g_new)
                
                for g_new in candidates:
                    # Check if replacing g_old with g_new keeps the cover valid
                    temp_count = total_count.copy()
                    for si in block_tsubs[g_old]:
                        temp_count[si] -= 1
                    for si in block_tsubs[g_new]:
                        temp_count[si] += 1
                    
                    if np.all(temp_count >= 1):
                        # Valid replacement
                        current[i] = g_new
                        current_set.discard(g_old)
                        current_set.add(g_new)
                        total_count = temp_count
                        improved = True
                        break
                
                if improved:
                    break  # Restart after swap
        
        return current

    # Phase 1: Get an initial solution via greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol_idx = list(sol_idx)
    best_len = len(best_sol_idx)
    
    # Phase 2: Try ILP with the greedy solution as warm start
    ilp_time_left = min(60.0, DEADLINE - time.time() - 10.0)
    if ilp_time_left > 5.0:
        ilp_result = solve_ilp(initial_solution=sol_idx, time_limit=ilp_time_left)
        if ilp_result is not None and len(ilp_result) < best_len:
            best_sol_idx = ilp_result
            best_len = len(best_sol_idx)
    
    # Phase 3: Apply local search to the best solution
    if time.time() < DEADLINE - 5.0:
        improved = local_search(best_sol_idx, max_iter=200)
        if len(improved) < best_len:
            best_sol_idx = improved
            best_len = len(best_sol_idx)
    
    # Phase 4: Try another ILP round if time allows
    if time.time() < DEADLINE - 15.0:
        ilp_time_left = min(30.0, DEADLINE - time.time() - 10.0)
        if ilp_time_left > 5.0:
            ilp_result = solve_ilp(initial_solution=best_sol_idx, time_limit=ilp_time_left)
            if ilp_result is not None and len(ilp_result) < best_len:
                best_sol_idx = ilp_result
                best_len = len(best_sol_idx)
    
    # Final local search
    if time.time() < DEADLINE - 5.0:
        improved = local_search(best_sol_idx, max_iter=100)
        if len(improved) < best_len:
            best_sol_idx = improved
            best_len = len(best_sol_idx)
    
    # Convert to blocks
    best_sol = [all_blocks[i] for i in best_sol_idx]
    
    return best_sol
