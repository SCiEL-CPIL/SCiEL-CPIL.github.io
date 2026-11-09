"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a permutation of job indices on flow-shop."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Efficient computation using 1D arrays
    prev = np.zeros(n_machines)
    for job in perm:
        curr = np.empty(n_machines)
        curr[0] = prev[0] + pt[job, 0]
        for m in range(1, n_machines):
            curr[m] = max(curr[m-1], prev[m]) + pt[job, m]
        prev = curr
    return prev[-1]


def _neh_heuristic(pt):
    """NEH (Nawaz, Enscore, Ham) heuristic for flow shop."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # Sort jobs by total processing time descending
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insert each job into best position
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _priority_heuristic(pt):
    """Priority-based heuristic using weighted machine priorities."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Weight machines by their position (later machines matter more for makespan)
    weights = np.linspace(1.0, 2.0, n_machines)
    priority = pt @ weights
    order = list(np.argsort(-priority))
    
    # Insert each job into best position using NEH-like insertion
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _johnsons_rule(pt):
    """Johnson's rule generalized for multi-machine flow shop."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_machines <= 2:
        if n_machines == 2:
            left = []
            right = []
            for j in range(n_jobs):
                if pt[j, 0] <= pt[j, 1]:
                    left.append((pt[j, 0], j))
                else:
                    right.append((pt[j, 1], j))
            left.sort(key=lambda x: x[0])
            right.sort(key=lambda x: -x[0])
            return [x[1] for x in left] + [x[1] for x in right]
        else:
            return list(np.argsort(pt[:, 0]))
    
    # For multi-machine: use sum of first half and second half
    mid = n_machines // 2
    a = pt[:, :mid].sum(axis=1)
    b = pt[:, mid:].sum(axis=1)
    
    left = []
    right = []
    for j in range(n_jobs):
        if a[j] <= b[j]:
            left.append((a[j], j))
        else:
            right.append((b[j], j))
    left.sort(key=lambda x: x[0])
    right.sort(key=lambda x: -x[0])
    return [x[1] for x in left] + [x[1] for x in right]


def _local_search_best_improvement(perm, pt, deadline=None, max_iterations=200):
    """
    Local search with best-improvement strategy using Or-opt and swap moves.
    Evaluates ALL moves in a neighborhood and picks the best one.
    """
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, pt)
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        best_move_ms = best_ms
        best_new_perm = None
        best_move_type = None
        best_move_params = None
        
        # Phase 1: Or-opt moves (move block of 1, 2, or 3 jobs)
        for block_len in [1, 2, 3]:
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
            
            for i in range(n_jobs - block_len + 1):
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                
                block = best_perm[i:i+block_len]
                
                # Try inserting the block at all valid positions
                for j in range(n_jobs - block_len + 1):
                    if j == i:
                        continue
                    
                    # Build new permutation
                    remaining = best_perm[:i] + best_perm[i+block_len:]
                    if j > i + block_len:
                        new_j = j - block_len
                    else:
                        new_j = j
                    
                    new_perm = remaining[:new_j] + block + remaining[new_j:]
                    
                    ms = _makespan(new_perm, pt)
                    
                    if ms < best_move_ms - 1e-12:
                        best_move_ms = ms
                        best_new_perm = new_perm
                        best_move_type = 'or_opt'
                        best_move_params = (i, j, block_len)
        
        # Phase 2: Swap moves
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
            for j in range(i + 1, n_jobs):
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                
                candidate = list(best_perm)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, pt)
                
                if ms < best_move_ms - 1e-12:
                    best_move_ms = ms
                    best_new_perm = candidate
                    best_move_type = 'swap'
                    best_move_params = (i, j)
        
        # Apply the best move found
        if best_new_perm is not None:
            best_perm = best_new_perm
            best_ms = best_move_ms
        else:
            # No improvement found, stop
            break
    
    return best_perm, best_ms


def _perturbation(perm, n_jobs, rng, strength=3):
    """Apply random perturbation: swap 'strength' pairs of adjacent jobs."""
    perturbed = list(perm)
    for _ in range(strength):
        i = rng.integers(0, n_jobs - 1)
        perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
    return perturbed


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # 1. NEH heuristic
    neh_perm = _neh_heuristic(pt)
    initial_solutions.append((neh_perm, _makespan(neh_perm, pt)))
    
    # 2. Johnson's rule
    johnson_perm = _johnsons_rule(pt)
    if johnson_perm != neh_perm:
        initial_solutions.append((johnson_perm, _makespan(johnson_perm, pt)))
    
    # 3. SPT on first machine
    spt_order = list(np.argsort(pt[:, 0]))
    if spt_order != neh_perm and spt_order != johnson_perm:
        initial_solutions.append((spt_order, _makespan(spt_order, pt)))
    
    # 4. LPT on last machine
    lpt_last = list(np.argsort(-pt[:, -1]))
    if lpt_last != neh_perm and lpt_last != johnson_perm and lpt_last != spt_order:
        initial_solutions.append((lpt_last, _makespan(lpt_last, pt)))
    
    # 5. Priority-based heuristic
    priority_perm = _priority_heuristic(pt)
    existing_perms = [p for p, _ in initial_solutions]
    if priority_perm not in existing_perms:
        initial_solutions.append((priority_perm, _makespan(priority_perm, pt)))
    
    # Sort initial solutions by makespan
    initial_solutions.sort(key=lambda x: x[1])
    
    best_perm, best_ms = initial_solutions[0]
    
    # Local search on best initial solution with best-improvement
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search_best_improvement(best_perm, pt, deadline, max_iterations=200)
    
    # Try local search on other initial solutions
    remaining = initial_solutions[1:]
    remaining.sort(key=lambda x: x[1])
    
    for alt_perm, alt_ms in remaining:
        if time.time() >= deadline - 15.0:
            break
        if alt_ms < best_ms + 0.15 * best_ms:  # Only if reasonably close
            alt_perm, alt_ms = _local_search_best_improvement(alt_perm, pt, deadline, max_iterations=100)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
    
    # Perturbation restarts if time remains
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() >= deadline - 10.0:
                break
            
            # Perturb the best solution
            perturbed = _perturbation(best_perm, n_jobs, rng, strength=3)
            
            # Local search from perturbed solution
            perturbed_perm, perturbed_ms = _local_search_best_improvement(
                perturbed, pt, deadline, max_iterations=50
            )
            
            if perturbed_ms < best_ms - 1e-12:
                best_perm = perturbed_perm
                best_ms = perturbed_ms
            elif time.time() >= deadline - 5.0:
                break
    
    return list(best_perm)
# EVOLVE-BLOCK-END
