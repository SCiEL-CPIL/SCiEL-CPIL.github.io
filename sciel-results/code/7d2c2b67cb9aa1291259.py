"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


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


def _makespan_fast(perm, pt):
    """Optimized makespan computation using 2D array for speed."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Build completion matrix C[j][m] = completion time of job j on machine m
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt_perm = pt[np.asarray(perm, dtype=np.int64)]
    
    # First machine: cumulative sum
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    if n_machines > 1:
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + pt_perm[0, m]
            for j in range(1, n_jobs):
                c_prev_m = C[j-1, m]
                c_curr_m1 = C[j, m-1]
                C[j, m] = (c_curr_m1 if c_curr_m1 > c_prev_m else c_prev_m) + pt_perm[j, m]
    
    return C[n_jobs-1, n_machines-1]


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
            ms = _makespan_fast(candidate, pt)
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


def _local_search(perm, pt, deadline=None, use_swap=False):
    """Local search with insertion and optional swap moves."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    improved = True
    
    while improved:
        improved = False
        
        # Phase 1: insertion moves
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                candidate = best_perm[:i] + best_perm[i+1:]
                insert_pos = j if j < i else j - 1
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                ms = _makespan_fast(candidate, pt)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    break
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
        
        # Phase 2: swap moves if enabled
        if use_swap and not improved and n_jobs > 1:
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                for j in range(i + 1, n_jobs):
                    if deadline is not None and time.time() > deadline:
                        return best_perm, best_ms
                    candidate = list(best_perm)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    ms = _makespan_fast(candidate, pt)
                    if ms < best_ms - 1e-12:
                        best_ms = ms
                        best_perm = candidate
                        improved = True
                        break
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
    
    return best_perm, best_ms


def _tabu_search(perm, pt, deadline, tabu_tenure=7, max_iterations=5000, diversification=0.3):
    """Tabu search with priority-list representation for flow shop."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # Priority-list representation: continuous vector determines insertion order
    # Initialize weights from current permutation
    weights = np.zeros(n_jobs)
    for i, job in enumerate(perm):
        weights[job] = n_jobs - i  # Higher weight = earlier position
    
    # Tabu list: store (job, from_pos, to_pos) with expiration
    tabu_list = []
    tabu_tenure = max(5, min(15, n_jobs // 3))
    
    # Best solution tracking
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Current solution
    curr_perm = list(perm)
    curr_ms = _makespan_fast(curr_perm, pt)
    
    # Diversification: track visited solutions
    visited = set()
    visited.add(tuple(curr_perm))
    
    # Elite set for intensification
    elite_set = [(best_ms, list(best_perm))]
    elite_size = 5
    
    no_improve_count = 0
    stagnation_limit = 200
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
        
        # Expire old tabu entries
        tabu_list = [(j, fp, tp, exp) for j, fp, tp, exp in tabu_list if exp > iteration]
        
        # Generate neighbors via priority perturbation
        best_neighbor = None
        best_neighbor_ms = float('inf')
        best_neighbor_tabu = False
        
        # Try k random perturbations of the weight vector
        n_perturb = min(10, n_jobs)
        for _ in range(n_perturb):
            # Perturb weights
            new_weights = weights.copy()
            idx1 = random.randint(0, n_jobs - 1)
            idx2 = random.randint(0, n_jobs - 1)
            if idx1 == idx2:
                continue
            
            # Swap or adjust weights
            if random.random() < 0.5:
                new_weights[idx1], new_weights[idx2] = new_weights[idx2], new_weights[idx1]
            else:
                delta = random.uniform(-0.5, 0.5)
                new_weights[idx1] += delta
                new_weights[idx2] -= delta
            
            # Convert to permutation by sorting on weights (descending)
            order = np.argsort(-new_weights, kind='stable')
            candidate_perm = [int(j) for j in order]
            
            # Check if this is a tabu move
            is_tabu = False
            for j, fp, tp, exp in tabu_list:
                # Simple check: if job j moved recently
                if candidate_perm[j] != curr_perm[j]:
                    # Check if this specific move is tabu
                    for j2, fp2, tp2, exp2 in tabu_list:
                        if j2 == candidate_perm.index(j) if j in candidate_perm else -1:
                            if abs(fp2 - tp2) == 1:  # Adjacent move
                                is_tabu = True
                                break
                    if is_tabu:
                        break
            
            # Aspiration criterion: accept tabu move if it improves best
            ms = _makespan_fast(candidate_perm, pt)
            
            if is_tabu and ms >= best_ms:
                continue
            
            if ms < best_neighbor_ms:
                best_neighbor_ms = ms
                best_neighbor = candidate_perm
                best_neighbor_tabu = is_tabu
        
        if best_neighbor is None:
            # No valid neighbor found, try a larger perturbation
            new_weights = np.random.randn(n_jobs) * 0.5
            order = np.argsort(-new_weights, kind='stable')
            best_neighbor = [int(j) for j in order]
            best_neighbor_ms = _makespan_fast(best_neighbor, pt)
            best_neighbor_tabu = False
        
        # Accept the best neighbor
        # Update tabu list
        for j in range(n_jobs):
            if best_neighbor[j] != curr_perm[j]:
                # Find old and new positions
                old_pos = curr_perm.index(best_neighbor[j])
                new_pos = j
                tabu_list.append((best_neighbor[j], old_pos, new_pos, iteration + tabu_tenure))
        
        # Update current solution
        curr_perm = best_neighbor
        curr_ms = best_neighbor_ms
        weights = np.zeros(n_jobs)
        for i, job in enumerate(curr_perm):
            weights[job] = n_jobs - i
        
        # Update best
        if curr_ms < best_ms - 1e-12:
            best_ms = curr_ms
            best_perm = list(curr_perm)
            no_improve_count = 0
            
            # Update elite set
            if len(elite_set) < elite_size or curr_ms < elite_set[-1][0]:
                elite_set.append((curr_ms, list(curr_perm)))
                elite_set.sort(key=lambda x: x[0])
                if len(elite_set) > elite_size:
                    elite_set.pop()
        else:
            no_improve_count += 1
        
        # Diversification: if stagnant, jump to a random elite or random solution
        if no_improve_count >= stagnation_limit:
            if random.random() < diversification and elite_set:
                # Jump to a random elite solution
                _, elite_perm = random.choice(elite_set)
                curr_perm = list(elite_perm)
                curr_ms = _makespan_fast(curr_perm, pt)
                weights = np.zeros(n_jobs)
                for i, job in enumerate(curr_perm):
                    weights[job] = n_jobs - i
            else:
                # Random restart
                curr_perm = list(range(n_jobs))
                random.shuffle(curr_perm)
                curr_ms = _makespan_fast(curr_perm, pt)
                weights = np.zeros(n_jobs)
                for i, job in enumerate(curr_perm):
                    weights[job] = n_jobs - i
            
            no_improve_count = 0
        
        # Track visited
        visited.add(tuple(curr_perm))
        if len(visited) > 1000:
            visited.clear()
            visited.add(tuple(curr_perm))
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s limit
    
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
    initial_solutions.append((neh_perm, _makespan_fast(neh_perm, pt)))
    
    # 2. Johnson's rule
    johnson_perm = _johnsons_rule(pt)
    if johnson_perm != neh_perm:
        initial_solutions.append((johnson_perm, _makespan_fast(johnson_perm, pt)))
    
    # 3. SPT on first machine
    spt_order = list(np.argsort(pt[:, 0]))
    if spt_order != neh_perm and spt_order != johnson_perm:
        initial_solutions.append((spt_order, _makespan_fast(spt_order, pt)))
    
    # 4. LPT on last machine
    lpt_last = list(np.argsort(-pt[:, -1]))
    if lpt_last != neh_perm and lpt_last != johnson_perm and lpt_last != spt_order:
        initial_solutions.append((lpt_last, _makespan_fast(lpt_last, pt)))
    
    # Sort initial solutions by makespan
    initial_solutions.sort(key=lambda x: x[1])
    
    best_perm, best_ms = initial_solutions[0]
    
    # Phase 1: Local search on best initial solution
    if time.time() < deadline - 20.0:
        best_perm, best_ms = _local_search(best_perm, pt, deadline, use_swap=True)
    
    # Phase 2: Tabu search for diversification
    if time.time() < deadline - 5.0:
        tabu_deadline = deadline - 3.0
        tabu_perm, tabu_ms = _tabu_search(best_perm, pt, tabu_deadline)
        if tabu_ms < best_ms - 1e-12:
            best_perm = tabu_perm
            best_ms = tabu_ms
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2.0:
        best_perm, best_ms = _local_search(best_perm, pt, deadline, use_swap=False)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
