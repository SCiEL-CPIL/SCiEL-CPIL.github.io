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


def _total_flow_time(perm, pt):
    """Compute total flow time (sum of completion times of all jobs on last machine)."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt_perm = pt[np.asarray(perm, dtype=np.int64)]
    
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    if n_machines > 1:
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + pt_perm[0, m]
            for j in range(1, n_jobs):
                c_prev_m = C[j-1, m]
                c_curr_m1 = C[j, m-1]
                C[j, m] = (c_curr_m1 if c_curr_m1 > c_prev_m else c_prev_m) + pt_perm[j, m]
    
    return np.sum(C[:, -1])


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


def _pareto_local_search(perm, pt, deadline, max_iterations=200):
    """
    Multi-objective local search maintaining a Pareto front of (makespan, flow_time).
    Uses non-dominated sorting and crowding distance to select promising neighbors.
    """
    n_jobs = len(perm)
    
    # Initialize Pareto front with the starting solution
    start_ms = _makespan_fast(perm, pt)
    start_ft = _total_flow_time(perm, pt)
    
    # Pareto front: list of (makespan, flow_time, permutation)
    pareto_front = [(start_ms, start_ft, list(perm))]
    
    best_ms = start_ms
    best_perm = list(perm)
    
    # Track visited to avoid cycling
    visited = set()
    visited.add(tuple(perm))
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Select a solution from the Pareto front using crowding distance
        if len(pareto_front) == 1:
            selected = pareto_front[0]
        else:
            # Compute crowding distance
            pareto_front.sort(key=lambda x: x[0])  # Sort by makespan
            n_front = len(pareto_front)
            crowding = [0.0] * n_front
            
            # Boundary points get infinite crowding
            crowding[0] = float('inf')
            crowding[-1] = float('inf')
            
            # For each objective, compute distances
            for obj_idx in [0, 1]:
                obj_values = [sol[obj_idx] for sol in pareto_front]
                obj_range = max(obj_values) - min(obj_values)
                if obj_range == 0:
                    continue
                for i in range(1, n_front - 1):
                    crowding[i] += (obj_values[i+1] - obj_values[i-1]) / obj_range
            
            # Select with highest crowding distance (with some randomness)
            max_crowd = max(crowding)
            candidates = [i for i in range(n_front) if crowding[i] > max_crowd * 0.8]
            selected_idx = random.choice(candidates)
            selected = pareto_front[selected_idx]
        
        sel_ms, sel_ft, sel_perm = selected
        
        # Generate neighbors via insertion moves
        n_jobs = len(sel_perm)
        best_neighbor = None
        best_neighbor_ms = float('inf')
        best_neighbor_ft = float('inf')
        
        # Try a subset of random insertion moves
        n_moves = min(20, n_jobs * 3)
        for _ in range(n_moves):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i == j:
                continue
            
            job = sel_perm[i]
            candidate = sel_perm[:i] + sel_perm[i+1:]
            insert_pos = j if j < i else j - 1
            candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
            
            if tuple(candidate) in visited:
                continue
            
            ms = _makespan_fast(candidate, pt)
            ft = _total_flow_time(candidate, pt)
            
            # Check if this candidate is non-dominated by the current front
            is_dominated = False
            for front_ms, front_ft, _ in pareto_front:
                if front_ms <= ms and front_ft <= ft and (front_ms < ms or front_ft < ft):
                    is_dominated = True
                    break
            
            if is_dominated:
                continue
            
            # Remove dominated solutions from the front
            new_front = []
            for front_ms, front_ft, front_perm in pareto_front:
                if not (ms <= front_ms and ft <= front_ft and (ms < front_ms or ft < front_ft)):
                    new_front.append((front_ms, front_ft, front_perm))
            
            new_front.append((ms, ft, list(candidate)))
            
            if ms < best_neighbor_ms:
                best_neighbor_ms = ms
                best_neighbor_ft = ft
                best_neighbor = candidate
                best_front = new_front
        
        if best_neighbor is not None:
            # Update Pareto front
            pareto_front = best_front
            visited.add(tuple(best_neighbor))
            
            # Track best makespan
            if best_neighbor_ms < best_ms:
                best_ms = best_neighbor_ms
                best_perm = list(best_neighbor)
            
            # Limit front size
            if len(pareto_front) > 50:
                # Keep top 50 by makespan
                pareto_front.sort(key=lambda x: x[0])
                pareto_front = pareto_front[:50]
            
            # Limit visited set
            if len(visited) > 5000:
                visited.clear()
                visited.add(tuple(best_perm))
        
        else:
            # No improvement found, try a swap move
            if n_jobs > 1:
                i = random.randint(0, n_jobs - 2)
                j = random.randint(i + 1, n_jobs - 1)
                candidate = list(sel_perm)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                if tuple(candidate) not in visited:
                    ms = _makespan_fast(candidate, pt)
                    ft = _total_flow_time(candidate, pt)
                    
                    is_dominated = False
                    for front_ms, front_ft, _ in pareto_front:
                        if front_ms <= ms and front_ft <= ft and (front_ms < ms or front_ft < ft):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        new_front = []
                        for front_ms, front_ft, front_perm in pareto_front:
                            if not (ms <= front_ms and ft <= front_ft and (ms < front_ms or ft < front_ft)):
                                new_front.append((front_ms, front_ft, front_perm))
                        
                        new_front.append((ms, ft, list(candidate)))
                        pareto_front = new_front
                        visited.add(tuple(candidate))
                        
                        if ms < best_ms:
                            best_ms = ms
                            best_perm = list(candidate)
    
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
    
    # Phase 2: Multi-objective Pareto local search for diversification
    if time.time() < deadline - 10.0:
        pareto_deadline = deadline - 8.0
        pareto_perm, pareto_ms = _pareto_local_search(best_perm, pt, pareto_deadline)
        if pareto_ms < best_ms - 1e-12:
            best_perm = pareto_perm
            best_ms = pareto_ms
    
    # Phase 3: Tabu search for further diversification
    if time.time() < deadline - 5.0:
        tabu_deadline = deadline - 3.0
        tabu_perm, tabu_ms = _tabu_search(best_perm, pt, tabu_deadline)
        if tabu_ms < best_ms - 1e-12:
            best_perm = tabu_perm
            best_ms = tabu_ms
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 2.0:
        best_perm, best_ms = _local_search(best_perm, pt, deadline, use_swap=False)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
