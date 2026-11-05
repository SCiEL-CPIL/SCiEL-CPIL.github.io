"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a PFS-specific TSP decomposition.
    Fixes the first job (highest total processing time) and optimizes the rest via TSP
    on incremental makespan contributions.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    def makespan(seq):
        """Compute makespan for a given sequence of job indices."""
        if len(seq) == 0:
            return 0.0
        completion = np.zeros(n_machines)
        for job in seq:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]
    
    # Pre-compute the N x N matrix of incremental makespan contributions.
    # cost[i, j] = incremental makespan when job j is appended after job i.
    # We approximate this by the completion time of job j on the last machine
    # minus the completion time of job i on the last machine, assuming the
    # rest of the sequence is "frozen" or by a simpler proxy.
    # Actually, for a TSP-like heuristic, we want a cost that reflects the
    # "delay" or "load" added. A good proxy is the total processing time of j
    # plus the max waiting time it would incur if i were the previous job.
    # However, computing exact incremental costs for all pairs is O(N^2 * M).
    # Let's compute a matrix where cost[i, j] is the makespan increase if j
    # is placed immediately after i in a sequence of just [i, j].
    # This is a valid lower bound / proxy for the incremental cost.
    
    # Compute completion times for single jobs
    # For a pair (i, j), the completion time of j is:
    # C_j = max(C_i, C_{j, m-1}) + p_{j, m}
    # This is complex. Let's use a simpler, well-known heuristic for TSP on FSP:
    # The cost of moving j after i is often approximated by the processing time of j
    # plus the idle time it causes.
    
    # Let's stick to the strategy: "edge weights are approximated by the incremental
    # makespan contribution of inserting job j after job i (computed via a pre-calculated
    # N x N matrix)".
    
    # To compute the exact incremental cost of inserting j after i in a context is hard
    # without the context. But for a TSP-like construction, we can use the cost of the
    # sequence [i, j] minus the cost of [i].
    # Cost([i]) = max(p_i0, p_i1, ..., p_iM) + p_iM? No, makespan of [i] is just the sum of p_i along the path.
    # Actually, makespan of [i] is simply the sum of processing times of i on all machines? No.
    # In a flow shop, job i starts at M1 at time 0. Finishes M1 at p_i0. Starts M2 at p_i0. Finishes M2 at p_i0 + p_i1.
    # So makespan of [i] is sum(p_i).
    
    # Makespan of [i, j]:
    # Job i finishes at sum(p_i).
    # Job j starts M1 at sum(p_i). Finishes M1 at sum(p_i) + p_j0.
    # Job j starts M2 at max(sum(p_i) + p_j0, sum(p_i) + p_i1)? No.
    # Job j can start M2 when M2 is free (after job i finishes M2) AND job j finishes M1.
    # Job i finishes M2 at p_i0 + p_i1.
    # Job j finishes M1 at sum(p_i) + p_j0.
    # So job j starts M2 at max(p_i0 + p_i1, sum(p_i) + p_j0).
    
    # Let's precompute the "tail" and "head" or just use the pairwise makespan increment.
    
    # cost[i, j] = makespan([i, j]) - makespan([i])
    
    cost_matrix = np.zeros((n_jobs, n_jobs))
    for i in range(n_jobs):
        ms_i = makespan([i])
        for j in range(n_jobs):
            if i == j:
                cost_matrix[i, j] = 0
            else:
                ms_ij = makespan([i, j])
                cost_matrix[i, j] = ms_ij - ms_i
    
    # Strategy:
    # 1. Fix the first job as the one with the highest total processing time.
    #    This is a strong heuristic for FSP (NEH starts with the largest job).
    # 2. Treat the remaining jobs as a TSP where we start at the fixed job
    #    and must visit all other jobs exactly once, minimizing the sum of
    #    edge costs.
    
    # However, the sum of pairwise costs is not exactly the makespan of the whole sequence.
    # But it is a good surrogate.
    
    # We will use a Nearest Neighbor heuristic to construct a path, then 2-opt to improve it.
    
    def tsp_nearest_neighbor(start_node, nodes_to_visit, cost_mat):
        """Construct a TSP path using Nearest Neighbor."""
        path = [start_node]
        unvisited = set(nodes_to_visit)
        
        while unvisited:
            current = path[-1]
            # Find nearest unvisited node
            best_node = None
            best_cost = float('inf')
            for node in unvisited:
                c = cost_mat[current, node]
                if c < best_cost:
                    best_cost = c
                    best_node = node
            
            if best_node is None:
                break
                
            path.append(best_node)
            unvisited.remove(best_node)
            
        return path
    
    def tsp_2opt(path, cost_mat, time_budget):
        """Improve a TSP path using 2-opt."""
        n = len(path)
        if n <= 2:
            return path
        
        # Precompute path cost
        def path_cost(p):
            c = 0.0
            for k in range(len(p) - 1):
                c += cost_mat[p[k], p[k+1]]
            return c
        
        best_path = list(path)
        best_cost = path_cost(best_path)
        
        while time.time() < time_budget:
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # 2-opt move: reverse segment [i+1 : j+1]
                    # New edges: (p[i], p[j+1]) and (p[j], p[i+1]) if j+1 < n
                    # Wait, standard 2-opt on a path (not cycle):
                    # If we reverse segment [i : j], the edges affected are:
                    # (p[i-1], p[i]) and (p[j], p[j+1]) become (p[i-1], p[j]) and (p[i], p[j+1])
                    # Let's use the standard reversal of [i : j] inclusive.
                    
                    # Indices in path: 0 to n-1
                    # Reverse path[i : j+1]
                    
                    # Old cost parts:
                    # Edge (i-1, i) if i > 0
                    # Edge (j, j+1) if j < n-1
                    # Internal edges in [i : j] are reversed but sum is same if symmetric?
                    # Cost matrix is NOT symmetric.
                    # So we must recompute or carefully update.
                    # Given the complexity and small N, let's just recompute path cost for the candidate.
                    
                    candidate = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                    c = path_cost(candidate)
                    delta = c - best_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-10:
                i, j = best_i, best_j
                best_path[i:j+1] = best_path[i:j+1][::-1]
                best_cost += best_delta
                improved = True
            else:
                break
        
        return best_path
    
    # Generate initial solutions
    candidates = []
    
    # 1. TSP-based construction
    # Fix first job
    totals = processing_times.sum(axis=1)
    first_job = int(np.argmax(totals))
    remaining_jobs = [j for j in range(n_jobs) if j != first_job]
    
    # Nearest Neighbor from first_job
    path_nn = tsp_nearest_neighbor(first_job, remaining_jobs, cost_matrix)
    candidates.append(path_nn)
    
    # 2. NEH heuristic (as a strong baseline)
    def neh_heuristic():
        order = list(np.argsort(-totals))
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq
    
    candidates.append(neh_heuristic())
    
    # 3. Palmer heuristic
    def palmer_heuristic():
        n = n_jobs
        priorities = []
        for job in range(n_jobs):
            p_first = processing_times[job, 0]
            p_last = processing_times[job, -1]
            priorities.append(n * p_first - p_last)
        priorities = np.array(priorities)
        return list(np.argsort(-priorities))
    
    candidates.append(palmer_heuristic())
    
    # 4. LPT
    candidates.append(list(np.argsort(-totals)))
    
    # 5. Random permutations
    n_random = min(3, max(1, n_jobs // 10))
    for _ in range(n_random):
        seq = list(range(n_jobs))
        random.shuffle(seq)
        candidates.append(seq)
    
    def tabu_search(seq, time_budget):
        """
        Tabu search with block-adjacent insertion neighborhood.
        Moves a block of 1-3 jobs from one position to another.
        Uses a dynamic tabu list to prevent cycling.
        """
        seq = list(seq)
        n = len(seq)
        if n <= 2:
            return seq, makespan(seq)
        
        current_ms = makespan(seq)
        best_ms = current_ms
        best_seq = list(seq)
        
        # Dynamic tabu list: stores (from_pos, to_pos) moves
        tabu_list = []
        tabu_tenure = max(10, n // 3)  # Dynamic tenure based on problem size
        
        # Aspiration criterion: allow tabu move if it improves best solution
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < time_budget:
            # Check time periodically
            if no_improve_count > max_no_improve:
                # Reset tabu list to escape local optimum
                tabu_list.clear()
                no_improve_count = 0
                # Slightly perturb to escape
                if n > 2:
                    i = random.randint(0, n - 2)
                    j = random.randint(i + 1, n - 1)
                    seq[i], seq[j] = seq[j], seq[i]
                    current_ms = makespan(seq)
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_seq = list(seq)
                    continue
            
            # Generate neighborhood: move block of size 1-3 from pos i to pos j
            best_move = None
            best_move_ms = float('inf')
            best_move_from = -1
            best_move_to = -1
            best_move_block_size = 1
            
            # Limit neighborhood size for larger problems
            block_sizes = [1, 2, 3]
            if n > 100:
                block_sizes = [1, 2]
            if n > 200:
                block_sizes = [1]
            
            for block_size in block_sizes:
                for i in range(n - block_size + 1):
                    # Skip if this move is tabu (check most recent tabu entries)
                    # We'll check tabu after finding the best non-tabu move
                    for j in range(n - block_size + 1):
                        if abs(i - j) <= block_size:
                            continue  # Same or adjacent position, no change
                        if j < i:
                            continue  # Only consider forward moves to reduce redundancy
                        # Actually, we should consider all moves. Let me reconsider.
                        # For efficiency, let's just sample a subset for large n
                        if n > 50 and random.random() > 0.1:
                            continue
                        
                        # Extract block
                        block = seq[i:i+block_size]
                        # Remove block and insert at position j
                        # Adjust j if j > i
                        new_seq = seq[:i] + seq[i+block_size:]
                        if j > i:
                            insert_pos = j - block_size
                        else:
                            insert_pos = j
                        if insert_pos < 0:
                            insert_pos = 0
                        if insert_pos > len(new_seq):
                            insert_pos = len(new_seq)
                        
                        new_seq = new_seq[:insert_pos] + block + new_seq[insert_pos:]
                        
                        ms = makespan(new_seq)
                        
                        # Check tabu status
                        is_tabu = False
                        for tabu_move in tabu_list[-10:]:  # Check recent moves
                            if tabu_move[0] == i and tabu_move[1] == j:
                                is_tabu = True
                                break
                        
                        # Aspiration: allow if it improves best
                        if is_tabu and ms >= best_ms:
                            continue
                        
                        if ms < best_move_ms:
                            best_move_ms = ms
                            best_move_from = i
                            best_move_to = j
                            best_move_block_size = block_size
                            best_move_seq = new_seq
                            best_move_is_tabu = is_tabu
            
            if best_move is None or best_move_from == -1:
                break
            
            # Apply the best move
            seq = best_move_seq
            current_ms = best_move_ms
            
            # Update tabu list
            tabu_list.append((best_move_from, best_move_to, best_move_block_size))
            if len(tabu_list) > tabu_tenure * 2:
                tabu_list.pop(0)
            
            # Update best
            if current_ms < best_ms - 1e-10:
                best_ms = current_ms
                best_seq = list(seq)
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return best_seq, best_ms
    
    best_seq = None
    best_ms = float('inf')
    
    # Apply local search to candidates
    for i, seq in enumerate(candidates):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        
        # For the NEH candidate (usually best), apply tabu search with more time
        if i == 1:  # NEH
            time_budget = min(remaining * 0.4, 40.0)
            seq_ls, ms = tabu_search(seq, time_budget)
            if ms < best_ms:
                best_ms = ms
                best_seq = seq_ls
        elif i == 0:  # TSP-based
            # Apply tabu search with less time
            time_budget = min(remaining * 0.2, 20.0)
            seq_ls, ms = tabu_search(seq, time_budget)
            if ms < best_ms:
                best_ms = ms
                best_seq = seq_ls
        else:
            # Just evaluate directly for other candidates
            ms = makespan(seq)
            if ms < best_ms:
                best_ms = ms
                best_seq = list(seq)
    
    # Use remaining time for additional tabu search iterations on the best solution
    if best_seq is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            best_seq, best_ms = tabu_search(best_seq, remaining)
    
    if best_seq is None:
        best_seq = candidates[0]
        
    return list(best_seq)
# EVOLVE-BLOCK-END
