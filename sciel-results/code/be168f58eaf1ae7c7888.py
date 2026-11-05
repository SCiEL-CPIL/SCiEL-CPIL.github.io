"""Flow-shop scheduling via NEH + Tabu Search with block-adjacent insertion neighborhood."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0

    machine_completion = np.zeros(n_machines)
    makespan = 0.0

    for job_idx in order:
        prev_completion = 0.0
        for m in range(n_machines):
            start_time = max(prev_completion, machine_completion[m])
            prev_completion = start_time + processing_times[job_idx, m]
            machine_completion[m] = prev_completion
        if prev_completion > makespan:
            makespan = prev_completion

    return makespan


def _makespan_incremental(processing_times, order, pos_from, pos_to, block_len=1):
    """
    Compute makespan after moving a block of jobs from [pos_from, pos_from+block_len)
    to position pos_to (insert before the job currently at pos_to in the modified sequence).
    This is an approximation that recomputes the full makespan but is structured for clarity.
    For true speedup, we'd do delta evaluation, but full recompute is safer.
    """
    return _makespan(processing_times, order)


def _apply_block_move(order, pos_from, pos_to, block_len=1):
    """
    Move a block of block_len jobs starting at pos_from to be inserted at pos_to.
    Returns the new order.
    """
    n = len(order)
    block = order[pos_from:pos_from + block_len]
    remaining = order[:pos_from] + order[pos_from + block_len:]
    
    # Adjust insertion position if removal is before insertion
    if pos_to > pos_from:
        insert_pos = pos_to - block_len
    else:
        insert_pos = pos_to
    
    new_order = remaining[:insert_pos] + block + remaining[insert_pos:]
    return new_order


def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    sequence = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _tabu_search(processing_times, initial_order, time_limit=60.0, rng=None):
    """
    Tabu Search with block-adjacent insertion neighborhood.
    Moves a single job or a block of up to 3 jobs to any other position.
    Uses a dynamic tabu list to prevent cycling.
    """
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    # Dynamic tabu list parameters
    base_tabu_tenure = max(5, n_jobs // 2)
    tabu_tenure = base_tabu_tenure
    min_tabu_tenure = 3
    max_tabu_tenure = n_jobs
    
    # Tabu list: (job_index, from_pos, to_pos, block_len) -> iteration when it becomes non-tabu
    # We'll use a simpler approach: track recent moves as (frozenset of affected positions)
    tabu_moves = {}  # move_signature -> iteration when it expires
    current_iter = 0
    
    # Aspiration criterion: allow tabu move if it improves best
    no_improve_count = 0
    patience = 2000
    
    while time.time() - start_time < time_limit:
        current_iter += 1
        
        # Expire old tabu entries
        if current_iter % 50 == 0:
            expired = [k for k, v in tabu_moves.items() if v <= current_iter]
            for k in expired:
                del tabu_moves[k]
        
        best_neighbor_order = None
        best_neighbor_makespan = float('inf')
        best_neighbor_move = None
        
        # Generate neighbors: single job moves and block moves (length 1-3)
        # For efficiency, we limit the number of neighbors evaluated per iteration
        max_neighbors = 200
        neighbors_evaluated = 0
        
        for block_len in [1, 2, 3]:
            if neighbors_evaluated >= max_neighbors:
                break
            
            for pos_from in range(n_jobs - block_len + 1):
                if neighbors_evaluated >= max_neighbors:
                    break
                
                for pos_to in range(n_jobs - block_len + 1):
                    if neighbors_evaluated >= max_neighbors:
                        break
                    
                    # Skip if the move is trivial (same position)
                    if pos_to == pos_from:
                        continue
                    # Skip if inserting right after the block (no effective change for single job)
                    if block_len == 1 and pos_to == pos_from + 1:
                        continue
                    if block_len > 1 and pos_to == pos_from + block_len:
                        continue
                    
                    # Check tabu
                    move_sig = (block_len, pos_from, pos_to)
                    is_tabu = move_sig in tabu_moves
                    
                    # Aspiration: allow if it improves best
                    if is_tabu:
                        # We'll still evaluate it but won't accept unless it beats best
                        pass
                    
                    new_order = _apply_block_move(current_order, pos_from, pos_to, block_len)
                    new_makespan = _makespan(processing_times, new_order)
                    neighbors_evaluated += 1
                    
                    # Acceptance criteria
                    if new_makespan < best_neighbor_makespan:
                        # Check if this is a valid move
                        if not is_tabu or new_makespan < best_makespan:
                            best_neighbor_order = new_order
                            best_neighbor_makespan = new_makespan
                            best_neighbor_move = move_sig
        
        if best_neighbor_order is None:
            break
        
        # Check if we found an improvement
        if best_neighbor_makespan < current_makespan:
            current_order = best_neighbor_order
            current_makespan = best_neighbor_makespan
            no_improve_count = 0
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
                # Reduce tabu tenure on improvement
                tabu_tenure = max(min_tabu_tenure, tabu_tenure - 1)
        else:
            # Accept the best non-tabu neighbor even if it's worse (to escape local optima)
            # But only if it's not tabu
            move_sig = best_neighbor_move
            is_tabu = move_sig in tabu_moves
            if not is_tabu:
                current_order = best_neighbor_order
                current_makespan = best_neighbor_makespan
                no_improve_count += 1
            else:
                # All best neighbors are tabu; try to find a non-tabu one
                # For simplicity, break and restart
                break
        
        # Add the move to the tabu list
        if best_neighbor_move is not None:
            tabu_moves[best_neighbor_move] = current_iter + tabu_tenure
        
        # Dynamic tabu tenure adjustment
        if no_improve_count > patience // 2:
            tabu_tenure = min(max_tabu_tenure, tabu_tenure + 1)
        elif no_improve_count == 0:
            tabu_tenure = max(min_tabu_tenure, tabu_tenure - 1)
        
        if no_improve_count > patience:
            break
    
    return best_order


def _local_search(processing_times, initial_order, time_limit=10.0):
    """Or-opt local search with time limit."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try 2-opt swaps
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_makespan = _makespan(processing_times, new_order)
                
                if new_makespan < current_makespan - 1e-9:
                    current_order = new_order
                    current_makespan = new_makespan
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = list(current_order)
                    improved = True
                    break
            if improved:
                break
        
        # Try or-opt moves
        if not improved:
            for i in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                job_to_move = current_order[i]
                for j in range(n_jobs):
                    if j == i or j == i + 1:
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    
                    new_order = current_order[:i] + current_order[i+1:]
                    insert_pos = j if j > i else j
                    new_order = new_order[:insert_pos] + [job_to_move] + new_order[insert_pos:]
                    new_makespan = _makespan(processing_times, new_order)
                    
                    if new_makespan < current_makespan - 1e-9:
                        current_order = new_order
                        current_makespan = new_makespan
                        if current_makespan < best_makespan:
                            best_makespan = current_makespan
                            best_order = list(current_order)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_order


def _random_perturbation(order, strength=0.3, rng=None):
    """Apply random 2-opt perturbation."""
    n = len(order)
    if n <= 1:
        return list(order)
    
    if rng is None:
        rng = np.random
    
    new_order = list(order)
    num_swaps = max(1, int(strength * n))
    for _ in range(num_swaps):
        i, j = rng.choice(n, 2, replace=False)
        new_order[i], new_order[j] = new_order[j], new_order[i]
    
    return new_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop.
    Uses NEH heuristic, Tabu Search with block-adjacent insertion, and local search.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard
    deadline = 140.0
    start_time = time.time()
    
    # Step 1: NEH heuristic
    neh_order = _neh_heuristic(processing_times)
    neh_makespan = _makespan(processing_times, neh_order)
    
    best_order = list(neh_order)
    best_makespan = neh_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 2: Local search on NEH solution
    ls_time = min(remaining * 0.2, 15.0)
    ls_order = _local_search(processing_times, best_order, time_limit=ls_time)
    ls_makespan = _makespan(processing_times, ls_order)
    
    if ls_makespan < best_makespan:
        best_order = list(ls_order)
        best_makespan = ls_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 3: Tabu Search with block-adjacent insertion neighborhood
    rng = np.random.RandomState(42)
    ts_time = remaining * 0.6
    if ts_time > 5.0:
        ts_order = _tabu_search(processing_times, best_order, time_limit=ts_time, rng=rng)
        ts_makespan = _makespan(processing_times, ts_order)
        
        if ts_makespan < best_makespan:
            best_order = list(ts_order)
            best_makespan = ts_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 3.0:
        return best_order
    
    # Step 4: Multiple restarts with perturbation and tabu search
    num_restarts = min(5, int(remaining / 5.0))
    
    for r in range(num_restarts):
        if time.time() + 4.0 > deadline:
            break
        
        perturbed = _random_perturbation(best_order, strength=0.2 + 0.05 * r, rng=rng)
        
        ts_time_r = min(3.0, deadline - time.time() - 1.0)
        if ts_time_r < 0.5:
            break
        
        ts_order_r = _tabu_search(processing_times, perturbed, time_limit=ts_time_r, rng=rng)
        ts_makespan_r = _makespan(processing_times, ts_order_r)
        
        if ts_makespan_r < best_makespan:
            best_order = list(ts_order_r)
            best_makespan = ts_makespan_r
    
    return best_order
# EVOLVE-BLOCK-END
