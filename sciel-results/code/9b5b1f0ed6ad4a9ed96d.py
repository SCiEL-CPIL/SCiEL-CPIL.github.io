"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    for job in perm:
        row = processing_times[job]
        completion[0] += row[0]
        for m in range(1, n_machines):
            c_prev = completion[m - 1]
            c_curr = completion[m]
            if c_curr < c_prev:
                c_curr = c_prev
            completion[m] = c_curr + row[m]
    return completion[-1]


def _v_shaped_heuristic(processing_times):
    """
    V-shaped heuristic for flow shop:
    Sort jobs by total processing time.
    Jobs with large totals go to the ends (front and back),
    jobs with small totals go to the middle.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time (ascending)
    sorted_indices = list(np.argsort(totals))
    
    # Build V-shape: alternate placing next largest at front or back
    # Start with smallest in middle, largest at ends
    sequence = []
    front = []
    back = []
    
    # Sort in descending order for placement
    desc_indices = list(np.argsort(-totals))
    
    mid = n_jobs // 2
    # Place jobs: first half go to front (in reverse order so largest is at very front),
    # second half go to back
    # Actually, let's do: take jobs in descending total order
    # Place them alternately at front and back
    
    assigned = [False] * n_jobs
    seq = [0] * n_jobs
    pos_front = 0
    pos_back = n_jobs - 1
    
    for job in desc_indices:
        if pos_front <= pos_back:
            if random.random() < 0.5:
                seq[pos_front] = job
                pos_front += 1
            else:
                seq[pos_back] = job
                pos_back -= 1
        else:
            seq[pos_back] = job
            pos_back -= 1
    
    return seq


def _neh_heuristic(processing_times):
    """NEH heuristic: sort by total processing time, insert each job at best position."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _tabu_search(processing_times, sequence, deadline):
    """
    Tabu Search with dynamic tabu list and diversification.
    Uses insertion moves (remove job i, insert at position j).
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Tabu list: store (job, position) pairs recently used
    tabu_list = []
    tabu_tenure = max(5, n_jobs // 2)
    
    # Parameters
    no_improve_count = 0
    max_no_improve = 50
    diversity_threshold = 100
    
    start_time = time.time()
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Check deadline
        if time.time() >= deadline:
            break
        
        # Determine if we need diversification
        if no_improve_count > max_no_improve:
            # Diversification: try a random restart or large perturbation
            if random.random() < 0.3:
                # Random perturbation: swap several pairs
                n_perturb = max(2, n_jobs // 5)
                for _ in range(n_perturb):
                    i, j = random.sample(range(n_jobs), 2)
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                current_makespan = _calc_makespan(processing_times, current_seq)
            no_improve_count = 0
            tabu_list.clear()
        
        # Find best non-tabu move
        best_move = None
        best_move_ms = float('inf')
        best_move_i = -1
        best_move_j = -1
        
        # Sample a subset of moves for efficiency
        if n_jobs <= 30:
            # Try all moves
            move_candidates = [(i, j) for i in range(n_jobs) for j in range(n_jobs) if i != j]
            random.shuffle(move_candidates)
        else:
            # Sample moves
            n_samples = min(200, n_jobs * (n_jobs - 1))
            move_candidates = []
            for _ in range(n_samples):
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i != j:
                    move_candidates.append((i, j))
        
        # Limit number of moves to evaluate per iteration
        max_moves_per_iter = 50
        if len(move_candidates) > max_moves_per_iter:
            move_candidates = move_candidates[:max_moves_per_iter]
        
        for i, j in move_candidates:
            # Check if move is tabu
            if (i, j) in tabu_list:
                continue
            
            # Apply move: remove job at i, insert at j
            job = current_seq[i]
            new_seq = current_seq[:i] + current_seq[i+1:]
            # Adjust j if j > i
            insert_pos = j - 1 if j > i else j
            if insert_pos < 0:
                insert_pos = 0
            if insert_pos > len(new_seq):
                insert_pos = len(new_seq)
            new_seq.insert(insert_pos, job)
            
            new_ms = _calc_makespan(processing_times, new_seq)
            
            # Aspiration criterion: accept tabu move if it improves best
            # (already skipped tabu moves, but we could add aspiration)
            
            if new_ms < best_move_ms:
                best_move_ms = new_ms
                best_move_i = i
                best_move_j = j
                best_move = new_seq
        
        if best_move is None:
            # No non-tabu moves found, accept any move
            break
        
        # Update tabu list
        tabu_list.append((best_move_i, best_move_j))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Accept the move
        current_seq = best_move
        current_makespan = best_move_ms
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = list(current_seq)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Dynamic tabu tenure adjustment
        if no_improve_count > 20:
            tabu_tenure = max(3, tabu_tenure - 1)
        elif no_improve_count == 0:
            tabu_tenure = min(n_jobs, tabu_tenure + 1)
        
        # Time check
        if time.time() >= deadline:
            break
    
    return best_seq, best_makespan


def _local_search_insertion(processing_times, sequence, deadline):
    """
    Insertion local search: try moving a job to every other position.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            
            best_j = -1
            best_ms = current_makespan
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Remove job at i, insert at j
                new_seq = current_seq[:i] + current_seq[i+1:]
                insert_idx = j - 1 if j > i else j
                if insert_idx < 0:
                    insert_idx = 0
                if insert_idx > len(new_seq):
                    insert_idx = len(new_seq)
                new_seq.insert(insert_idx, job)
                
                new_ms = _calc_makespan(processing_times, new_seq)
                
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_j = j
            
            if best_j >= 0:
                # Apply the best move
                new_seq = current_seq[:i] + current_seq[i+1:]
                insert_idx = best_j - 1 if best_j > i else best_j
                if insert_idx < 0:
                    insert_idx = 0
                if insert_idx > len(new_seq):
                    insert_idx = len(new_seq)
                new_seq.insert(insert_idx, job)
                current_seq = new_seq
                current_makespan = best_ms
                improved = True
                break
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with adaptive temperature and insertion moves.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.15
    min_T = 1e-4
    cooling_rate = 0.998
    
    while time.time() < deadline and T > min_T:
        if n_jobs < 2:
            break
        
        if random.random() < 0.7:
            # Insertion move
            i = random.randint(0, n_jobs - 1)
            job = current_seq[i]
            new_seq = current_seq[:i] + current_seq[i+1:]
            j = random.randint(0, len(new_seq) - 1)
            new_seq.insert(j, job)
        else:
            # Swap move
            i, j = random.sample(range(n_jobs), 2)
            new_seq = current_seq[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        new_makespan = _calc_makespan(processing_times, new_seq)
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Phase 1: Generate multiple initial solutions
    candidates = []
    
    # NEH heuristic
    neh_seq = _neh_heuristic(processing_times)
    candidates.append(neh_seq)
    
    # V-shaped heuristic
    v_seq = _v_shaped_heuristic(processing_times)
    candidates.append(v_seq)
    
    # Random permutations
    for _ in range(3):
        rand_seq = list(range(n_jobs))
        random.shuffle(rand_seq)
        candidates.append(rand_seq)
    
    # Evaluate all candidates and pick the best
    best_seq = candidates[0]
    best_makespan = _calc_makespan(processing_times, best_seq)
    for seq in candidates[1:]:
        ms = _calc_makespan(processing_times, seq)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = seq
    
    # Phase 2: Local search on best initial solution
    if time.time() < deadline - 10.0:
        ls_seq, ls_ms = _local_search_insertion(processing_times, best_seq, deadline)
        if ls_ms < best_makespan:
            best_seq, best_makespan = ls_seq, ls_ms
    
    # Phase 3: Tabu search
    if time.time() < deadline - 15.0:
        ts_seq, ts_ms = _tabu_search(processing_times, best_seq, deadline)
        if ts_ms < best_makespan:
            best_seq, best_makespan = ts_seq, ts_ms
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 10.0:
        sa_seq, sa_ms = _simulated_annealing(processing_times, best_seq, deadline)
        if sa_ms < best_makespan:
            best_seq, best_makespan = sa_seq, sa_ms
    
    # Phase 5: Final local search polish
    if time.time() < deadline - 5.0:
        final_seq, final_ms = _local_search_insertion(processing_times, best_seq, deadline)
        if final_ms < best_makespan:
            best_seq, best_makespan = final_seq, final_ms
    
    # Phase 6: Another tabu search pass with different parameters
    if time.time() < deadline - 10.0:
        ts2_seq, ts2_ms = _tabu_search(processing_times, best_seq, deadline)
        if ts2_ms < best_makespan:
            best_seq, best_makespan = ts2_seq, ts2_ms
    
    # Phase 7: Final polish
    if time.time() < deadline - 3.0:
        polish_seq, polish_ms = _local_search_insertion(processing_times, best_seq, deadline)
        if polish_ms < best_makespan:
            best_seq, best_makespan = polish_seq, polish_ms
    
    return best_seq


# EVOLVE-BLOCK-END
