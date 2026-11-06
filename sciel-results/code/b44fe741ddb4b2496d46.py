"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    machine_times = np.zeros(n_machines)
    for j in order:
        pt = processing_times[j]
        for m in range(n_machines):
            if m == 0:
                machine_times[m] += pt[m]
            else:
                machine_times[m] = max(machine_times[m], machine_times[m-1]) + pt[m]
    return machine_times[-1]


def _makespan_fast(processing_times, order):
    """Vectorized-ish makespan computation for speed."""
    n_machines = processing_times.shape[1]
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    pts = processing_times[order]
    prev = np.zeros(n_machines)
    for j in range(n_jobs):
        curr = np.empty(n_machines)
        curr[0] = prev[0] + pts[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                curr[m] = max(prev[m], curr[m-1]) + pts[j, m]
        prev = curr
    return prev[-1]


def _neh_construct(processing_times):
    """NEH heuristic: sort by total processing time, insert each job in best position."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order_indices = list(np.argsort(-total_times))
    
    seq = [order_indices[0]]
    for j in order_indices[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan_fast(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def _local_search(processing_times, seq, time_limit=110.0):
    """Local search with insertion moves (I1) and swap moves (I2)."""
    n_jobs = len(seq)
    best_ms = _makespan_fast(processing_times, seq)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Try all insertion moves: remove job i, insert at position j
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = seq[i]
            seq_without = seq[:i] + seq[i+1:]
            for j in range(len(seq_without) + 1):
                if j == i or j == i + 1:
                    continue
                new_seq = seq_without[:j] + [job] + seq_without[j:]
                ms = _makespan_fast(processing_times, new_seq)
                if ms < best_ms - 1e-9:
                    seq = new_seq
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved and time.time() - start_time < time_limit:
            for i in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i+1, n_jobs):
                    new_seq = seq.copy()
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    ms = _makespan_fast(processing_times, new_seq)
                    if ms < best_ms - 1e-9:
                        seq = new_seq
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
    
    return seq, best_ms


def _compute_prefix_completions(perm, processing_times):
    """Compute completion time vectors for every prefix of the permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix_comps = [None] * n_jobs
    if n_jobs == 0:
        return prefix_comps
    
    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()
        
    return prefix_comps


def _eval_insert_incremental(perm, i, j, processing_times, prefix_comps):
    """
    Evaluate makespan of moving perm[i] to position j using prefix completions.
    This is an approximation that only considers the prefix up to the insertion point,
    but combined with the local search it provides a faster evaluation for nearby moves.
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(perm)
    
    if i == j:
        return _makespan_fast(processing_times, perm)
    
    # Build the new sequence
    new_perm = perm[:i] + perm[i+1:]
    if j > i:
        j -= 1
    new_perm = new_perm[:j] + [perm[i]] + new_perm[j:]
    
    # For efficiency, if the move is local, we can do a partial computation
    # But for correctness, we compute the full makespan if it's not too expensive
    # Since n_jobs is typically small in these instances, full computation is fine
    return _makespan_fast(processing_times, new_perm)


def _local_search_incremental(processing_times, seq, time_limit=110.0):
    """Local search using incremental evaluation for speed."""
    n_jobs = len(seq)
    best_ms = _makespan_fast(processing_times, seq)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Try all insertion moves
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = seq[i]
            seq_without = seq[:i] + seq[i+1:]
            for j in range(len(seq_without) + 1):
                if j == i or j == i + 1:
                    continue
                new_seq = seq_without[:j] + [job] + seq_without[j:]
                ms = _makespan_fast(processing_times, new_seq)
                if ms < best_ms - 1e-9:
                    seq = new_seq
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved and time.time() - start_time < time_limit:
            for i in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i+1, n_jobs):
                    new_seq = seq.copy()
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    ms = _makespan_fast(processing_times, new_seq)
                    if ms < best_ms - 1e-9:
                        seq = new_seq
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
    
    return seq, best_ms


def _perturb(seq, strength):
    """Perturb the sequence by making random moves."""
    n_jobs = len(seq)
    if n_jobs < 2:
        return seq
    
    n_moves = max(1, int(strength * n_jobs))
    new_seq = list(seq)
    
    for _ in range(n_moves):
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(n_jobs), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        else:
            # Insert
            i = random.randint(0, n_jobs - 1)
            job = new_seq.pop(i)
            j = random.randint(0, n_jobs - 1)
            new_seq.insert(j, job)
    
    return new_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Local search with insertion and swap moves to refine.
    3. Iterated local search with perturbation to escape local optima.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Stage 1: NEH construction
    seq = _neh_construct(processing_times)
    best_seq = list(seq)
    best_ms = _makespan_fast(processing_times, seq)
    
    if time.time() > deadline:
        return list(best_seq)
    
    # Stage 2: Local search
    seq, best_ms = _local_search(processing_times, seq, time_limit=deadline - time.time())
    best_seq = list(seq)
    
    if time.time() > deadline:
        return list(best_seq)
    
    # Stage 3: Iterated Local Search (ILS)
    rng = random.Random(42)
    current_seq = list(seq)
    current_ms = best_ms
    
    while time.time() < deadline - 10.0:  # Leave 10s buffer
        # Perturb
        strength = 0.1 + 0.2 * rng.random()
        perturbed = _perturb(current_seq, strength)
        
        # Local search from perturbed solution
        new_seq, new_ms = _local_search(processing_times, perturbed, time_limit=deadline - time.time() - 5.0)
        
        # Accept if better or with some probability (simulated annealing-like)
        if new_ms < current_ms - 1e-9:
            current_seq = new_seq
            current_ms = new_ms
            if new_ms < best_ms - 1e-9:
                best_ms = new_ms
                best_seq = list(new_seq)
        elif rng.random() < 0.3:
            # Accept worse solution with some probability to escape local optima
            current_seq = new_seq
            current_ms = new_ms
    
    return list(best_seq)
# EVOLVE-BLOCK-END
