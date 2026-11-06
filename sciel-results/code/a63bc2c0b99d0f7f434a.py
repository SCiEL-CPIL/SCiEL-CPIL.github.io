# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _calc_makespan(processing_times, order):
    """
    Calculate the makespan for a given job order efficiently.
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines, dtype=np.float64)
    for job in order:
        pt = processing_times[job]
        prev_completion = 0.0
        for m in range(n_machines):
            start_time = max(completion[m], prev_completion)
            completion[m] = start_time + pt[m]
            prev_completion = completion[m]
    return float(completion[-1])

def _compute_prefix(processing_times, seq):
    """
    prefix[m][i] = completion time of machine m after processing seq[0:i]
    """
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    prefix = np.zeros((n_machines, n_jobs + 1), dtype=np.float64)
    for i, job in enumerate(seq):
        pt = processing_times[job]
        prev_c = 0.0
        for m in range(n_machines):
            start_time = max(prefix[m][i], prev_c)
            prefix[m][i + 1] = start_time + pt[m]
            prev_c = prefix[m][i + 1]
    return prefix

def _compute_suffix(processing_times, seq):
    """
    suffix[m][i] = completion time of machine m when processing seq[i:] starting at time 0
    """
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    suffix = np.zeros((n_machines, n_jobs + 1), dtype=np.float64)
    for i in range(n_jobs - 1, -1, -1):
        completion = np.zeros(n_machines, dtype=np.float64)
        for j in range(i, n_jobs):
            pt = processing_times[seq[j]]
            prev_c = 0.0
            for m in range(n_machines):
                start_time = max(completion[m], prev_c)
                completion[m] = start_time + pt[m]
                prev_c = completion[m]
        suffix[:, i] = completion
    return suffix

def _eval_move(processing_times, seq, i, j, prefix, suffix):
    """
    Evaluate the makespan of moving job from position i to position j.
    Uses prefix and suffix arrays for efficient evaluation.
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(seq)
    
    if i == j:
        return _calc_makespan(processing_times, seq)
    
    job = seq[i]
    pt = processing_times[job]
    
    # Remove job from position i
    # New sequence: seq[0:i] + seq[i+1:j+1] + [job] + seq[j+1:]  (if i < j)
    # Or: seq[0:i] + [job] + seq[i+1:j+1] + seq[j+1:]  (if i > j)
    # Actually, let's think about this more carefully.
    
    # If we move job from i to j:
    # The jobs between i and j (exclusive) shift by one position.
    
    # Let's compute the makespan by simulating the new sequence,
    # but using prefix for the part before the move and suffix for the part after.
    
    # This is complex, so let's just compute directly for now,
    # but we can optimize later.
    
    new_seq = list(seq)
    job_to_move = new_seq.pop(i)
    new_seq.insert(j, job_to_move)
    
    return _calc_makespan(processing_times, new_seq)

def _neh_initial(processing_times, rng=None, randomize=False):
    """
    NEH heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if rng is None:
        rng = random.Random(42)
        
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    if randomize and n_jobs > 2:
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.5:
                order[i], order[i+1] = order[i+1], order[i]
                
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _calc_makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        
    return seq

def _insertion_local_search(processing_times, seq, deadline=None):
    """
    Local search using insertion moves with early termination on deadline.
    """
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_ms = _calc_makespan(processing_times, seq)
    best_ms = current_ms
    best_seq = list(seq)
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs):
            job = best_seq[i]
            seq_reduced = best_seq[:i] + best_seq[i+1:]
            
            best_pos = i
            best_insert_ms = best_ms
            
            for pos in range(n_jobs - 1):
                trial = seq_reduced[:pos] + [job] + seq_reduced[pos:]
                ms = _calc_makespan(processing_times, trial)
                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_pos = pos
                    
            if best_insert_ms < best_ms - 1e-9:
                best_seq = seq_reduced[:best_pos] + [job] + seq_reduced[best_pos:]
                best_ms = best_insert_ms
                improved = True
                if deadline is not None and time.time() > deadline:
                    return best_seq, best_ms
                    
        if not improved:
            break
            
    return best_seq, best_ms

def _simulated_annealing(processing_times, seq, deadline=None, rng=None):
    """
    Simulated Annealing to escape local minima.
    """
    if rng is None:
        rng = random.Random()
        
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_seq = list(seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    temperature = max(current_ms * 0.1, 1.0)
    cooling_rate = 0.995
    
    while (deadline is None or time.time() < deadline) and temperature > 1e-5:
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        if i == j:
            continue
            
        new_seq = list(current_seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_ms = _calc_makespan(processing_times, new_seq)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_seq = new_seq
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
                
        temperature *= cooling_rate
        
    return best_seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    best_perm = None
    best_ms = float('inf')
    
    # Phase 1: NEH variants
    if time.time() < deadline - 10.0:
        perm1 = _neh_initial(processing_times, rng, randomize=False)
        ms1 = _calc_makespan(processing_times, perm1)
        if ms1 < best_ms:
            best_ms = ms1
            best_perm = perm1
            
    if time.time() < deadline - 10.0:
        perm2 = _neh_initial(processing_times, rng, randomize=True)
        ms2 = _calc_makespan(processing_times, perm2)
        if ms2 < best_ms:
            best_ms = ms2
            best_perm = perm2
            
    if time.time() < deadline - 10.0:
        rng3 = random.Random(12345)
        perm3 = _neh_initial(processing_times, rng3, randomize=True)
        ms3 = _calc_makespan(processing_times, perm3)
        if ms3 < best_ms:
            best_ms = ms3
            best_perm = perm3
            
    # Phase 2: Local Search
    if time.time() < deadline - 5.0 and best_perm is not None:
        ls_deadline = deadline - 2.0
        best_perm, best_ms = _insertion_local_search(processing_times, best_perm, deadline=ls_deadline)
        
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 2.0 and best_perm is not None:
        sa_deadline = deadline - 1.0
        rng_sa = random.Random(99999)
        sa_perm, sa_ms = _simulated_annealing(processing_times, best_perm, deadline=sa_deadline, rng=rng_sa)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm
            
            if time.time() < deadline - 0.5:
                best_perm, best_ms = _insertion_local_search(processing_times, best_perm, deadline=deadline - 0.1)
                
    return best_perm

# EVOLVE-BLOCK-END
