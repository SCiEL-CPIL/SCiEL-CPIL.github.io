"""Combined program: NEH + Simulated Annealing + Local Search Polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Optimized for speed: avoids list comprehensions where possible, uses local vars.
    """
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Initialize for the first job in the sequence
    first_job = order[0]
    # For the first job, C(0, m) = sum(P[0, 0..m])
    curr_row = np.cumsum(P[first_job])
    
    for idx in range(1, n_jobs):
        j = order[idx]
        prev_row = curr_row
        # Compute new curr_row for job j
        # C(j, 0) = C(j-1, 0) + P(j, 0)
        # C(j, m) = max(C(j-1, m), C(j, m-1)) + P(j, m)
        new_row = np.empty(n_machines)
        new_row[0] = prev_row[0] + P[j, 0]
        if n_machines > 1:
            # Unroll loop for small M if needed, but standard loop is fine
            for m in range(1, n_machines):
                # max(prev_row[m], new_row[m-1]) + P[j, m]
                val = prev_row[m]
                left = new_row[m-1]
                if left > val:
                    val = left
                new_row[m] = val + P[j, m]
        curr_row = new_row
    
    return curr_row[-1]

def _neh_heuristic(P, sort_key='total'):
    """
    NEH heuristic for flow shop scheduling.
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
    if sort_key == 'total':
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
        sorted_indices = np.argsort(-key_vals)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, job)
        
    return current_seq

def _simulated_annealing(order, P, deadline, rng):
    """
    Simulated Annealing to escape local optima.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    best_order = list(current_order)
    best_ms = current_ms
    
    # Initial temperature: heuristic based on problem size
    # If makespan is large, temp should be large.
    initial_temp = max(current_ms * 0.1, 10.0)
    temp = initial_temp
    cooling_rate = 0.9995  # Slow cooling to allow more exploration
    
    while time.time() < deadline - 5.0:  # Leave 5s for final polish
        # Generate neighbor: Swap two random jobs
        i = rng.integers(0, n_jobs)
        j = rng.integers(0, n_jobs)
        if i == j:
            continue
        
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_ms = _makespan(new_order, P)
        
        delta = new_ms - current_ms
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0 or (temp > 1e-6 and rng.random() < np.exp(-delta / temp)):
            current_order = new_order
            current_ms = new_ms
            
            if new_ms < best_ms:
                best_ms = new_ms
                best_order = list(new_order)
        
        temp *= cooling_rate
        
        # If temp is very low, we are essentially doing local search
        # We could break early if no improvement in many steps, but time check is safer
    
    return best_order

def _local_search_polish(order, P, deadline):
    """
    Final local search polish using Adjacent Swaps and Insertions.
    Uses first-improvement strategy to be fast.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    improved = True
    # Limit passes to avoid infinite loops in case of numerical issues, though time check is primary
    max_passes = 100
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
            
        improved = False
        
        # 1. Adjacent Swaps
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                return current_order
            
            # Swap i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan(new_order, P)
            
            if new_ms < current_ms - 1e-9:
                current_order = new_order
                current_ms = new_ms
                improved = True
                # Restart swap loop from beginning as order changed
                break
        
        if not improved:
            # 2. Insertion Moves (Or-opt style: remove one, insert elsewhere)
            # This is more expensive O(N^2 * M) per pass, so do it less frequently
            # or only if time permits. Given N is likely small, it's okay.
            for i in range(n_jobs):
                if time.time() > deadline:
                    return current_order
                
                job = current_order[i]
                # Remove job
                base_order = current_order[:i] + current_order[i+1:]
                
                # Try inserting at all positions
                best_pos = i
                best_ms = current_ms
                
                for pos in range(len(base_order) + 1):
                    if time.time() > deadline:
                        break
                    candidate = base_order[:pos] + [job] + base_order[pos:]
                    ms = _makespan(candidate, P)
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        best_pos = pos
                
                if best_ms < current_ms - 1e-9:
                    current_order = base_order[:best_pos] + [job] + base_order[best_pos:]
                    current_ms = best_ms
                    improved = True
                    # Restart insertion loop
                    break
            
            if not improved:
                break
        
        if not improved:
            break
            
    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses NEH heuristics, Simulated Annealing, and Local Search Polish.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a deadline with a safety margin (150s limit, keep 5s buffer)
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate multiple candidates using different NEH variants
    candidates = []
    
    # 1. Standard NEH (Total processing time)
    seq1 = _neh_heuristic(P, sort_key='total')
    candidates.append(seq1)
    
    # 2. NEH with first/last machine focus
    if time.time() < deadline - 10.0:
        seq2 = _neh_heuristic(P, sort_key='first_last')
        candidates.append(seq2)
        
    # 3. Random restarts if time permits and N is small enough
    if n_jobs <= 100 and time.time() < deadline - 30.0:
        for _ in range(2):
            if time.time() > deadline - 10.0:
                break
            random_order = list(range(n_jobs))
            rng.shuffle(random_order)
            candidates.append(random_order)

    # Select the best candidate based on makespan
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    
    for seq in candidates[1:]:
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
    
    # Apply Simulated Annealing to the best candidate
    # SA is good for escaping local minima, especially from random starts
    if time.time() < deadline - 10.0:
        sa_seq = _simulated_annealing(best_seq, P, deadline, rng)
        sa_ms = _makespan(sa_seq, P)
        
        if sa_ms < best_ms:
            best_seq = sa_seq
            best_ms = sa_ms
    
    # Apply Local Search Polish (Strict Improvement)
    # This ensures we end up in a local optimum
    if time.time() < deadline - 1.0:
        polished_seq = _local_search_polish(best_seq, P, deadline)
        polished_ms = _makespan(polished_seq, P)
        
        if polished_ms < best_ms:
            best_seq = polished_seq
            best_ms = polished_ms
    
    return list(best_seq)

# EVOLVE-BLOCK-END
