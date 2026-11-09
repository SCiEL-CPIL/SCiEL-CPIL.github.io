"""
Flow-shop permutation construction using Multi-start Iterated Local Search (ILS).
Strategy:
1. Initialize with NEH heuristic.
2. Perform 2-opt local search using incremental makespan updates for speed.
3. If stuck, apply a perturbation: remove a random block of jobs and re-insert them using NEH logic.
4. Repeat with multiple restarts if time permits.
5. Keep the best solution found across all restarts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_full(order, P):
    """
    Compute makespan for a given job order using vectorized NumPy operations.
    """
    if len(order) == 0:
        return 0.0
    
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    
    # Extract processing times for the sequence
    seq_P = P[order]
    
    # C_machines[j] will hold the completion time of job j on the current machine.
    c_machines = np.zeros(n_machines)
    last_completion = 0.0
    
    for j in range(n_jobs_seq):
        row = seq_P[j]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
        
    return last_completion

def _neh_heuristic(P):
    """
    NEH heuristic for initial solution.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan_full(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _local_search_2opt(seq, P, time_limit):
    """
    Perform 2-opt local search.
    Uses a simple full recomputation for simplicity but optimized inner loop.
    If this is too slow, we would need incremental updates, but for n<100, 
    full recomputation of O(n*m) per move with n moves is O(n^2*m).
    For n=100, m=10, that's 100,000 ops per iteration. 1000 iterations = 10^8 ops. 
    Python is slow. 
    Let's try to make this faster or limit iterations.
    Actually, let's use a simpler swap-based local search which is faster to evaluate 
    if we can do incremental, but without incremental, we just do limited iterations.
    """
    n = len(seq)
    current_seq = seq.copy()
    current_ms = _makespan_full(current_seq, P)
    best_seq = current_seq.copy()
    best_ms = current_ms
    
    no_improve_count = 0
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit:
        improved = False
        # Randomly sample a few swaps to avoid full O(n^2) scan which might be too slow in Python
        # Or do a full scan if n is small
        if n <= 50:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
        else:
            # Sample 20% of pairs
            num_samples = max(10, int(0.2 * n * n))
            pairs = []
            for _ in range(num_samples):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j:
                    if i > j: i, j = j, i
                    pairs.append((i, j))
            # Remove duplicates roughly
            pairs = list(set(pairs))
            random.shuffle(pairs)
        
        for i, j in pairs:
            new_seq = current_seq.copy()
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            new_ms = _makespan_full(new_seq, P)
            
            if new_ms < current_ms:
                current_seq = new_seq
                current_ms = new_ms
                improved = True
                no_improve_count = 0
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seq = current_seq.copy()
                break # Restart scan from beginning or continue? Continue is fine.
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                break
    return best_seq, best_ms

def _perturb(seq, P):
    """
    Perturb the sequence by removing a random block and re-inserting using NEH logic.
    """
    n = len(seq)
    if n < 4:
        return seq.copy()
    
    # Remove a block of size 1 to 3
    block_size = random.randint(1, min(3, n-1))
    start = random.randint(0, n - block_size)
    
    block = seq[start:start+block_size]
    remaining = seq[:start] + seq[start+block_size:]
    
    # Re-insert block elements one by one using NEH-style insertion
    new_seq = remaining.copy()
    for job in block:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(new_seq) + 1):
            candidate = new_seq[:pos] + [job] + new_seq[pos:]
            ms = _makespan_full(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        new_seq.insert(best_pos, job)
        
    return new_seq

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Multi-start ILS.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_seq_global = None
    best_ms_global = float('inf')
    
    # Number of restarts depends on time. 
    # Each restart: NEH (O(n^2*m)) + LS (O(k * n^2 * m) roughly) + Perturb (O(n^2*m))
    # Let's try a few restarts.
    
    # Initial NEH
    seq = _neh_heuristic(P)
    ms = _makespan_full(seq, P)
    
    best_seq_global = seq.copy()
    best_ms_global = ms
    
    # Local Search on initial
    seq, ms = _local_search_2opt(seq, P, time_limit * 0.2)
    if ms < best_ms_global:
        best_ms_global = ms
        best_seq_global = seq.copy()
    
    # ILS Loop
    while (time.time() - start_time) < time_limit * 0.9:
        # Perturb
        new_seq = _perturb(best_seq_global, P)
        
        # Local Search on perturbed
        new_seq, new_ms = _local_search_2opt(new_seq, P, time_limit * 0.1) # Cap LS time per iteration
        
        if new_ms < best_ms_global:
            best_ms_global = new_ms
            best_seq_global = new_seq.copy()
        else:
            # With some probability, accept worse solution to escape local optima (Simulated Annealing style)
            # But ILS usually just iterates. Let's stick to strict improvement for now to ensure progress.
            # Or accept if it's not too bad?
            # Let's try accepting worse solutions with decreasing probability.
            # For simplicity, just keep best.
            pass
            
    return best_seq_global
# EVOLVE-BLOCK-END
