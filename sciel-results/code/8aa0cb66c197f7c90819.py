"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    prev_row = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines, dtype=np.float64)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])


def _makespan_from_prev_row(perm, processing_times, start_idx, prev_row):
    """Compute makespan from start_idx onward, given prev_row from before start_idx.
    Returns (makespan, final_prev_row)."""
    n_machines = processing_times.shape[1]
    for j in range(start_idx, len(perm)):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines, dtype=np.float64)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1]), prev_row


def _neh_order(processing_times):
    """Generate NEH heuristic ordering."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    
    return partial


def _neh_order_forward_backward(processing_times, max_rounds=5):
    """NEH with forward and backward passes."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    best_perm = list(np.argsort(-totals))
    best_ms = _makespan(best_perm, processing_times)
    
    for rnd in range(max_rounds):
        if rnd % 2 == 0:
            # Forward pass
            order = list(np.argsort(-totals))
        else:
            # Backward pass
            order = list(np.argsort(totals))
        
        partial = [order[0]]
        for i in range(1, n_jobs):
            job = order[i]
            best_pos = 0
            best_ms_candidate = float('inf')
            for pos in range(len(partial) + 1):
                candidate = partial[:pos] + [job] + partial[pos:]
                ms = _makespan(candidate, processing_times)
                if ms < best_ms_candidate:
                    best_ms_candidate = ms
                    best_pos = pos
            partial = partial[:best_pos] + [job] + partial[best_pos:]
        
        ms = _makespan(partial, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(partial)
    
    return best_perm


def _swap_adjacent(perm, i):
    """Swap adjacent jobs at positions i and i+1."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _insert_job(perm, job_idx, new_pos):
    """Move job at position job_idx to position new_pos."""
    job = perm[job_idx]
    new_perm = perm[:job_idx] + perm[job_idx+1:]
    if new_pos > job_idx:
        new_pos -= 1
    new_perm = new_perm[:new_pos] + [job] + new_perm[new_pos:]
    return new_perm


def _local_search_2opt(perm, processing_times, max_iterations=200):
    """2-opt style local search with adjacent swaps."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs - 1):
            new_perm = _swap_adjacent(best_perm, i)
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms - 1e-10:
                best_perm = new_perm
                best_ms = ms
                improved = True
        if not improved:
            break
    
    return best_perm


def _insertion_local_search(perm, processing_times, max_iterations=100):
    """Insertion-based local search: try moving each job to every other position."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                new_perm = _insert_job(best_perm, i, j)
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-10:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_perm


def _simulated_annealing(perm, processing_times, max_iterations=500, t_start=100.0, t_end=1.0):
    """Simulated annealing with insertion moves."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    for it in range(max_iterations):
        t = t_start * (t_end / t_start) ** (it / max(1, max_iterations - 1))
        
        # Random insertion move
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j:
            continue
        new_perm = _insert_job(current_perm, i, j)
        new_ms = _makespan(new_perm, processing_times)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-10)):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms - 1e-10:
                best_ms = current_ms
                best_perm = list(current_perm)
    
    return best_perm


def _multi_start(processing_times, n_starts, deadline):
    """Multi-start: generate multiple NEH variants and refine each."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    best_perm = list(np.argsort(-totals))
    best_ms = _makespan(best_perm, processing_times)
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Different random perturbations of the base order
        base_order = list(np.argsort(-totals))
        if s > 0:
            # Randomly shuffle a subset
            k = random.randint(1, min(3, n_jobs - 1))
            indices = random.sample(range(n_jobs), k)
            for idx in indices:
                new_pos = random.randint(0, n_jobs - 1)
                job = base_order[idx]
                base_order = base_order[:idx] + base_order[idx+1:]
                if new_pos > idx:
                    new_pos -= 1
                base_order = base_order[:new_pos] + [job] + base_order[new_pos:]
        
        # NEH insertion on this order
        partial = [base_order[0]]
        for i in range(1, n_jobs):
            job = base_order[i]
            best_pos = 0
            best_ms_candidate = float('inf')
            for pos in range(len(partial) + 1):
                candidate = partial[:pos] + [job] + partial[pos:]
                ms = _makespan(candidate, processing_times)
                if ms < best_ms_candidate:
                    best_ms_candidate = ms
                    best_pos = pos
            partial = partial[:best_pos] + [job] + partial[best_pos:]
        
        ms = _makespan(partial, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(partial)
        
        # Quick local search on this candidate
        if time.time() < deadline and n_jobs <= 50:
            polished = _local_search_2opt(partial, processing_times, max_iterations=50)
            polished_ms = _makespan(polished, processing_times)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = list(polished)
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: 20s before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set random seed for reproducibility within a run
    random.seed(42)
    
    # Baseline: longest processing time first
    baseline_perm = list(np.argsort(-processing_times.sum(axis=1)))
    best_perm = baseline_perm
    best_ms = _makespan(best_perm, processing_times)
    
    # Try NEH heuristic (forward)
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times)
        neh_ms = _makespan(neh_perm, processing_times)
        if neh_ms < best_ms:
            best_perm = neh_perm
            best_ms = neh_ms
    
    # Try NEH forward-backward
    if time.time() < deadline:
        neh_fb = _neh_order_forward_backward(processing_times, max_rounds=3)
        neh_fb_ms = _makespan(neh_fb, processing_times)
        if neh_fb_ms < best_ms:
            best_perm = neh_fb
            best_ms = neh_fb_ms
    
    # Multi-start with perturbations
    if time.time() < deadline:
        if n_jobs <= 30:
            n_starts = 5
        elif n_jobs <= 50:
            n_starts = 3
        else:
            n_starts = 2
        ms_perm = _multi_start(processing_times, n_starts, deadline)
        ms_perm_ms = _makespan(ms_perm, processing_times)
        if ms_perm_ms < best_ms:
            best_perm = ms_perm
            best_ms = ms_perm_ms
    
    # Local search polish: 2-opt first
    if time.time() < deadline and n_jobs > 1:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        # Estimate time per makespan computation
        # O(n_jobs * n_machines) per eval, with Python overhead
        est_time_per_eval = n_jobs * n_machines * 1e-6 + 1e-4
        
        if n_jobs <= 30:
            max_iter_2opt = min(500, int(remaining * 0.4 / max(est_time_per_eval, 1e-5)))
        elif n_jobs <= 50:
            max_iter_2opt = min(200, int(remaining * 0.3 / max(est_time_per_eval, 1e-5)))
        else:
            max_iter_2opt = min(50, int(remaining * 0.2 / max(est_time_per_eval, 1e-5)))
        
        max_iter_2opt = max(10, max_iter_2opt)
        
        polished = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter_2opt)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Insertion local search
    if time.time() < deadline and n_jobs > 1:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        if n_jobs <= 25:
            max_iter_ins = min(100, int(remaining * 0.3 / max(n_jobs * n_machines * 1e-6 + 1e-4, 1e-5)))
        elif n_jobs <= 50:
            max_iter_ins = min(30, int(remaining * 0.2 / max(n_jobs * n_machines * 1e-6 + 1e-4, 1e-5)))
        else:
            max_iter_ins = 5
        
        max_iter_ins = max(5, max_iter_ins)
        
        polished = _insertion_local_search(best_perm, processing_times, max_iterations=max_iter_ins)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Simulated annealing for broader exploration
    if time.time() < deadline and n_jobs > 1:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        est_time_per_eval = n_jobs * n_machines * 1e-6 + 1e-4
        if n_jobs <= 30:
            max_iter_sa = min(1000, int(remaining * 0.5 / max(est_time_per_eval, 1e-5)))
        elif n_jobs <= 50:
            max_iter_sa = min(300, int(remaining * 0.4 / max(est_time_per_eval, 1e-5)))
        else:
            max_iter_sa = min(100, int(remaining * 0.3 / max(est_time_per_eval, 1e-5)))
        
        max_iter_sa = max(50, max_iter_sa)
        
        sa_perm = _simulated_annealing(best_perm, processing_times, max_iterations=max_iter_sa)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Final polish: tight local search
    if time.time() < deadline and n_jobs > 1:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        if remaining > 5.0:
            if n_jobs <= 30:
                final_iter = min(200, int(remaining * 0.8 / max(n_jobs * n_machines * 1e-6 + 1e-4, 1e-5)))
            else:
                final_iter = min(50, int(remaining * 0.5 / max(n_jobs * n_machines * 1e-6 + 1e-4, 1e-5)))
            
            final_iter = max(10, final_iter)
            polished = _local_search_2opt(best_perm, processing_times, max_iterations=final_iter)
            polished_ms = _makespan(polished, processing_times)
            if polished_ms < best_ms:
                best_perm = polished
                best_ms = polished_ms
    
    # Check deadline one more time
    if time.time() > deadline:
        # Return best found so far
        pass
    
    return list(best_perm)
# EVOLVE-BLOCK-END
