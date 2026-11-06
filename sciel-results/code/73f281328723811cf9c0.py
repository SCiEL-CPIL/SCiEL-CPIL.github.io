"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan_vectorized(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    This implementation is optimized for small N (jobs) and M (machines).
    For flow shop, C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m].
    We iterate over jobs (outer) and machines (inner).
    Since M is usually small (e.g., 5-20), the inner loop is short.
    Using a list of floats for the current row completion times is faster than numpy arrays for very small M due to overhead.
    However, if M is large, numpy might be better. 
    Given typical flow shop instances, M is small.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # perm is a list of job indices
    # processing_times is (N, M)
    
    # completion[m] holds the completion time of the last job on machine m
    # We initialize with zeros
    # Actually, we need to be careful with types.
    
    # Let's stick to a tight python loop but optimized.
    # Pre-extract rows? No, perm is dynamic.
    
    # For very small N, this is fine.
    # For larger N, we might need C extensions or better vectorization.
    # But let's try to keep it simple and robust.
    
    n_machines = processing_times.shape[1]
    
    # If n_machines is 1, makespan is sum of processing times of all jobs.
    if n_machines == 1:
        return np.sum(processing_times[perm, 0])
        
    # Standard loop
    # completion[m] for the current last job processed
    completion = [0.0] * n_machines
    
    for job_idx in perm:
        row = processing_times[job_idx]
        prev_c = 0.0
        for m in range(n_machines):
            # c_prev = completion[m] (completion of previous job on this machine)
            # c_left = prev_c (completion of current job on previous machine)
            # new_c = max(c_prev, c_left) + row[m]
            if completion[m] > prev_c:
                new_c = completion[m] + row[m]
            else:
                new_c = prev_c + row[m]
            completion[m] = new_c
            prev_c = new_c
            
    return completion[-1]

def _neh_heuristic(processing_times):
    """
    NEH (Nowicki and Enscore) heuristic for flow shop.
    Returns a permutation list.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    # Insert each remaining job at the best position
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_ms = float('inf')
        
        # Try all possible positions for insertion
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan_vectorized(processing_times, candidate_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def _tabu_search(processing_times, initial_perm, time_limit):
    """
    Tabu Search with a simple tabu list and aspiration criterion.
    Uses swap moves.
    """
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    
    current_perm = initial_perm[:]
    current_ms = _calc_makespan_vectorized(processing_times, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    # Tabu list: store (i, j) swaps recently made. 
    # To avoid storing pairs, we can store the last swap time for each index or use a queue of moves.
    # A simple approach: store the last K swaps as a list of (i, j) tuples.
    # Or, for each position i, store the last time a job was moved *from* or *to* i.
    # Let's use a list of moves: [(i1, j1), (i2, j2), ...] and check if a move is in the list.
    # For efficiency, a set of tuples is better, but we need to expire old ones.
    # Let's use a deque for the tabu list with a max length.
    
    tabu_length = 10  # Keep last 10 moves as tabu
    tabu_list = []  # List of (i, j) tuples
    max_tabu_size = 100
    
    # Aspiration: accept tabu move if it improves best_ms
    
    # Neighborhood: swap any two jobs
    # To speed up, we can restrict to swaps that are likely beneficial? 
    # Or just random sampling of swaps?
    # For small N, full neighborhood is O(N^2).
    # For N=50, 2500 swaps. Each makespan calc is O(N*M).
    # If M=10, 50*10=500 ops. 2500 * 500 = 1.25M ops per iteration.
    # In Python, this might be slow. 1.25M simple ops ~ 0.1s? Maybe.
    # So we can do ~1000 iterations in 100s? 
    # Let's try full neighborhood for N < 30, otherwise random sampling.
    
    if n_jobs <= 30:
        full_neighborhood = True
    else:
        full_neighborhood = False
        sample_size = min(100, n_jobs * (n_jobs - 1) // 2)
        
    while time.time() < start_time + time_limit:
        best_move = None
        best_move_ms = float('inf')
        best_i, best_j = -1, -1
        
        if full_neighborhood:
            for i in range(n_jobs):
                for j in range(i + 1, n_jobs):
                    # Check tabu
                    # We need to check if (i,j) or (j,i) is in tabu_list
                    # Since tabu_list is a list, this is O(K). K is small.
                    is_tabu = False
                    for ti, tj in tabu_list:
                        if (ti == i and tj == j) or (ti == j and tj == i):
                            is_tabu = True
                            break
                    
                    # Aspiration
                    if is_tabu:
                        # Only accept if it improves global best
                        # We don't know the makespan yet without calculating.
                        # So we must calculate makespan to check aspiration.
                        pass # We'll calculate makespan anyway
                
                # Actually, checking tabu before calculation is good.
                # But we need the makespan to know if it's better than current best move.
                # So we calculate makespan for all non-tabu moves, and for tabu moves only if they might be aspiration (i.e., we don't know without calc).
                # Standard TS: generate neighbors, calculate cost, filter tabu, apply aspiration.
                
                    # Create new perm
                    new_perm = current_perm[:]
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    ms = _calc_makespan_vectorized(processing_times, new_perm)
                    
                    # Aspiration criterion: accept if ms < best_ms
                    if is_tabu and not (ms < best_ms):
                        continue
                        
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = new_perm
                        best_i, best_j = i, j
        else:
            # Random sampling
            moves_tried = set()
            for _ in range(sample_size):
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i == j:
                    continue
                if (i, j) in moves_tried or (j, i) in moves_tried:
                    continue
                moves_tried.add((i, j))
                
                is_tabu = False
                for ti, tj in tabu_list:
                    if (ti == i and tj == j) or (ti == j and tj == i):
                        is_tabu = True
                        break
                
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _calc_makespan_vectorized(processing_times, new_perm)
                
                if is_tabu and not (ms < best_ms):
                    continue
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = new_perm
                    best_i, best_j = i, j
        
        if best_move is None:
            # No valid move found (all tabu and not aspiration)
            # This is rare with random sampling, but possible with full neighborhood if stuck.
            # Break or random restart?
            # Let's just break to avoid infinite loop.
            break
        
        # Update current solution
        current_perm = best_move
        current_ms = best_move_ms
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm[:]
        
        # Update tabu list
        tabu_list.append((best_i, best_j))
        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(0)
            
        # Optional: Diversification
        # If no improvement for a long time, restart from best with random perturbation?
        # For now, keep it simple.
        
    return best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    # NEH is O(N^3 * M) roughly, which is fine for N < 100.
    # For larger N, we might need a faster heuristic or just random.
    # Let's use NEH for N <= 100, else random sort by total time.
    
    if n_jobs <= 100:
        initial_perm = _neh_heuristic(processing_times)
    else:
        totals = processing_times.sum(axis=1)
        initial_perm = np.argsort(-totals).tolist()
        
    # 2. Tabu Search
    # Allocate most of the time to Tabu Search.
    ts_time_limit = deadline - time.time()
    
    if ts_time_limit > 1.0:
        best_perm, best_ms = _tabu_search(processing_times, initial_perm, ts_time_limit)
    else:
        best_perm = initial_perm
        best_ms = _calc_makespan_vectorized(processing_times, best_perm)
        
    return best_perm
# EVOLVE-BLOCK-END
