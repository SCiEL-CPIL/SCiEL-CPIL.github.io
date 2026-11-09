"""
Flow-shop permutation construction using Multi-Start Tabu Search.
Strategy:
1. Initialize with multiple diverse solutions (NEH variants and random).
2. Run independent Tabu Search instances with a dynamic time budget.
3. Use a 'Polishing' phase on the best candidate found, allowing deeper local search.
4. Tabu list prevents cycling and forces exploration of new regions (diversification).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order.
    P: (n_jobs, n_machines)
    order: list of job indices
    """
    if len(order) == 0:
        return 0.0
    
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    
    # c_machines[m] holds the completion time of the current job on machine m
    # We iterate through jobs
    c_machines = [0.0] * n_machines
    
    # Pre-extract rows for speed if possible, but direct indexing is okay for small N
    # For larger N, P[order] creates a copy which is O(N*M). 
    # Let's stick to direct indexing to avoid memory overhead in tight loops, 
    # but we can optimize by accessing P rows.
    
    last_completion = 0.0
    for j in range(n_jobs_seq):
        job_idx = order[j]
        row = P[job_idx]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
        
    return last_completion

def _neh_heuristic(P, mode=0):
    """
    NEH heuristic. 
    mode 0: Standard (descending total processing time)
    mode 1: Randomized NEH (perturbed order)
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    
    if mode == 0:
        order_indices = np.argsort(-totals)
    else:
        # Add noise to totals for diversity
        noise = np.random.uniform(0, np.max(totals) * 0.2, size=n_jobs)
        perturbed_totals = totals + noise
        order_indices = np.argsort(-perturbed_totals)

    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _tabu_search(seq, P, time_limit, tabu_tenure, max_iters):
    """
    Run Tabu Search starting from seq.
    Returns best_seq found and its makespan.
    """
    n_jobs = len(seq)
    n_machines = P.shape[1]
    
    current_seq = seq.copy()
    current_makespan = _makespan(current_seq, P)
    best_seq = current_seq.copy()
    best_makespan = current_makespan
    
    # Tabu list: stores (job, from_pos, to_pos) or just positions?
    # Standard swap-based Tabu: tabu[job] = iteration when it becomes non-tabu
    # Or better: tabu_pairs set of (i, j) where i, j are positions?
    # Let's use a dictionary mapping job index to the iteration it was last moved.
    # If a job was moved at iter t, we cannot move it back to its previous position until t + tabu_tenure.
    # To keep it simple and effective for flow shop, we use swap moves (i, j).
    # Tabu condition: swap(i, j) is tabu if i or j was the position of the last moved job?
    # A common simple approach: Tabu list of jobs. If job j was moved in the last T iterations, 
    # we cannot swap it with any other job? No, that's too restrictive.
    # Let's use: Tabu list of (job_index). If job_index is in tabu_list, we cannot select it as the 
    # 'source' of a swap? Or we cannot reverse the last move.
    # Reverse move tabu is standard. 
    # Let's store last_move = (job_a, pos_a, job_b, pos_b).
    # Tabu if we try to swap job_a and job_b again? No, that's too narrow.
    # Let's use: Tabu list of jobs. If job j is in the tabu list, it cannot be moved.
    # This forces diversification.
    
    tabu_list = {} # job_idx -> iteration when it expires
    iteration = 0
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit and iteration < max_iters:
        iteration += 1
        
        best_neighbor_seq = None
        best_neighbor_makespan = float('inf')
        best_move = None
        
        # Generate neighbors: Swap two jobs
        # To speed up, we don't evaluate all N*(N-1)/2 swaps.
        # We sample a subset or use a specific heuristic.
        # For robustness, let's evaluate a random sample of swaps if N is large, 
        # or all swaps if N is small.
        
        if n_jobs <= 50:
            candidates = [(i, j) for i in range(n_jobs) for j in range(i+1, n_jobs)]
            random.shuffle(candidates)
            # Limit candidates per iteration to ensure speed
            candidates = candidates[:min(100, len(candidates))]
        else:
            # Sample random swaps
            candidates = set()
            for _ in range(100):
                i, j = random.sample(range(n_jobs), 2)
                if i > j: i, j = j, i
                candidates.add((i, j))
            candidates = list(candidates)
        
        for i, j in candidates:
            # Check tabu
            # If job at i or job at j is tabu, skip?
            # Let's define tabu as: Job X is tabu if it was moved in the last T iterations.
            job_i = current_seq[i]
            job_j = current_seq[j]
            
            if job_i in tabu_list and tabu_list[job_i] > iteration:
                continue
            if job_j in tabu_list and tabu_list[job_j] > iteration:
                continue
            
            # Perform swap
            new_seq = current_seq.copy()
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            
            new_makespan = _makespan(new_seq, P)
            
            # Aspiration criterion: Accept if it improves best known
            if new_makespan < best_neighbor_makespan:
                best_neighbor_makespan = new_makespan
                best_neighbor_seq = new_seq
                best_move = (job_i, job_j, i, j)
                
        if best_neighbor_seq is None:
            # No valid moves found (stuck in tabu), clear tabu or pick best regardless
            # For simplicity, pick a random move ignoring tabu
            i, j = random.sample(range(n_jobs), 2)
            best_neighbor_seq = current_seq.copy()
            best_neighbor_seq[i], best_neighbor_seq[j] = best_neighbor_seq[j], best_neighbor_seq[i]
            best_neighbor_makespan = _makespan(best_neighbor_seq, P)
            best_move = (current_seq[i], current_seq[j], i, j)
            
        # Update current solution
        current_seq = best_neighbor_seq
        current_makespan = best_neighbor_makespan
        
        # Update tabu list
        if best_move:
            job_i, job_j, _, _ = best_move
            # Mark both jobs as tabu for next T iterations
            tabu_list[job_i] = iteration + tabu_tenure
            tabu_list[job_j] = iteration + tabu_tenure
            
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = current_seq.copy()
            
    return best_seq, best_makespan

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Multi-Start Tabu Search.
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
    # 150s limit, use 145s to be safe
    total_time_limit = 145.0
    
    # Phase 1: Multi-Start Exploration
    # Allocate 70% of time to exploration
    exploration_time = total_time_limit * 0.7
    
    # Determine number of restarts based on time per restart
    # Estimate time per restart: ~100ms - 500ms depending on N
    # We will run restarts until time runs out.
    
    best_seq_overall = None
    best_makespan_overall = float('inf')
    
    # Initialize with NEH
    initial_seq = _neh_heuristic(P, mode=0)
    initial_ms = _makespan(initial_seq, P)
    best_seq_overall = initial_seq
    best_makespan_overall = initial_ms
    
    restart_count = 0
    # Tabu tenure: typically sqrt(N) or 2*sqrt(N)
    base_tenure = max(5, int(2 * np.sqrt(n_jobs)))
    
    while (time.time() - start_time) < exploration_time:
        restart_count += 1
        
        # Generate diverse initial solution
        # 50% chance NEH variant, 50% random
        if random.random() < 0.5:
            mode = 0 if restart_count % 2 == 0 else 1
            current_start_seq = _neh_heuristic(P, mode=mode)
        else:
            current_start_seq = list(range(n_jobs))
            random.shuffle(current_start_seq)
        
        # Calculate time budget for this restart
        # Leave 5s for polishing phase
        elapsed = time.time() - start_time
        remaining_for_exploration = exploration_time - elapsed
        if remaining_for_exploration <= 0:
            break
            
        # Allocate a portion of remaining time to this restart
        # If many restarts, each gets less time.
        # Let's aim for ~2-5 seconds per restart for small N, less for large N.
        # Simple heuristic: 10% of remaining time, min 0.1s, max 5s
        restart_time = max(0.1, min(5.0, remaining_for_exploration * 0.1))
        
        # Adjust tabu tenure slightly for diversity
        tenure = base_tenure + random.randint(-2, 2)
        
        seq, ms = _tabu_search(current_start_seq, P, restart_time, tenure, max_iters=10000)
        
        if ms < best_makespan_overall:
            best_makespan_overall = ms
            best_seq_overall = seq
            
    # Phase 2: Polishing (Intensive Local Search on Best)
    elapsed = time.time() - start_time
    polishing_time = total_time_limit - elapsed - 1.0 # 1s safety margin
    
    if polishing_time > 0.5 and best_seq_overall is not None:
        # Use a longer tabu tenure or different move set?
        # Let's just run another Tabu search with more time and slightly larger tenure
        polish_tenure = int(base_tenure * 1.5)
        final_seq, final_ms = _tabu_search(best_seq_overall, P, polishing_time, polish_tenure, max_iters=20000)
        
        if final_ms < best_makespan_overall:
            best_makespan_overall = final_ms
            best_seq_overall = final_seq
            
    return best_seq_overall
# EVOLVE-BLOCK-END
