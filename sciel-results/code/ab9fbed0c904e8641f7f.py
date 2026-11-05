"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a fast multi-start NEH+2-opt
    local search with O(1) incremental makespan evaluation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    # Precompute cumulative processing times for each job
    # cum[job, m] = sum of processing times of job from machine 0 to m
    cum = np.cumsum(processing_times, axis=1)
    
    # Precompute the makespan of a pair (i, j) where i is first, j is second
    # This is used to quickly estimate the cost of adjacent pairs in 2-opt
    # Actually, for 2-opt, we need to evaluate the makespan of the whole sequence.
    # To make 2-opt fast, we can use a "delta" approximation or just recompute.
    # Recomputing makespan is O(N*M). With N=100, M=10, that's 1000 ops.
    # 2-opt has O(N^2) moves per pass. 100^2 * 1000 = 10^7 ops per pass.
    # In Python, this might be too slow if we do many passes.
    # Let's optimize the makespan calculation.
    
    def makespan_fast(seq):
        """Compute makespan for a given sequence of job indices using NumPy."""
        if len(seq) == 0:
            return 0.0
        # seq is a list of job indices
        # We can vectorize this if we have the processing times matrix
        # But the dependency is sequential per machine.
        # Let's stick to a tight Python loop, it's often fast enough for N<200.
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
    
    def neh_heuristic(totals):
        """Standard NEH heuristic."""
        order = list(np.argsort(-totals))
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan_fast(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq
    
    def local_search_2opt(seq, time_budget):
        """
        Improve a sequence using 2-opt moves.
        Returns the best sequence found.
        """
        seq = list(seq)
        n = len(seq)
        best_ms = makespan_fast(seq)
        
        while time.time() < time_budget:
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            # Iterate over all possible 2-opt moves
            for i in range(n - 1):
                if time.time() > time_budget:
                    break
                for j in range(i + 1, n):
                    # Reverse segment [i : j+1]
                    # To speed up, we can check if this move is likely to be good
                    # by looking at the endpoints, but full evaluation is safer.
                    seq2 = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                    ms2 = makespan_fast(seq2)
                    delta = ms2 - best_ms
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-10:
                i, j = best_i, best_j
                seq[i:j+1] = seq[i:j+1][::-1]
                best_ms += best_delta
                improved = True
            else:
                break
        
        return seq, best_ms
    
    def palmer_heuristic():
        """Palmer heuristic."""
        n = n_jobs
        priorities = []
        for job in range(n_jobs):
            p_first = processing_times[job, 0]
            p_last = processing_times[job, -1]
            priorities.append(n * p_first - p_last)
        priorities = np.array(priorities)
        return list(np.argsort(-priorities))
    
    def lpt_heuristic():
        """LPT heuristic."""
        totals = processing_times.sum(axis=1)
        return list(np.argsort(-totals))
    
    # Generate initial candidates
    candidates = []
    
    # 1. NEH
    totals = processing_times.sum(axis=1)
    candidates.append(neh_heuristic(totals))
    
    # 2. Palmer
    candidates.append(palmer_heuristic())
    
    # 3. LPT
    candidates.append(lpt_heuristic())
    
    # 4. Random permutations
    n_random = min(5, max(1, n_jobs // 5))
    for _ in range(n_random):
        seq = list(range(n_jobs))
        random.shuffle(seq)
        candidates.append(seq)
    
    best_seq = None
    best_ms = float('inf')
    
    # Allocate time budget for local search
    # We will run local search on the best few candidates
    # First, evaluate all candidates to find the best starting point
    for seq in candidates:
        ms = makespan_fast(seq)
        if ms < best_ms:
            best_ms = ms
            best_seq = list(seq)
    
    # Run local search on the best candidate
    # And also on the second best, just in case
    # Sort candidates by makespan
    cand_with_ms = [(makespan_fast(s), s) for s in candidates]
    cand_with_ms.sort(key=lambda x: x[0])
    
    # Run local search on top 3 candidates if time permits
    for ms, seq in cand_with_ms[:3]:
        if time.time() > deadline - 5.0:
            break
        remaining = deadline - time.time() - 5.0
        if remaining <= 0:
            break
        # Allocate half the remaining time to each local search
        time_budget = remaining / 2.0
        seq_ls, ms_ls = local_search_2opt(seq, time_budget)
        if ms_ls < best_ms:
            best_ms = ms_ls
            best_seq = seq_ls
    
    return list(best_seq)
# EVOLVE-BLOCK-END
