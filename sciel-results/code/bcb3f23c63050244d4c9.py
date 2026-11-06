import random
import time
import numpy as np

def _makespan(order, processing_times):
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    
    prev = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        curr_0 = prev[0] + p[0]
        curr_m_minus_1 = curr_0
        for m in range(1, n_machines):
            curr_m = max(curr_m_minus_1, prev[m]) + p[m]
            prev[m] = curr_m
            curr_m_minus_1 = curr_m
        prev[0] = curr_0
    return prev[-1]

def _neh_initial(processing_times):
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        sequence.insert(best_pos, job)
    return sequence

def _tabu_search(initial_seq, processing_times, time_limit=20.0, tabu_tenure=10):
    n_jobs = len(initial_seq)
    current_seq = initial_seq[:]
    current_ms = _makespan(current_seq, processing_times)
    best_seq = current_seq[:]
    best_ms = current_ms
    
    tabu_list = {}
    no_improve_count = 0
    stagnation_limit = 100
    start_time = time.time()
    it = 0
    
    while time.time() - start_time < time_limit:
        best_neighbor_seq = None
        best_neighbor_ms = float('inf')
        best_neighbor_job = -1
        
        sample_size = max(20, min(100, int(1000 / n_jobs)))
        
        for _ in range(sample_size):
            i_idx = random.randint(0, n_jobs - 1)
            j_idx = random.randint(0, n_jobs - 1)
            if i_idx == j_idx:
                continue
            
            job = current_seq[i_idx]
            
            temp_seq = current_seq[:i_idx] + current_seq[i_idx+1:]
            if j_idx > i_idx:
                j_idx -= 1
            candidate_seq = temp_seq[:j_idx] + [job] + temp_seq[j_idx:]
            
            ms = _makespan(candidate_seq, processing_times)
            
            is_tabu = False
            if job in tabu_list and tabu_list[job] > it:
                is_tabu = True
                
            if is_tabu and ms >= best_ms:
                continue
                
            if ms < best_neighbor_ms:
                best_neighbor_ms = ms
                best_neighbor_seq = candidate_seq
                best_neighbor_job = job

        if best_neighbor_seq is None:
            tabu_list.clear()
            i_idx = random.randint(0, n_jobs - 1)
            j_idx = random.randint(0, n_jobs - 1)
            if i_idx == j_idx:
                continue
            job = current_seq[i_idx]
            temp_seq = current_seq[:i_idx] + current_seq[i_idx+1:]
            if j_idx > i_idx:
                j_idx -= 1
            best_neighbor_seq = temp_seq[:j_idx] + [job] + temp_seq[j_idx:]
            best_neighbor_ms = _makespan(best_neighbor_seq, processing_times)
            best_neighbor_job = job
            
        current_seq = best_neighbor_seq
        current_ms = best_neighbor_ms
        tabu_list[best_neighbor_job] = it + tabu_tenure
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = current_seq[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > stagnation_limit:
            current_seq = list(range(n_jobs))
            random.shuffle(current_seq)
            current_ms = _makespan(current_seq, processing_times)
            no_improve_count = 0
            tabu_list.clear()
            
        it += 1
        
    return best_seq, best_ms

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    total_time_limit = 110.0
    start_time = time.time()
    
    best_global_seq = []
    best_global_ms = float('inf')
    
    seq_neh = _neh_initial(processing_times)
    time_remaining = total_time_limit - (time.time() - start_time)
    if time_remaining > 1:
        best_seq, best_ms = _tabu_search(seq_neh, processing_times, time_limit=time_remaining/2)
        if best_ms < best_global_ms:
            best_global_ms = best_ms
            best_global_seq = best_seq
            
    time_remaining = total_time_limit - (time.time() - start_time)
    restarts = 5
    time_per_restart = time_remaining / restarts
    
    for _ in range(restarts):
        if time.time() - start_time > total_time_limit:
            break
        seq_random = list(range(n_jobs))
        random.shuffle(seq_random)
        best_seq, best_ms = _tabu_search(seq_random, processing_times, time_limit=time_per_restart)
        if best_ms < best_global_ms:
            best_global_ms = best_ms
            best_global_seq = best_seq
            
    return best_global_seq
