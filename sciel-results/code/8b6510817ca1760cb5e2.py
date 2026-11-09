"""
Flow-shop permutation construction using NEH/CDS initialization,
Adaptive Local Search (2-Opt/Insertion), and Simulated Annealing,
with strict deadline management.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Optimized for speed using numpy vectorization where possible, 
    but the loop structure is kept simple for clarity and low overhead.
    """
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    c_machines = np.zeros(n_machines)
    c_curr = 0.0
    
    for j in order:
        row = P[j]
        # c_curr is the completion time of the current job on the previous machine
        # c_machines[m] is the completion time of the previous job on machine m
        # The start time for job j on machine m is max(c_curr, c_machines[m])
        # The completion time is start_time + row[m]
        c_curr = np.maximum(c_machines, c_curr) + row
        c_machines = c_curr
        
    return float(c_curr[-1])

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
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

def _cds_heuristic(P):
    """
    Campbell, Duke, and Sarma (CDS) heuristic.
    """
    n_jobs, n_machines = P.shape
    if n_machines < 2:
        return list(range(n_jobs))
        
    best_seq = None
    best_ms = float('inf')
    
    for i in range(1, n_machines):
        P2 = np.zeros((n_jobs, 2))
        P2[:, 0] = P[:, :i].sum(axis=1)
        P2[:, 1] = P[:, n_machines-i:].sum(axis=1)
        
        # Johnson's rule
        set_A = []
        set_B = []
        for j in range(n_jobs):
            if P2[j, 0] < P2[j, 1]:
                set_A.append((P2[j, 0], j))
            else:
                set_B.append((P2[j, 1], j))
                
        set_A.sort(key=lambda x: x[0])
        set_B.sort(key=lambda x: -x[0])
        
        seq = [j for _, j in set_A] + [j for _, j in set_B]
        ms = _makespan(seq, P)
        
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
            
    return best_seq if best_seq else list(range(n_jobs))

def _local_search_insertion(order, P, deadline, window_size=None):
    """
    Local search using insertion moves.
    If window_size is None, it attempts full neighborhood for small N,
    otherwise it restricts to a window around the current position.
    """
    n = len(order)
    if n < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    # Determine search strategy based on N
    if window_size is None:
        if n <= 50:
            window_size = n # Full neighborhood
        else:
            window_size = 15 # Limited window for large N
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_move = None
        best_new_ms = current_ms
        best_new_order = None
        
        # Check time more frequently for large N
        check_interval = 100 if n > 50 else 10
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            job_i = current_order[i]
            # Create a copy of the order without job i
            remaining = current_order[:i] + current_order[i+1:]
            
            # Determine range of insertion positions
            # We want to try positions near i
            # In 'remaining', index i corresponds to the position after removing i.
            # If we insert at j in 'remaining', it corresponds to position j in original if j < i, 
            # and j+1 in original if j >= i.
            
            # We restrict j to be within window_size of i
            min_j = max(0, i - window_size)
            max_j = min(len(remaining), i + window_size)
            
            for j in range(min_j, max_j + 1):
                # Skip if it's the same position (j == i in remaining means pos i or i+1 in original)
                # Actually, if we remove i, the element at i in remaining is originally i+1.
                # Inserting at i puts it back. Inserting at i+1 puts it after original i+1? No.
                # Let's just evaluate. The cost is low.
                
                new_order = remaining[:j] + [job_i] + remaining[j:]
                ms = _makespan(new_order, P)
                
                if ms < best_new_ms:
                    best_new_ms = ms
                    best_new_order = new_order
                    
            # Early exit if we found a good improvement? 
            # For best-improvement, we must check all. 
            # For first-improvement, we could break, but best-improvement is usually better for FSP.
            # Given the time limit, we stick to best-improvement within the window.
            
            if (i - min_j) % check_interval == 0 and time.time() > deadline:
                break

        if best_new_order is not None:
            current_order = best_new_order
            current_ms = best_new_ms
            improved = True
        else:
            improved = False
            
    return current_order

def _simulated_annealing(order, P, deadline, initial_temp=None):
    """
    Simulated Annealing to escape local optima.
    """
    n = len(order)
    if n < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    best_order = current_order[:]
    best_ms = current_ms
    
    if initial_temp is None:
        initial_temp = current_ms * 0.1
    
    T = initial_temp
    T_end = initial_temp * 1e-6
    
    no_improve = 0
    patience = 1000
    
    while time.time() < deadline and T > T_end:
        # Generate neighbor
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Insertion
            i = random.randint(0, n - 1)
            job_i = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            j = random.randint(0, len(remaining))
            new_order = remaining[:j] + [job_i] + remaining[j:]
            
        new_ms = _makespan(new_order, P)
        delta = new_ms - current_ms
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        # Cooling
        if no_improve > patience:
            T *= 0.8
            no_improve = 0
        else:
            T *= 0.999
            
    return best_order

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH/CDS initialization,
    Local Search, and Simulated Annealing.
    """
    P = np.asarray(processing_times, dtype=np.float64)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    # Safety margin: The limit is 150s. We stop at 145s to be safe.
    deadline = start_time + 145.0
    
    # 1. Generate initial candidates
    candidates = []
    
    # NEH is usually strong
    seq_neh = _neh_heuristic(P)
    candidates.append(seq_neh)
    
    # CDS
    seq_cds = _cds_heuristic(P)
    candidates.append(seq_cds)
    
    # Random
    random_perm = list(range(n_jobs))
    random.shuffle(random_perm)
    candidates.append(random_perm)
    
    # Select best initial sequence
    best_order = None
    best_makespan = float('inf')
    for seq in candidates:
        ms = _makespan(seq, P)
        if ms < best_makespan:
            best_makespan = ms
            best_order = seq
            
    # 2. Local Search
    # Allocate 50% of time to initial local search
    ls_deadline = start_time + (deadline - start_time) * 0.5
    
    # Use adaptive window size
    if n_jobs <= 50:
        window = None # Full
    else:
        window = 10 # Limited
    
    best_order = _local_search_insertion(best_order, P, ls_deadline, window)
    best_makespan = _makespan(best_order, P)
    
    # 3. Simulated Annealing
    # Use remaining time minus a buffer for final polish
    sa_deadline = deadline - 5.0
    
    if time.time() < sa_deadline:
        best_order = _simulated_annealing(best_order, P, sa_deadline)
        best_makespan = _makespan(best_order, P)
        
    # 4. Final Polish
    if time.time() < deadline - 1.0:
        # Quick local search with small window to fix any SA regressions
        final_window = 5 if n_jobs > 50 else None
        best_order = _local_search_insertion(best_order, P, deadline - 1.0, final_window)
        best_makespan = _makespan(best_order, P)
        
    return best_order
# EVOLVE-BLOCK-END
