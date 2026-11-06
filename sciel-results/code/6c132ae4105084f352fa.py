# EVOLVE-BLOCK-START
import numpy as np

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Uses an optimized NEH heuristic with incremental completion matrix updates
    and a local search phase including adjacent swaps and 2-opt reversals,
    leveraging vectorized NumPy operations for fast neighborhood evaluation.
    """
    n_jobs, n_machines = processing_times.shape
    n = n_jobs
    M = n_machines
    
    if n <= 1:
        return list(range(n))
    
    p = processing_times
    
    def compute_C(perm, p):
        """
        Compute makespan and completion matrix C using vectorized NumPy operations.
        """
        n_seq = len(perm)
        if n_seq == 0:
            return 0.0, np.zeros((0, M))
        
        C = np.zeros((n_seq, M))
        p_perm = p[perm]
        
        # First row
        C[0, 0] = p_perm[0, 0]
        if M > 1:
            C[0, 1:] = np.cumsum(p_perm[0, 1:])
            
        # Subsequent rows
        for i in range(1, n_seq):
            C[i, 0] = C[i-1, 0] + p_perm[i, 0]
            if M > 1:
                # C[i, m] = max(C[i-1, m], C[i, m-1]) + p[i, m]
                # Vectorized for m=1..M-1
                C[i, 1:] = np.maximum(C[i-1, 1:], C[i, :-1]) + p_perm[i, 1:]
        
        return C[-1, -1], C

    def neh_initial(p):
        """
        NEH heuristic with incremental updates.
        """
        totals = p.sum(axis=1)
        order = list(np.argsort(-totals))
        
        seq = [order[0]]
        ms, C = compute_C(seq, p)
        
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            best_C = None
            current_len = len(seq)
            
            for pos in range(current_len + 1):
                candidate = seq[:pos] + [job] + seq[pos:]
                cand_ms, cand_C = compute_C(candidate, p)
                
                if cand_ms < best_ms:
                    best_ms = cand_ms
                    best_pos = pos
                    best_C = cand_C
            
            seq = seq[:best_pos] + [job] + seq[best_pos:]
            C = best_C
            ms = best_ms
            
        return seq, ms, C

    best_perm, best_ms, best_C = neh_initial(p)
    
    def try_adjacent_swaps(perm, ms, C):
        """
        Local search using adjacent swaps with fast re-evaluation.
        """
        improved = True
        max_passes = 30
        for _ in range(max_passes):
            if not improved:
                break
            improved = False
            n_seq = len(perm)
            best_swap = -1
            best_swap_ms = ms
            best_swap_C = None
            
            for i in range(n_seq - 1):
                new_perm = perm[:]
                new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
                
                # Recompute C from i onwards
                new_C = C.copy()
                if i > 0:
                    new_C[:i, :] = C[:i, :]
                
                p_perm = p[new_perm]
                
                for pos in range(i, n_seq):
                    if pos == 0:
                        new_C[pos, 0] = p_perm[pos, 0]
                        if M > 1:
                            new_C[pos, 1:] = np.cumsum(p_perm[pos, 1:])
                    else:
                        new_C[pos, 0] = new_C[pos-1, 0] + p_perm[pos, 0]
                        if M > 1:
                            new_C[pos, 1:] = np.maximum(new_C[pos-1, 1:], new_C[pos, :-1]) + p_perm[pos, 1:]
                
                new_ms = new_C[-1, -1]
                
                if new_ms < best_swap_ms - 1e-10:
                    best_swap_ms = new_ms
                    best_swap = i
                    best_swap_C = new_C
            
            if best_swap != -1:
                i = best_swap
                perm[i], perm[i+1] = perm[i+1], perm[i]
                ms = best_swap_ms
                C = best_swap_C
                improved = True
                
        return perm, ms, C

    def try_2opt(perm, ms, C):
        """
        Local search using 2-opt reversals.
        """
        improved = True
        max_passes = 10
        n_seq = len(perm)
        
        for _ in range(max_passes):
            if not improved:
                break
            improved = False
            best_move = None
            best_move_ms = ms
            best_move_C = None
            
            for i in range(n_seq):
                for j in range(i + 2, n_seq):
                    new_perm = perm[:]
                    new_perm[i:j+1] = new_perm[i:j+1][::-1]
                    
                    new_ms, new_C = compute_C(new_perm, p)
                    
                    if new_ms < best_move_ms - 1e-10:
                        best_move_ms = new_ms
                        best_move = (i, j)
                        best_move_C = new_C
            
            if best_move:
                i, j = best_move
                perm[i:j+1] = perm[i:j+1][::-1]
                ms = best_move_ms
                C = best_move_C
                improved = True
                
        return perm, ms, C

    best_perm, best_ms, best_C = try_adjacent_swaps(best_perm, best_ms, best_C)
    best_perm, best_ms, best_C = try_2opt(best_perm, best_ms, best_C)
    
    # Restarts
    best_overall = best_perm[:]
    best_overall_ms = best_ms
    
    rng = np.random.RandomState(42)
    for restart in range(10):
        perm = best_overall[:]
        for _ in range(3):
            i = rng.randint(n)
            j = rng.randint(n)
            if i != j:
                perm[i], perm[j] = perm[j], perm[i]
        
        ms, C = compute_C(perm, p)
        
        perm, ms, C = try_adjacent_swaps(perm, ms, C)
        perm, ms, C = try_2opt(perm, ms, C)
        
        if ms < best_overall_ms:
            best_overall = perm[:]
            best_overall_ms = ms
            
    return best_overall
# EVOLVE-BLOCK-END
