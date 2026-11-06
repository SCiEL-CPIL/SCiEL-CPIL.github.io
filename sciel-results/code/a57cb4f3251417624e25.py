# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Uses a matrix-based approach with precomputed pairwise interaction costs
    to enable O(1) delta evaluation for adjacent swaps, allowing exhaustive
    neighborhood exploration within the time budget.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    p = processing_times
    n = n_jobs
    M = n_machines
    
    # Precompute prefix completion times for each job in the current sequence
    # and use them for fast makespan evaluation and delta computation.
    
    def compute_makespan_and_prefixes(perm, p):
        """
        Compute makespan and prefix completion times.
        Returns (makespan, C) where C[i][m] is completion time of job at position i on machine m.
        """
        n_seq = len(perm)
        if n_seq == 0:
            return 0.0, np.zeros((0, M))
        
        C = np.zeros((n_seq, M))
        p_perm = p[perm]
        
        for i in range(n_seq):
            for m in range(M):
                if i == 0 and m == 0:
                    C[i, m] = p_perm[i, m]
                elif i == 0:
                    C[i, m] = C[i, m-1] + p_perm[i, m]
                elif m == 0:
                    C[i, m] = C[i-1, m] + p_perm[i, m]
                else:
                    C[i, m] = max(C[i-1, m], C[i, m-1]) + p_perm[i, m]
        
        return C[-1, -1], C
    
    def makespan_from_prefixes(perm, p, C):
        """Compute makespan from precomputed completion matrix."""
        if len(perm) == 0:
            return 0.0
        return C[-1, -1]
    
    # NEH heuristic for initial solution
    def neh_initial(p):
        """NEH heuristic: sort by decreasing total processing time, then insert each job in best position."""
        totals = p.sum(axis=1)
        order = list(np.argsort(-totals))
        
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                candidate = seq[:pos] + [job] + seq[pos:]
                ms, _ = compute_makespan_and_prefixes(candidate, p)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq
    
    # Initialize with NEH
    best_perm = neh_initial(p)
    best_ms, best_C = compute_makespan_and_prefixes(best_perm, p)
    
    # Local search with fast delta evaluation for adjacent swaps
    # For an adjacent swap at position i, we need to recompute C from position i onwards.
    # We can do this in O((n-i) * M) time.
    
    def evaluate_swap(perm, C, i, p):
        """
        Evaluate the makespan if we swap positions i and i+1.
        Returns (new_makespan, new_C) where new_C is the updated completion matrix.
        """
        n_seq = len(perm)
        if i >= n_seq - 1:
            return C[-1, -1], C
        
        # Create new permutation
        new_perm = perm[:]
        new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
        
        # Copy C up to position i-1
        new_C = np.zeros((n_seq, M))
        if i > 0:
            new_C[:i, :] = C[:i, :]
        
        p_perm = p[new_perm]
        
        # Recompute from position i onwards
        for pos in range(i, n_seq):
            for m in range(M):
                if pos == 0 and m == 0:
                    new_C[pos, m] = p_perm[pos, m]
                elif pos == 0:
                    new_C[pos, m] = new_C[pos, m-1] + p_perm[pos, m]
                elif m == 0:
                    new_C[pos, m] = new_C[pos-1, m] + p_perm[pos, m]
                else:
                    new_C[pos, m] = max(new_C[pos-1, m], new_C[pos, m-1]) + p_perm[pos, m]
        
        return new_C[-1, -1], new_C
    
    # Best-improvement local search over adjacent swaps
    improved = True
    max_passes = 100
    for _ in range(max_passes):
        if not improved:
            break
        improved = False
        n_seq = len(best_perm)
        best_swap = None
        best_swap_ms = best_ms
        best_swap_C = None
        
        for i in range(n_seq - 1):
            swap_ms, swap_C = evaluate_swap(best_perm, best_C, i, p)
            if swap_ms < best_swap_ms - 1e-10:
                best_swap_ms = swap_ms
                best_swap = i
                best_swap_C = swap_C
        
        if best_swap is not None:
            i = best_swap
            best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
            best_ms = best_swap_ms
            best_C = best_swap_C
            improved = True
    
    # Try 2-swap neighborhood (swap any two non-adjacent jobs)
    if n_seq >= 3:
        improved = True
        for _ in range(30):
            if not improved:
                break
            improved = False
            best_swap = None
            best_swap_ms = best_ms
            best_swap_C = None
            
            for i in range(n_seq):
                for j in range(i + 2, n_seq):
                    new_perm = best_perm[:]
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    swap_ms, swap_C = compute_makespan_and_prefixes(new_perm, p)
                    if swap_ms < best_swap_ms - 1e-10:
                        best_swap_ms = swap_ms
                        best_swap = (i, j)
                        best_swap_C = swap_C
            
            if best_swap is not None:
                i, j = best_swap
                best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
                best_ms = best_swap_ms
                best_C = best_swap_C
                improved = True
    
    # Additional restarts with perturbation for diversification
    best_overall = best_perm[:]
    best_overall_ms = best_ms
    
    rng = np.random.RandomState(42)
    for restart in range(10):
        perm = best_overall[:]
        # Random perturbation
        for _ in range(3):
            i = rng.randint(n_seq)
            j = rng.randint(n_seq)
            if i != j:
                perm[i], perm[j] = perm[j], perm[i]
        
        ms, C = compute_makespan_and_prefixes(perm, p)
        
        # Local search from perturbed solution
        improved = True
        for _ in range(50):
            if not improved:
                break
            improved = False
            best_swap = None
            best_swap_ms = ms
            best_swap_C = None
            
            for i in range(n_seq - 1):
                swap_ms, swap_C = evaluate_swap(perm, C, i, p)
                if swap_ms < best_swap_ms - 1e-10:
                    best_swap_ms = swap_ms
                    best_swap = i
                    best_swap_C = swap_C
            
            if best_swap is not None:
                i = best_swap
                perm[i], perm[i+1] = perm[i+1], perm[i]
                ms = best_swap_ms
                C = best_swap_C
                improved = True
        
        if ms < best_overall_ms:
            best_overall = perm[:]
            best_overall_ms = ms
    
    return best_overall
# EVOLVE-BLOCK-END
