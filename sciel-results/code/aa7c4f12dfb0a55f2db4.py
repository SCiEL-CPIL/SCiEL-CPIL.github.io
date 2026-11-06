# EVOLVE-BLOCK-START
import numpy as np
import time


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses NEH initialization + fast local search with completion-matrix-based
    delta evaluation for adjacent swaps, plus 2-opt moves and ILS restarts.
    """
    n_jobs, n_machines = processing_times.shape
    n = n_jobs
    M = n_machines
    p = processing_times

    if n <= 1:
        return list(range(n))

    # Time budget: 115 seconds (leaving 5s margin)
    t_start = time.time()
    time_limit = 115.0

    def compute_C(perm, p):
        """Compute completion time matrix C for a permutation."""
        n_seq = len(perm)
        if n_seq == 0:
            return np.zeros((0, M))
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
        return C

    def makespan(C):
        if len(C) == 0:
            return 0.0
        return C[-1, -1]

    # NEH heuristic
    def neh_initial(p):
        totals = p.sum(axis=1)
        order = list(np.argsort(-totals))
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                candidate = seq[:pos] + [job] + seq[pos:]
                C = compute_C(candidate, p)
                ms = makespan(C)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq

    # Fast swap evaluation: recompute C from position i onwards
    def eval_adj_swap(perm, C, i, p):
        """Evaluate makespan after swapping positions i and i+1. Returns (ms, new_C)."""
        n_seq = len(perm)
        if i >= n_seq - 1:
            return makespan(C), C.copy()
        
        new_perm = perm[:]
        new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
        
        new_C = C.copy()
        p_perm = p[new_perm]
        
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

    # First-improvement adjacent swap local search
    def local_search_adj(perm, C, max_passes=200):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            n_seq = len(perm)
            for i in range(n_seq - 1):
                swap_ms, swap_C = eval_adj_swap(perm, C, i, p)
                if swap_ms < makespan(C) - 1e-10:
                    perm[i], perm[i+1] = perm[i+1], perm[i]
                    C = swap_C
                    improved = True
                    # Continue from the same position to catch cascading improvements
                    if i > 0:
                        i = i - 1
                        if i < 0:
                            i = 0
        return perm, C

    # 2-opt move: swap any two positions
    def local_search_2opt(perm, C, max_passes=20):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            n_seq = len(perm)
            best_swap = None
            best_ms = makespan(C)
            best_C = None
            for i in range(n_seq):
                for j in range(i + 2, n_seq):
                    new_perm = perm[:]
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_C = compute_C(new_perm, p)
                    ms = makespan(new_C)
                    if ms < best_ms - 1e-10:
                        best_ms = ms
                        best_swap = (i, j)
                        best_C = new_C
            if best_swap is not None:
                i, j = best_swap
                perm[i], perm[j] = perm[j], perm[i]
                C = best_C
                improved = True
        return perm, C

    # Initialize with NEH
    best_perm = neh_initial(p)
    best_C = compute_C(best_perm, p)
    best_ms = makespan(best_C)

    # Local search
    best_perm, best_C = local_search_adj(best_perm, best_C)
    best_ms = makespan(best_C)

    if n >= 3:
        best_perm, best_C = local_search_2opt(best_perm, best_C)
        best_ms = makespan(best_C)

    # ILS with restarts
    rng = np.random.RandomState(42)
    for restart in range(50):
        if time.time() - t_start > time_limit - 5:
            break
        
        # Perturb
        perm = best_perm[:]
        # Random 2-opt perturbation
        n_pert = max(2, n // 5)
        for _ in range(n_pert):
            i = rng.randint(n)
            j = rng.randint(n)
            if i != j:
                perm[i], perm[j] = perm[j], perm[i]
        
        C = compute_C(perm, p)
        ms = makespan(C)
        
        # Local search
        perm, C = local_search_adj(perm, C, max_passes=100)
        ms = makespan(C)
        
        if n >= 3 and time.time() - t_start < time_limit - 10:
            perm, C = local_search_2opt(perm, C, max_passes=10)
            ms = makespan(C)
        
        if ms < best_ms - 1e-10:
            best_perm = perm[:]
            best_C = C
            best_ms = ms
    
    return best_perm
# EVOLVE-BLOCK-END
