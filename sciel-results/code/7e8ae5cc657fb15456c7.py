# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        for mi in range(m):
            prev_machine = comp[mi - 1] if mi > 0 else 0.0
            same_machine_prev = comp[mi]
            comp[mi] = max(same_machine_prev, prev_machine) + pt[job, mi]
    return comp[-1]

def _makespan_fast(perm, pt):
    """Optimized makespan computation with local variable caching."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = [0.0] * m
    pt_local = pt
    for j in range(n):
        job = perm[j]
        row = pt_local[job]
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            cur = comp[mi]
            if cur >= prev:
                comp[mi] = cur + row[mi]
            else:
                comp[mi] = prev + row[mi]
    return comp[m - 1]

def _neh_initial(pt):
    """NEH heuristic: sort jobs by decreasing total processing time."""
    total_pt = pt.sum(axis=1)
    return np.argsort(-total_pt).astype(int)

def _palmer_initial(pt):
    """Palmer's heuristic: sort by decreasing Palmer key."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    keys = np.zeros(n)
    for j in range(n):
        for mi in range(m):
            keys[j] += (n - 1 - 2 * mi) * pt[j, mi]
    return np.argsort(-keys).astype(int)

def _bottleneck_initial(pt):
    """Bottleneck-based: sort by decreasing processing time on the bottleneck machine."""
    n, m = pt.shape
    if m == 0:
        return np.arange(n)
    machine_totals = pt.sum(axis=0)
    bottleneck_machine = np.argmax(machine_totals)
    return np.argsort(-pt[:, bottleneck_machine]).astype(int)

def _first_machine_initial(pt):
    """Sort by decreasing processing time on the first machine."""
    return np.argsort(-pt[:, 0]).astype(int)

def _last_machine_initial(pt):
    """Sort by decreasing processing time on the last machine."""
    return np.argsort(-pt[:, -1]).astype(int)

def _random_initial(n, rng):
    """Random permutation."""
    return rng.permutation(n)

def _insertion_ls_best(perm, pt, time_limit=5.0, start_time=None):
    """Best-improvement insertion local search: find the best move in each iteration."""
    if start_time is None:
        start_time = time.time()
    
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_move = None
        best_move_ms = best_ms
        
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            for j in range(n):
                if j == i:
                    continue
                # Create new permutation by removing job from i and inserting at j
                new_perm = best_perm.copy()
                remaining = np.delete(new_perm, i)
                if j > i:
                    j_adj = j - 1
                else:
                    j_adj = j
                new_perm = np.insert(remaining, j_adj, job)
                ms = _makespan_fast(new_perm, pt)
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = (i, j_adj, job, new_perm)
        
        if best_move is not None:
            best_perm = best_move[3]
            best_ms = best_move_ms
            improved = True
    
    return best_perm, best_ms

def _insertion_ls_first(perm, pt, time_limit=5.0, start_time=None):
    """First-improvement insertion local search."""
    if start_time is None:
        start_time = time.time()
    
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            for j in range(n):
                if j == i:
                    continue
                new_perm = best_perm.copy()
                remaining = np.delete(new_perm, i)
                if j > i:
                    j_adj = j - 1
                else:
                    j_adj = j
                new_perm = np.insert(remaining, j_adj, job)
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    i = j_adj
                    break
    
    return best_perm, best_ms

def _reversal_ls_best(perm, pt, time_limit=5.0, start_time=None):
    """Best-improvement segment reversal local search."""
    if start_time is None:
        start_time = time.time()
    
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_move = None
        best_move_ms = best_ms
        
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 2, n):
                if time.time() - start_time >= time_limit:
                    break
                new_perm = best_perm.copy()
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = (i, j, new_perm)
        
        if best_move is not None:
            best_perm = best_move[2]
            best_ms = best_move_ms
            improved = True
    
    return best_perm, best_ms

def _polish_pass(perm, pt, time_limit=5.0, start_time=None):
    """High-precision polish: alternate best-improvement insertion and reversal with tight tolerance."""
    if start_time is None:
        start_time = time.time()
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    # Alternate between insertion and reversal with best-improvement
    while time.time() - start_time < time_limit:
        # Insertion pass
        old_ms = best_ms
        best_perm, best_ms = _insertion_ls_best(best_perm, pt, time_limit=1.0, start_time=start_time)
        
        if time.time() - start_time >= time_limit:
            break
        
        # Reversal pass
        best_perm, best_ms = _reversal_ls_best(best_perm, pt, time_limit=1.0, start_time=start_time)
        
        # Check for convergence
        if best_ms >= old_ms - 1e-10:
            # No improvement, try one more insertion
            old_ms = best_ms
            best_perm, best_ms = _insertion_ls_best(best_perm, pt, time_limit=0.5, start_time=start_time)
            if best_ms >= old_ms - 1e-10:
                break
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    effective_deadline = deadline - safety_margin
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial solutions using different heuristics
    initial_solutions = []
    
    # NEH
    perm_neh = _neh_initial(processing_times)
    initial_solutions.append(perm_neh)
    
    # Palmer
    perm_palmer = _palmer_initial(processing_times)
    initial_solutions.append(perm_palmer)
    
    # Bottleneck
    perm_bn = _bottleneck_initial(processing_times)
    initial_solutions.append(perm_bn)
    
    # First machine
    perm_first = _first_machine_initial(processing_times)
    initial_solutions.append(perm_first)
    
    # Last machine
    perm_last = _last_machine_initial(processing_times)
    initial_solutions.append(perm_last)
    
    # Add some random solutions
    for _ in range(10):
        perm_rand = _random_initial(n, rng)
        initial_solutions.append(perm_rand)
    
    # Evaluate all initial solutions
    best_perm = None
    best_ms = float('inf')
    
    for perm in initial_solutions:
        ms = _makespan_fast(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
    
    # Phase 1: Best-improvement insertion local search
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 1.0:
        best_perm, best_ms = _insertion_ls_best(best_perm, processing_times, time_limit=min(15.0, remaining), start_time=start_time)
    
    # Phase 2: Best-improvement reversal local search
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 1.0:
        best_perm, best_ms = _reversal_ls_best(best_perm, processing_times, time_limit=min(10.0, remaining), start_time=start_time)
    
    # Phase 3: Multi-start with alternative first jobs to escape local optima
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 10.0:
        # Try different initial solutions with local search
        num_restarts = min(3, int(remaining // 15))
        for restart in range(num_restarts):
            if time.time() - start_time >= effective_deadline:
                break
            
            # Use a different initial solution
            init_idx = (restart + 1) % len(initial_solutions)
            perm_init = initial_solutions[init_idx].copy()
            
            # Apply best-improvement insertion search
            perm_candidate, ms_candidate = _insertion_ls_best(perm_init, processing_times, time_limit=5.0, start_time=start_time)
            
            if ms_candidate < best_ms:
                best_ms = ms_candidate
                best_perm = perm_candidate.copy()
    
    # Phase 4: Final high-precision polish pass
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 2.0:
        best_perm, best_ms = _polish_pass(best_perm, processing_times, time_limit=min(20.0, remaining), start_time=start_time)
    
    # Final safety check
    if time.time() - start_time > effective_deadline:
        # Return whatever we have
        pass
    
    return list(best_perm)
# EVOLVE-BLOCK-END
