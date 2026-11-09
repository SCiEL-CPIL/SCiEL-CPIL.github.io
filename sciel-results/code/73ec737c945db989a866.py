# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Compute makespan for a given permutation."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    prev = [0.0] * n
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    
    for k in range(1, m):
        curr = [0.0] * n
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i-1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]


def _neh_heuristic(pt, sort_key='sum'):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    if sort_key == 'sum':
        totals = pt.sum(axis=1)
    elif sort_key == 'mean':
        totals = pt.mean(axis=1)
    elif sort_key == 'min':
        totals = pt.min(axis=1)
    elif sort_key == 'max':
        totals = pt.max(axis=1)
    else:
        totals = pt.sum(axis=1)
    
    order = np.argsort(-totals)
    
    perm = []
    
    for job in order:
        if not perm:
            perm = [job]
            continue
        
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=np.int64)


def _johnson_rule(pt):
    """
    Apply Johnson's rule for two-machine flow shop.
    We aggregate the first and last machines to create a pseudo 2-machine problem.
    For each job i:
      a_i = pt[i, 0]  (time on first machine)
      b_i = pt[i, -1] (time on last machine)
    
    Johnson's rule:
      - Partition jobs into set A (a_i <= b_i) and set B (a_i > b_i)
      - Sort A by increasing a_i
      - Sort B by decreasing b_i
      - Concatenate A + B
    """
    n, m = pt.shape
    if n <= 1:
        return np.array(range(n), dtype=np.int64)
    
    a = pt[:, 0]
    b = pt[:, -1]
    
    # Partition
    set_a_indices = []
    set_b_indices = []
    
    for i in range(n):
        if a[i] <= b[i]:
            set_a_indices.append(i)
        else:
            set_b_indices.append(i)
    
    # Sort A by increasing a_i
    set_a_indices.sort(key=lambda i: a[i])
    # Sort B by decreasing b_i
    set_b_indices.sort(key=lambda i: -b[i])
    
    perm = set_a_indices + set_b_indices
    return np.array(perm, dtype=np.int64)


def _local_search_2opt(perm, pt, max_iters=200, deadline=None):
    """2-opt local search for flow shop with time limit."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                return best_perm
            for j in range(i + 1, n):
                new_perm = best_perm.copy()
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                new_ms = _makespan_fast(new_perm, pt)
                if new_ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_swap(perm, pt, max_iters=150, deadline=None):
    """Swap-based local search for flow shop."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                return best_perm
            for j in range(i + 1, n):
                new_perm = best_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_ms = _makespan_fast(new_perm, pt)
                if new_ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_insert(perm, pt, max_iters=80, deadline=None):
    """Insertion-based local search: move one job to a different position."""
    n = len(perm)
    if n <= 2:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_perm
            job = best_perm[i]
            remaining = np.delete(best_perm, i)
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                new_perm = np.insert(remaining, pos, job)
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
    
    return best_perm


def _tabu_search(perm, pt, deadline, tabu_tenure=15, max_iters=3000):
    """Tabu search with insertion moves for flow shop."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    current_perm = perm.copy()
    current_ms = best_ms
    
    # Tabu list: store (i, j) moves, where job at position i moved to position j
    tabu_list = {}
    tabu_iter = 0
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        if tabu_iter % 500 == 0 and tabu_iter > 0:
            # Clear old tabu entries
            tabu_list = {k: v for k, v in tabu_list.items() if v > tabu_iter - tabu_tenure}
        
        best_move = None
        best_move_ms = float('inf')
        best_move_tabu = False
        best_move_aspiration = False
        
        # Generate all insertion moves
        for i in range(n):
            if time.time() >= deadline:
                return best_perm
            job = current_perm[i]
            remaining = np.delete(current_perm, i)
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                new_perm = np.insert(remaining, pos, job)
                ms = _makespan_fast(new_perm, pt)
                
                move_key = (i, pos)
                is_tabu = tabu_list.get(move_key, -1) >= tabu_iter - tabu_tenure
                
                # Aspiration: accept tabu move if it improves global best
                if is_tabu and ms < best_ms - 1e-9:
                    is_tabu = False
                    best_move_aspiration = True
                
                if not is_tabu or best_move_aspiration:
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = (i, pos)
                        best_move_tabu = is_tabu
                        best_move_aspiration = best_move_aspiration
        
        if best_move is None:
            break
        
        # Apply best move
        i, pos = best_move
        job = current_perm[i]
        remaining = np.delete(current_perm, i)
        current_perm = np.insert(remaining, pos, job)
        current_ms = best_move_ms
        
        # Update tabu list
        tabu_list[(i, pos)] = tabu_iter
        # Also tabu the reverse move
        reverse_pos = pos if pos > i else pos
        tabu_list[(pos, i)] = tabu_iter
        
        tabu_iter += 1
        
        if current_ms < best_ms - 1e-9:
            best_ms = current_ms
            best_perm = current_perm.copy()
    
    return best_perm


def _simulated_annealing(perm, pt, deadline, n_restarts=5):
    """Simulated annealing for flow shop to escape local optima."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            current_perm = best_perm.copy()
        else:
            # Perturb from best solution
            current_perm = best_perm.copy()
            for _ in range(min(5, n // 2)):
                i, j = np.random.choice(n, 2, replace=False)
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        current_ms = _makespan_fast(current_perm, pt)
        
        # SA parameters
        T = current_ms * 0.1
        cooling_rate = 0.95
        min_T = 1e-6
        max_iterations = 800
        
        for it in range(max_iterations):
            if time.time() >= deadline:
                break
            if T < min_T:
                break
            
            # Generate neighbor by random swap or insertion
            if np.random.random() < 0.5:
                i, j = np.random.choice(n, 2, replace=False)
                new_perm = current_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            else:
                i = np.random.randint(n)
                j = np.random.randint(n)
                new_perm = current_perm.copy()
                job = new_perm[i]
                new_perm = np.delete(new_perm, i)
                new_perm = np.insert(new_perm, j, job)
            
            new_ms = _makespan_fast(new_perm, pt)
            
            delta = new_ms - current_ms
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current_perm = new_perm
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_perm = current_perm.copy()
                    best_ms = current_ms
            
            T *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with multiple initial heuristics: NEH variants + Johnson's rule
    best_perm = None
    best_ms = float('inf')
    
    # NEH variants
    for sort_key in ['sum', 'mean', 'min', 'max']:
        if time.time() >= deadline:
            break
        perm = _neh_heuristic(pt, sort_key)
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_perm = perm
            best_ms = ms
    
    # Johnson's rule
    if time.time() < deadline:
        j_perm = _johnson_rule(pt)
        j_ms = _makespan_fast(j_perm, pt)
        if j_ms < best_ms:
            best_perm = j_perm
            best_ms = j_ms
    
    if best_perm is None:
        best_perm = _neh_heuristic(pt, 'sum')
        best_ms = _makespan_fast(best_perm, pt)
    
    # Local search with 2-opt
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_iters=200, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try swap local search
    if time.time() < deadline:
        swp_perm = _local_search_swap(best_perm, pt, max_iters=150, deadline=deadline)
        swp_ms = _makespan_fast(swp_perm, pt)
        if swp_ms < best_ms:
            best_perm = swp_perm
            best_ms = swp_ms
    
    # Try insertion local search
    if time.time() < deadline:
        ins_perm = _local_search_insert(best_perm, pt, max_iters=80, deadline=deadline)
        ins_ms = _makespan_fast(ins_perm, pt)
        if ins_ms < best_ms:
            best_perm = ins_perm
            best_ms = ins_ms
    
    # Tabu search to escape local optima
    if time.time() < deadline:
        tabu_perm = _tabu_search(best_perm, pt, deadline, tabu_tenure=15, max_iters=3000)
        tabu_ms = _makespan_fast(tabu_perm, pt)
        if tabu_ms < best_ms:
            best_perm = tabu_perm
            best_ms = tabu_ms
    
    # Simulated annealing to escape local optima
    if time.time() < deadline:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, n_restarts=5)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 8
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick local search from random start
            ls_r = _local_search_swap(random_perm, pt, max_iters=50, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # If improved, try insertion local search
            if time.time() < deadline:
                ls_r2 = _local_search_insert(ls_r, pt, max_iters=30, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
    
    # Final polish with swap local search
    if time.time() < deadline:
        final_perm = _local_search_swap(best_perm, pt, max_iters=100, deadline=deadline)
        final_ms = _makespan_fast(final_perm, pt)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    # Final polish with insertion local search
    if time.time() < deadline:
        final_perm2 = _local_search_insert(best_perm, pt, max_iters=50, deadline=deadline)
        final_ms2 = _makespan_fast(final_perm2, pt)
        if final_ms2 < best_ms:
            best_perm = final_perm2
            best_ms = final_ms2
    
    return list(best_perm)
# EVOLVE-BLOCK-END
