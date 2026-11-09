"""Flow-shop makespan minimizer: NEH + fast local search with tabu search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """
    Calculate makespan for a permutation using the standard recurrence.
    perm: 1-D array of job indices (length n_jobs)
    pt: (n_jobs, n_machines) processing times
    Returns: float makespan
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(pt[perm[0]].sum())
    
    p = pt[perm]  # (n_jobs, n_machines)
    c = np.zeros(n_jobs, dtype=np.float64)
    
    for m in range(n_machines):
        c[0] = p[0, m]
        for j in range(1, n_jobs):
            c[j] = max(c[j], c[j-1]) + p[j, m]
    
    return float(c[-1])


def _makespan_fast(perm, pt):
    """
    Faster makespan calculation using vectorized numpy operations.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(pt[perm[0]].sum())
    
    p = pt[perm]  # (n_jobs, n_machines)
    
    # Build completion time matrix
    # C[j, m] = completion time of job j on machine m
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    C[:, 0] = np.cumsum(p[:, 0])
    
    for m in range(1, n_machines):
        C[:, m] = np.maximum(C[:, m-1], C[:, m]) + p[:, m]
        # Actually need to be more careful
        # C[j, m] = max(C[j, m-1], C[j-1, m]) + p[j, m]
        # Let's do it properly
        C[:, m] = C[:, m]  # this won't work, let's use loop
    
    # The vectorized approach is tricky due to dependencies. Let's use a different approach.
    # Actually, let's just use the original but with some numpy tricks.
    
    # Build the full completion matrix
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    C[0, :] = p[0, :]
    
    for j in range(1, n_jobs):
        C[j, 0] = C[j-1, 0] + p[j, 0]
        for m in range(1, n_machines):
            C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    
    return float(C[-1, -1])


def _makespan_delta_swap(perm, pt, i, j):
    """
    Calculate the change in makespan when swapping positions i and j.
    This is an approximation - we recompute the full makespan for correctness.
    """
    new_perm = perm.copy()
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return _makespan(new_perm, pt) - _makespan(perm, pt)


def _neh(processing_times):
    """
    NEH heuristic: order jobs by total processing time desc, insert each optimally.
    """
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(i + 1):
            cand = perm[:pos] + [job] + perm[pos:]
            cost = _makespan(np.array(cand, dtype=np.int64), processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=np.int64)


def _neh_variant(processing_times, sort_key='total'):
    """
    NEH with different sorting keys.
    """
    n_jobs, n_machines = processing_times.shape
    
    if sort_key == 'total':
        order = list(np.argsort(-processing_times.sum(axis=1)))
    elif sort_key == 'first':
        order = list(np.argsort(-processing_times[:, 0]))
    elif sort_key == 'last':
        order = list(np.argsort(-processing_times[:, -1]))
    elif sort_key == 'avg':
        order = list(np.argsort(-processing_times.mean(axis=1)))
    else:
        order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(i + 1):
            cand = perm[:pos] + [job] + perm[pos:]
            cost = _makespan(np.array(cand, dtype=np.int64), processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=np.int64)


def _cig(processing_times):
    """
    CIG (Constructive Insertion Greedy) heuristic.
    Insert jobs one by one in the position that minimizes the increase in makespan.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Start with the job that has the largest total processing time
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(len(perm) + 1):
            cand = perm[:pos] + [job] + perm[pos:]
            cost = _makespan(np.array(cand, dtype=np.int64), processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=np.int64)


def _local_search_insertion(perm, pt, deadline, margin=3.0):
    """
    Local search using insertion moves: remove job at i and insert at j.
    Returns improved permutation and its cost.
    """
    n_jobs = len(perm)
    best = perm.copy()
    best_cost = _makespan(best, pt)
    improved = True
    
    while improved:
        improved = False
        
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(n_jobs):
                if j == i:
                    continue
                cand = list(best)
                job = cand.pop(i)
                insert_pos = j if j <= i else j - 1
                cand.insert(insert_pos, job)
                cand_arr = np.array(cand, dtype=np.int64)
                cost = _makespan(cand_arr, pt)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand_arr
                    improved = True
                    break
            if improved:
                break
    
    return best, best_cost


def _local_search_swap(perm, pt, deadline, margin=3.0):
    """
    Local search using swap moves: exchange two positions i, j.
    """
    n_jobs = len(perm)
    best = perm.copy()
    best_cost = _makespan(best, pt)
    improved = True
    
    while improved:
        improved = False
        
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                cost = _makespan(cand, pt)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
    
    return best, best_cost


def _local_search_2opt(perm, pt, deadline, margin=3.0):
    """
    Local search using 2-opt (segment reversal) moves.
    """
    n_jobs = len(perm)
    best = perm.copy()
    best_cost = _makespan(best, pt)
    improved = True
    
    while improved:
        improved = False
        
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i:j + 1] = cand[i:j + 1][::-1]
                cost = _makespan(cand, pt)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
    
    return best, best_cost


def _tabu_search(perm, pt, deadline, margin, rng, tabu_tenure=10, max_iter=5000):
    """
    Tabu search with insertion moves.
    """
    n_jobs = len(perm)
    
    current = perm.copy()
    current_cost = _makespan(current, pt)
    best = current.copy()
    best_cost = current_cost
    
    # Tabu list: store (i, j) moves that are forbidden
    tabu_list = []  # list of (i, j, expiry_iter)
    tabu_memory = {}  # (i, j) -> expiry_iter
    
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time.monotonic() > deadline - margin:
            break
        
        # Generate all insertion moves
        best_move = None
        best_move_cost = float('inf')
        best_move_i = -1
        best_move_j = -1
        is_tabu = False
        
        for i in range(n_jobs):
            for j in range(n_jobs):
                if j == i:
                    continue
                insert_pos = j if j <= i else j - 1
                
                # Check if this move is tabu
                move_key = (i, insert_pos)
                if move_key in tabu_memory and tabu_memory[move_key] > it:
                    # Check aspiration criterion: if it improves best, allow it
                    cand = list(current)
                    job = cand.pop(i)
                    cand.insert(insert_pos, job)
                    cand_arr = np.array(cand, dtype=np.int64)
                    cand_cost = _makespan(cand_arr, pt)
                    if cand_cost >= best_cost:
                        continue  # Skip tabu move that doesn't improve best
                    # Aspiration: allow tabu move if it improves best
                    is_tabu = False
                else:
                    is_tabu = False
                
                cand = list(current)
                job = cand.pop(i)
                cand.insert(insert_pos, job)
                cand_arr = np.array(cand, dtype=np.int64)
                cand_cost = _makespan(cand_arr, pt)
                
                if cand_cost < best_move_cost:
                    best_move_cost = cand_cost
                    best_move_i = i
                    best_move_j = insert_pos
                    best_move = cand_arr
        
        if best_move is None:
            break
        
        # Update tabu list
        move_key = (best_move_i, best_move_j)
        tabu_memory[move_key] = it + tabu_tenure
        
        # Accept the move
        current = best_move
        current_cost = best_move_cost
        
        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Diversification: if stuck for too long, perturb
        if no_improve_count > max_no_improve:
            # Random restart from best
            current = best.copy()
            current_cost = best_cost
            no_improve_count = 0
            # Clear tabu list
            tabu_memory = {}
            # Perturb
            for _ in range(3):
                i = rng.integers(0, n_jobs)
                j = rng.integers(0, n_jobs)
                if i == j:
                    continue
                cand = list(current)
                job = cand.pop(i)
                insert_pos = j if j <= i else j - 1
                cand.insert(insert_pos, job)
                current = np.array(cand, dtype=np.int64)
                current_cost = _makespan(current, pt)
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
    
    return best, best_cost


def _multi_start(processing_times, deadline, margin, rng):
    """
    Multi-start with different constructive heuristics and local search.
    """
    n_jobs, n_machines = processing_times.shape
    
    best_perm = None
    best_cost = float('inf')
    
    # Try multiple constructive heuristics
    heuristics = ['total', 'first', 'last', 'avg']
    
    for h in heuristics:
        if time.monotonic() > deadline - margin - 20.0:
            break
        
        perm = _neh_variant(processing_times, sort_key=h)
        cost = _makespan(perm, processing_times)
        
        if cost < best_cost:
            best_perm = perm
            best_cost = cost
    
    # Also try CIG
    if time.monotonic() < deadline - margin - 20.0:
        cig_perm = _cig(processing_times)
        cig_cost = _makespan(cig_perm, processing_times)
        if cig_cost < best_cost:
            best_perm = cig_perm
            best_cost = cig_cost
    
    # Local search polish
    if best_perm is not None and time.monotonic() < deadline - margin - 5.0:
        best_perm, best_cost = _local_search_insertion(best_perm, processing_times, deadline, margin + 5.0)
        if time.monotonic() < deadline - margin - 3.0:
            best_perm, best_cost = _local_search_swap(best_perm, processing_times, deadline, margin + 3.0)
    
    # Tabu search
    if best_perm is not None and time.monotonic() < deadline - margin - 10.0:
        ts_best, ts_cost = _tabu_search(best_perm, processing_times, deadline, margin, rng)
        if ts_cost < best_cost:
            best_perm = ts_best
            best_cost = ts_cost
    
    return best_perm, best_cost


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    deadline = t0 + 145.0  # Safety margin (150s limit, 5s buffer)
    margin = 5.0

    rng = np.random.default_rng(42)

    # Multi-start with diverse heuristics + local search + tabu
    best_perm, best_cost = _multi_start(processing_times, deadline, margin, rng)

    # If we still have time, do additional SA-like exploration
    if time.monotonic() < deadline - margin - 15.0 and best_perm is not None:
        current = best_perm.copy()
        current_cost = best_cost
        
        T = max(best_cost * 0.05, 1.0)
        T_final = max(best_cost * 0.001, 0.01)
        
        while time.monotonic() < deadline - margin and T > T_final:
            # Random insertion move
            i = rng.integers(0, n_jobs)
            j = rng.integers(0, n_jobs)
            if i == j:
                T *= 0.995
                continue
            
            cand = list(current)
            job = cand.pop(i)
            insert_pos = j if j <= i else j - 1
            cand.insert(insert_pos, job)
            cand_arr = np.array(cand, dtype=np.int64)
            cand_cost = _makespan(cand_arr, processing_times)
            
            delta = cand_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current = cand_arr
                current_cost = cand_cost
                if current_cost < best_cost:
                    best_perm = current.copy()
                    best_cost = current_cost
            
            T *= 0.995
        
        # Final polish
        if time.monotonic() < deadline - margin:
            best_perm, best_cost = _local_search_insertion(best_perm, processing_times, deadline, margin)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
