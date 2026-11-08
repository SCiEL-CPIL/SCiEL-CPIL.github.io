# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Compute makespan for a permutation. O(n*m)."""
    # perm is array of job indices
    # pt[job, machine] is processing time
    # We need to compute completion times on last machine
    # C[m] = max(C[m-1], C_prev_machine[m]) + pt[job_m, m]
    # But since it's flow shop, the last machine completion is:
    # sum over jobs in order of max over suffix... actually simpler:
    # For flow shop, makespan = sum of processing times on last machine
    # + max over all prefixes of (sum of processing times on machine k for prefix)
    # Actually the standard formula:
    # Let C_j be completion time of job j on last machine.
    # C_j = max(C_{j-1}, C'_j) + pt[j, m-1] where C'_j is completion on machine m-2
    # This is complex. Let's just simulate.
    
    n = len(perm)
    m = pt.shape[1]
    if m == 0:
        return 0.0
    if n == 0:
        return 0.0
    
    # Simulate the flow shop
    # completion[i][j] = completion time of job i on machine j
    # But we only need the last machine.
    # Let's use the recurrence:
    # Let T[j] = completion time of the j-th job in permutation on the last machine
    # This requires knowing completion on previous machines.
    
    # Efficient simulation:
    # For each machine, we process jobs in order.
    # Let's track completion time of each job on each machine, but only keep last machine.
    
    # Actually, the makespan for flow shop can be computed as:
    # MS = sum_{j in perm} pt[j, m-1] + max_{k=0..m-2} max_{i=0..n-1} (sum_{j=0..i} pt[perm[j], k] - sum_{j=0..i} pt[perm[j], m-1])
    # No, that's not right either.
    
    # Let's just do the O(n*m) simulation directly.
    # completion[j] = completion time of job perm[j] on current machine
    # We iterate through machines, and for each machine, compute completion times.
    
    # Start with machine 0:
    # completion on machine 0 for job i is sum of pt[perm[0..i], 0]
    
    # For machine k, completion[perm[i]] = max(completion_prev[perm[i]], completion_current[perm[i-1]]) + pt[perm[i], k]
    
    # Let's do it properly.
    prev_completion = np.cumsum(pt[perm, 0])  # completion on machine 0
    for k in range(1, m):
        curr_completion = np.empty(n, dtype=np.float64)
        curr_completion[0] = prev_completion[0] + pt[perm[0], k]
        for i in range(1, n):
            curr_completion[i] = max(prev_completion[i], curr_completion[i-1]) + pt[perm[i], k]
        prev_completion = curr_completion
    
    return float(prev_completion[-1])


def _neh_heuristic(pt):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build permutation greedily
    perm = []
    remaining = list(order)
    
    for job in remaining:
        # Try inserting job at each position in current perm
        best_pos = 0
        best_ms = float('inf')
        
        if not perm:
            perm = [job]
            continue
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=np.int64)


def _local_search(perm, pt, max_iters=1000):
    """2-opt local search for flow shop."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_perm = best_perm.copy()
                new_perm[i:j+1] = best_perm[i:j+1][::-1]
                ms = _makespan(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan(best_perm, pt)
    
    # Local search
    if time.time() < deadline:
        ls_perm = _local_search(best_perm, pt, max_iters=500)
        ls_ms = _makespan(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try multiple NEH variants with different sorting criteria
    if time.time() < deadline:
        # Sort by average processing time
        avgs = pt.mean(axis=1)
        order2 = np.argsort(-avgs)
        perm2 = list(order2)
        # Greedy insertion
        perm2_final = []
        for job in order2:
            best_pos = 0
            best_ms2 = float('inf')
            if not perm2_final:
                perm2_final = [job]
                continue
            for pos in range(len(perm2_final) + 1):
                new_perm = perm2_final[:pos] + [job] + perm2_final[pos:]
                ms = _makespan(np.array(new_perm, dtype=np.int64), pt)
                if ms < best_ms2:
                    best_ms2 = ms
                    best_pos = pos
            perm2_final.insert(best_pos, job)
        perm2_arr = np.array(perm2_final, dtype=np.int64)
        ms2 = _makespan(perm2_arr, pt)
        if ms2 < best_ms:
            best_perm = perm2_arr
            best_ms = ms2
        if time.time() < deadline:
            ls_perm2 = _local_search(best_perm, pt, max_iters=300)
            ls_ms2 = _makespan(ls_perm2, pt)
            if ls_ms2 < best_ms:
                best_perm = ls_perm2
                best_ms = ls_ms2
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        for _ in range(5):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n)
            ms_r = _makespan(random_perm, pt)
            if ms_r < best_ms:
                best_perm = random_perm
                best_ms = ms_r
            # Local search from random start
            ls_r = _local_search(random_perm, pt, max_iters=200)
            ls_ms_r = _makespan(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
    
    return list(best_perm)
# EVOLVE-BLOCK-END
