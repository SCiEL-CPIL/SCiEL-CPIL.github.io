# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_routes(n=10):
    """Precompute which edges each demand uses for CW and CCW routing."""
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        
        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)
    
    return cw_edges, ccw_edges, demands


def _compute_loads(routing, cw_edges, ccw_edges, demands):
    """Compute edge loads given a routing vector."""
    n = 10
    D = len(demands)
    loads = np.zeros(n)
    
    for d in range(D):
        s, t, w = demands[d]
        if routing[d] >= 0.5:
            for e in ccw_edges[d]:
                loads[e] += w
        else:
            for e in cw_edges[d]:
                loads[e] += w
    
    return loads


def _compute_max_load(routing, cw_edges, ccw_edges, demands):
    """Compute just the maximum load for a routing."""
    loads = _compute_loads(routing, cw_edges, ccw_edges, demands)
    return np.max(loads)


def _exact_search(cw_edges, ccw_edges, demands, deadline):
    """Exact brute-force search using vectorized numpy operations.
    For D=15, there are 2^15 = 32768 combinations. We process them in
    chunks using matrix operations for speed."""
    D = len(demands)
    n = 10
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # Build incidence matrices: for each demand, which edges are used in CW and CCW
    # cw_inc[d, e] = 1 if demand d uses edge e in CW routing
    # ccw_inc[d, e] = 1 if demand d uses edge e in CCW routing
    cw_inc = np.zeros((D, n))
    ccw_inc = np.zeros((D, n))
    for d in range(D):
        for e in cw_edges[d]:
            cw_inc[d, e] = 1.0
        for e in ccw_edges[d]:
            ccw_inc[d, e] = 1.0
    
    # For a routing vector r (0=CW, 1=CCW), the load on edge e is:
    # load[e] = sum_d (r[d] * ccw_inc[d,e] + (1-r[d]) * cw_inc[d,e]) * weight[d]
    #         = sum_d cw_inc[d,e] * weight[d] + sum_d r[d] * (ccw_inc[d,e] - cw_inc[d,e]) * weight[d]
    # Let base[e] = sum_d cw_inc[d,e] * weight[d]
    # Let delta[e] = sum_d r[d] * (ccw_inc[d,e] - cw_inc[d,e]) * weight[d]
    # load[e] = base[e] + delta[e]
    
    base = cw_inc.T @ weights  # shape (n,)
    diff = (ccw_inc - cw_inc) * weights  # shape (D, n), diff[d,e] = (ccw_inc[d,e]-cw_inc[d,e])*weight[d]
    
    # Now for each routing combination, load = base + diff.T @ r
    # We want to minimize max(load)
    
    total_combos = 1 << D
    chunk_size = 10000
    best_max = float('inf')
    best_routing = np.zeros(D)
    
    for start in range(0, total_combos, chunk_size):
        if time.time() > deadline:
            break
        end = min(start + chunk_size, total_combos)
        count = end - start
        
        # Generate routing vectors for this chunk
        # For each combination, r[d] = bit d of the combination
        combos = np.arange(start, end, dtype=np.uint32)
        # Extract bits: r[d, i] = (combos[i] >> d) & 1
        bits = (combos[:, np.newaxis] >> np.arange(D, dtype=np.uint32)[np.newaxis, :]) & 1
        r = bits.astype(np.float64)  # shape (count, D)
        
        # Compute loads: load[i, e] = base[e] + sum_d r[i,d] * diff[d,e]
        # delta[i, e] = r[i] @ diff[:, e] = (diff.T @ r[i])[e]
        # Using matrix multiply: loads = base[None, :] + r @ diff
        loads = base[None, :] + r @ diff  # shape (count, n)
        
        max_loads = np.max(loads, axis=1)  # shape (count,)
        min_idx = np.argmin(max_loads)
        if max_loads[min_idx] < best_max:
            best_max = max_loads[min_idx]
            best_routing = r[min_idx].copy()
    
    return best_routing, best_max


def _local_search(routing, cw_edges, ccw_edges, demands, deadline, max_iter=100000):
    """Local search with 1-flip moves."""
    D = len(demands)
    best_routing = routing.copy()
    best_max = _compute_max_load(best_routing, cw_edges, ccw_edges, demands)
    
    improved = True
    iters = 0
    while improved and (time.time() < deadline) and (iters < max_iter):
        improved = False
        iters += 1
        for d in range(D):
            trial = best_routing.copy()
            trial[d] = 1.0 - trial[d]
            trial_max = _compute_max_load(trial, cw_edges, ccw_edges, demands)
            if trial_max < best_max - 1e-9:
                best_routing = trial
                best_max = trial_max
                improved = True
    
    return best_routing, best_max


def _pair_local_search(routing, cw_edges, ccw_edges, demands, deadline, max_iter=50000):
    """Local search with 2-flip moves."""
    D = len(demands)
    best_routing = routing.copy()
    best_max = _compute_max_load(best_routing, cw_edges, ccw_edges, demands)
    
    improved = True
    iters = 0
    while improved and (time.time() < deadline) and (iters < max_iter):
        improved = False
        iters += 1
        for d1 in range(D):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, D):
                trial = best_routing.copy()
                trial[d1] = 1.0 - trial[d1]
                trial[d2] = 1.0 - trial[d2]
                trial_max = _compute_max_load(trial, cw_edges, ccw_edges, demands)
                if trial_max < best_max - 1e-9:
                    best_routing = trial
                    best_max = trial_max
                    improved = True
                    break
            if time.time() > deadline:
                break
    
    return best_routing, best_max


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using exact search for small D,
    with heuristic fallback for larger instances.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    cw_edges, ccw_edges, demands = _build_routes()
    D = len(demands)
    n = 10
    
    best_routing = None
    best_max = float('inf')
    
    # For D=15, exact search is feasible (2^15 = 32768 combos)
    if D <= 20:
        er, em = _exact_search(cw_edges, ccw_edges, demands, deadline - 5)
        if em < best_max:
            best_max = em
            best_routing = er.copy()
        
        # Polish with local search to ensure we're at a local optimum
        if best_routing is not None and time.time() < deadline - 2:
            lr, lm = _local_search(best_routing, cw_edges, ccw_edges, demands, deadline, max_iter=10000)
            if lm < best_max - 1e-9:
                best_max = lm
                best_routing = lr.copy()
    else:
        # Fallback: heuristic search for larger D
        np_random = np.random.default_rng(2024)
        
        for restart in range(50):
            if time.time() > deadline - 30:
                break
            routing = np_random.random(D)
            lr, lm = _local_search(routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=50000)
            if lm < best_max - 1e-9:
                best_max = lm
                best_routing = lr.copy()
        
        if best_routing is not None and time.time() < deadline - 20:
            pr, pm = _pair_local_search(best_routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=20000)
            if pm < best_max - 1e-9:
                best_max = pm
                best_routing = pr.copy()
    
    if best_routing is None:
        best_routing = np.zeros(D)
    
    return best_routing


# EVOLVE-BLOCK-END
