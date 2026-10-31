# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0          # explicit safety margin vs 150s wall-clock kill
_TOUR = None                             # precomputed tour: list of city indices
_FINGERPRINT = None
_DONE = False
_FALLBACK_W = 0.0                        # pure NN fallback if tour not ready

def _elapsed():
    return time.time() - _START

_START = time.time()

def _fingerprint(d):
    """Cheap hash of the distance matrix to cache tours across calls."""
    # Use a small set of entries + shape as a fingerprint
    n = d.shape[0]
    idx = np.linspace(0, n * n - 1, min(64, n * n), dtype=int)
    vals = d.ravel()[idx]
    return (n, tuple(vals.astype(np.float64).round(6).tolist()))

def _tour_length(tour, d):
    L = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        L += d[a, b]
    return L

def _nn_tour(d):
    n = d.shape[0]
    start = 0
    unv = set(range(1, n))
    cur = start
    tour = [start]
    while unv:
        arr = np.fromiter(unv, dtype=int)
        g = d[cur, arr]
        nxt = int(arr[int(np.argmin(g))])
        cur = nxt
        tour.append(cur)
        unv.discard(cur)
    return tour

def _two_opt(tour, d):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, dd = tour[j], tour[(j + 1) % n]
                delta = d[a, c] + d[b, dd] - d[a, b] - d[c, dd]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour

def _or_opt(tour, d):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            a, b, c = tour[i], tour[(i + 1) % n], tour[(i + 2) % n]
            # Try moving segment of length 1
            for L in [1, 2, 3]:
                if i + L >= n:
                    continue
                seg = tour[i:i + L]
                if not seg:
                    continue
                prev = tour[i - 1]
                nxt = tour[i + L]
                # Remove seg
                old_cost = d[prev, seg[0]] + d[seg[-1], nxt]
                new_cost = d[prev, nxt]
                # Insert after position j
                for j in range(n):
                    if j >= i - 1 and j < i + L:
                        continue
                    if j + 1 >= n:
                        continue
                    # Check if insertion position is valid (not adjacent to original)
                    # Simplified: just check improvement
                    ins_prev = tour[j]
                    ins_next = tour[j + 1]
                    ins_old = d[ins_prev, ins_next]
                    ins_new = d[ins_prev, seg[0]] + d[seg[-1], ins_next]
                    total_old = old_cost + ins_old
                    total_new = new_cost + ins_new
                    if total_new < total_old - 1e-10:
                        # Perform move
                        del tour[i:i + L]
                        # Find insertion point (shifted indices)
                        if j >= i:
                            j -= L
                        tour.insert(j + 1, seg[0])
                        if len(seg) > 1:
                            for k in range(1, len(seg)):
                                tour.insert(j + 1 + k, seg[k])
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return tour

def _greedy_insertion(d):
    n = d.shape[0]
    # Start with 3-city tour
    tour = [0, 1, 2]
    unv = list(range(3, n))
    while unv:
        best_cost = np.inf
        best_pos = -1
        best_city = -1
        for city in unv:
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos + 1) % len(tour)]
                cost = d[a, city] + d[city, b] - d[a, b]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
                    best_city = city
        tour.insert(best_pos + 1, best_city)
        unv.remove(best_city)
    return tour

def _ils(d, seed=42):
    rng = np.random.RandomState(seed)
    n = d.shape[0]
    best = _nn_tour(d)
    best = _two_opt(best, d)
    best = _or_opt(best, d)
    best_len = _tour_length(best, d)
    
    cur = best[:]
    cur_len = best_len
    
    t_start = time.time()
    budget = _DEADLINE - time.time() - 5.0  # leave 5s margin
    if budget <= 0:
        return best
    
    max_iters = int(budget / 0.005)  # rough estimate
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        # Perturbation: random 2-opt or or-opt move
        if rng.random() < 0.5:
            i, j = sorted(rng.randint(0, n, 2))
            if j - i > 1:
                cur[i:j + 1] = cur[i:j + 1][::-1]
        else:
            i = rng.randint(0, n)
            j = (i + 1 + rng.randint(1, 3)) % n
            if i != j:
                cur[i], cur[j] = cur[j], cur[i]
        
        cur = _two_opt(cur, d)
        cur_len = _tour_length(cur, d)
        
        if cur_len < best_len:
            best = cur[:]
            best_len = cur_len
    
    return best

def _compute_best_tour(d):
    global _TOUR, _DONE
    # Try greedy insertion first, then ILS
    t_start = time.time()
    tour = _greedy_insertion(d)
    tour = _two_opt(tour, d)
    tour = _or_opt(tour, d)
    best_tour = tour
    best_len = _tour_length(tour, d)
    
    # Run ILS if time permits
    if time.time() < _DEADLINE - 10.0:
        for seed in range(20):
            if time.time() > _DEADLINE - 5.0:
                break
            t = _ils(d, seed=seed)
            tl = _tour_length(t, d)
            if tl < best_len:
                best_tour = t
                best_len = tl
    
    _TOUR = best_tour
    _DONE = True
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _TOUR, _FINGERPRINT, _DONE, _START, _DEADLINE
    unvisited = np.asarray(unvisited_nodes)
    if unvisited.size == 0:
        return int(current_node)
    
    d = distance_matrix
    n = int(d.shape[0])
    
    # Check if we should compute the tour
    fp = _fingerprint(d)
    if _FINGERPRINT is None or fp != _FINGERPRINT:
        _FINGERPRINT = fp
        _TOUR = None
        _DONE = False
    
    if not _DONE and time.time() < _DEADLINE - 2.0:
        try:
            _compute_best_tour(d)
        except Exception:
            _DONE = True
    
    # Follow the precomputed tour
    if _TOUR is not None and len(_TOUR) == n:
        # Find current position in tour
        try:
            idx = _TOUR.index(int(current_node))
            # Find next unvisited city in tour order
            for k in range(1, n):
                nxt = _TOUR[(idx + k) % n]
                if nxt in set(unvisited.tolist()):
                    return int(nxt)
        except ValueError:
            pass
    
    # Fallback: nearest neighbor
    arr = unvisited
    g = d[current_node, arr]
    return int(arr[int(np.argmin(g))])
# EVOLVE-BLOCK-END
