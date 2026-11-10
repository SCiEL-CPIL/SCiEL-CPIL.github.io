import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 300.0 * 0.95 # Safety margin

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _tour_length(d, tour):
    """Calculate the length of a tour given as a list of node indices."""
    n = len(tour)
    if n == 0:
        return 0
    # Use numpy for speed if possible, but for n=50, simple sum is fine
    # d[tour, np.roll(tour, -1)] works if tour is array
    if isinstance(tour, np.ndarray):
        return float(np.sum(d[tour, np.roll(tour, -1)]))
    else:
        return float(sum(d[tour[i], tour[(i + 1) % n]] for i in range(n)))

def _two_opt(d, tour):
    """
    Improve a tour using 2-opt.
    tour: list of node indices
    """
    n = len(tour)
    if n < 4:
        return tour, _tour_length(d, tour)
    
    tour = list(tour)
    best_len = _tour_length(d, tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Edges (i-1, i) and (j, j+1) removed
                # Edges (i-1, j) and (i, j+1) added
                a = tour[i - 1]
                b = tour[i]
                c = tour[j]
                e = tour[(j + 1) % n]
                
                # Check if reversing segment [i..j] improves
                # Current: d[a,b] + d[c,e]
                # New: d[a,c] + d[b,e]
                delta = d[a, c] + d[b, e] - d[a, b] - d[c, e]
                
                if delta < -1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    best_len += delta
                    improved = True
                    
    return tour, best_len

def _or_opt(d, tour):
    """
    Improve a tour using Or-opt (moving 1 node).
    """
    n = len(tour)
    if n < 4:
        return tour, _tour_length(d, tour)
    
    tour = list(tour)
    best_len = _tour_length(d, tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            u = tour[i]
            prev_node = tour[(i - 1) % n]
            next_node = tour[(i + 1) % n]
            
            # Remove u
            # Cost to remove: -d[prev, u] - d[u, next] + d[prev, next]
            # Note: if n==2, prev==next, but n>=4 here
            
            len_without_u = best_len - d[prev_node, u] - d[u, next_node] + d[prev_node, next_node]
            
            # Try inserting u at all other positions
            # We need to be careful with indices when removing and inserting
            # A simpler way: create a temp list without u, then try all insertions
            
            temp_tour = tour[:]
            temp_tour.pop(i)
            m = len(temp_tour)
            
            best_insert_idx = -1
            best_insert_len = np.inf
            
            for k in range(m):
                # Insert u between temp_tour[k-1] and temp_tour[k]
                # In the original tour context, we are inserting u into the gap
                # Let's just calculate the delta for insertion
                
                # The gap is between temp_tour[(k-1)%m] and temp_tour[k]
                # But wait, the "gap" in the remaining tour corresponds to a specific edge.
                # Let's just compute the full length of the new tour to be safe, 
                # or calculate delta carefully.
                
                # Since n is small (50), computing full length of temp_tour + u
                # is O(N). Doing this for N insertions is O(N^2).
                # Total Or-opt is O(N^3). For N=50, 125k ops. Very fast.
                
                new_tour = temp_tour[:]
                new_tour.insert(k, u)
                
                # If k == m, we insert at end, which is same as index 0 for closed tour?
                # No, insert(k) adds at index k.
                # If k=0, u is first.
                # If k=m, u is last.
                # Since it's a cycle, inserting at m is same as inserting at 0?
                # Not exactly, the neighbors change.
                # Let's just use the length function.
                
                l = _tour_length(d, new_tour)
                
                if l < best_insert_len:
                    best_insert_len = l
                    best_insert_idx = k
            
            if best_insert_idx >= 0 and best_insert_len < best_len - 1e-10:
                # Reconstruct tour
                # We found the best insertion in temp_tour
                # We need to map this back to the original tour structure?
                # Actually, we can just set tour to the new_tour that gave best_insert_len
                # But we need to identify which new_tour it was.
                
                # Let's redo the logic to keep the best tour object
                best_new_tour = None
                best_new_len = np.inf
                for k in range(m):
                    new_tour = temp_tour[:]
                    new_tour.insert(k, u)
                    l = _tour_length(d, new_tour)
                    if l < best_new_len:
                        best_new_len = l
                        best_new_tour = new_tour
                
                if best_new_tour is not None and best_new_len < best_len - 1e-10:
                    tour = best_new_tour
                    best_len = best_new_len
                    improved = True

    return tour, best_len

def _nearest_neighbor_tour(d, start):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    cur = start
    
    for _ in range(n - 1):
        dist = d[cur].copy()
        dist[visited] = np.inf
        nxt = int(np.argmin(dist))
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
        
    return tour

def _find_best_tour(d):
    n = d.shape[0]
    if n <= 2:
        return list(range(n))
        
    best_tour = None
    best_len = np.inf
    
    # Multi-start
    # We can start from each node, or random starts
    # With N=50, starting from all 50 nodes is cheap.
    
    # First pass: NN from all starts
    for s in range(n):
        if _time_left() < 5.0:
            break
            
        tour = _nearest_neighbor_tour(d, s)
        tour, l = _two_opt(d, tour)
        tour, l = _or_opt(d, tour)
        # Polish again
        tour, l = _two_opt(d, tour)
        
        if l < best_len:
            best_len = l
            best_tour = tour[:]
            
    # Second pass: Random restarts if time permits
    # Use the best tour found so far as a template for perturbations?
    # Or just random tours.
    
    # Let's do random restarts
    rng = np.random.default_rng(42)
    while _time_left() > 10.0:
        # Random tour
        tour = list(range(n))
        rng.shuffle(tour)
        
        # Or start from best_tour and perturb
        # Perturbation: reverse a segment
        if best_tour is not None and rng.random() < 0.5:
            tour = best_tour[:]
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i > j:
                i, j = j, i
            tour[i:j+1] = tour[i:j+1][::-1]
        else:
            # NN from random start
            s = int(rng.integers(0, n))
            tour = _nearest_neighbor_tour(d, s)
            
        tour, l = _two_opt(d, tour)
        tour, l = _or_opt(d, tour)
        tour, l = _two_opt(d, tour)
        
        if l < best_len:
            best_len = l
            best_tour = tour[:]
            
    if best_tour is None:
        best_tour = _nearest_neighbor_tour(d, 0)
        
    return best_tour

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    if n <= 1:
        return d.copy()
    
    # Base heuristic: 1 / distance^3
    d_safe = np.where(d > 1e-10, d, 1e10)
    H = 1.0 / (d_safe ** 3)
    np.fill_diagonal(H, 1e-9)
    
    # Find best tour
    if n > 2 and _time_left() > 15.0:
        best_tour = _find_best_tour(d)
        
        # Boost tour edges
        # Use a very large value to ensure ACO follows this tour initially
        # 1e6 is large enough to dominate 1/d^3 (max ~1)
        boost_val = 1e6
        
        for i in range(n):
            u = best_tour[i]
            v = best_tour[(i + 1) % n]
            H[u, v] = boost_val
            H[v, u] = boost_val
            
    # Sanitize
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
