import time
import random

_START = time.time()
_DEADLINE = 145  # seconds, leave 5s margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns the set if found, else None."""
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v]
            cnt = new_cand.bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
            
        v = best_v
        new_candidates = candidates & ~((1 << v) | adj[v])
        
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        res = backtrack(candidates & ~(1 << v), current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _greedy_insert(n, adj_base, time_limit, seed=None):
    """
    Greedy vertex insertion: start from a valid base graph, try to add vertices
    one at a time, choosing edges to avoid K4 and independent 6-sets.
    """
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    adj = list(adj_base)
    current_n = len(adj)
    
    while current_n < n:
        if time.time() - start > time_limit:
            break
        
        new_v = current_n
        adj.append(0)
        
        # Try to find a valid connection pattern for new_v
        found_valid = False
        for trial in range(200):
            if time.time() - start > time_limit:
                break
            
            # Randomly choose which old vertices to connect to
            conn = 0
            for i in range(current_n):
                if random.random() < 0.4:
                    conn |= (1 << i)
            
            # Apply
            adj[new_v] = conn
            for i in range(current_n):
                if conn & (1 << i):
                    adj[i] |= (1 << new_v)
            
            # Check validity
            if _has_k4(adj, current_n + 1) is None and _has_ind_6(adj, current_n + 1) is None:
                found_valid = True
                break
            else:
                # Revert
                adj[new_v] = 0
                for i in range(current_n):
                    adj[i] &= ~(1 << new_v)
        
        if not found_valid:
            # Remove the new vertex
            adj.pop()
            break
        
        current_n += 1
    
    if current_n == n:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    return None

def _try_local_search(n, adj_init, time_limit, seed=None):
    """Try to find a valid graph on n vertices using local search from initial adj."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    adj = list(adj_init)
    
    while time.time() - start < time_limit:
        k4 = _has_k4(adj, n)
        if k4:
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        ind6 = _has_ind_6(adj, n)
        if ind6:
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def _random_adj(n, p_edge):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    # Start with Paley 17 as baseline
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Build up from Paley 17 using greedy insertion
    paley_adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                paley_adj[i] |= (1 << j)
                paley_adj[j] |= (1 << i)
    
    # Try greedy insertion to extend beyond 17
    for target_n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        time_slice = min(3.0, _DEADLINE - 3 - _elapsed())
        if time_slice < 0.5:
            break
        
        # Try greedy insertion from Paley 17
        for attempt in range(3):
            if _elapsed() > _DEADLINE - 3:
                break
            alloc = time_slice / (3 - attempt)
            edges = _greedy_insert(target_n, paley_adj, alloc, seed=attempt * 100 + 7)
            if edges is not None:
                best_n = target_n
                best_edges = edges
                break
        
        if best_n >= target_n:
            continue
        
        # Fallback: local search
        if _elapsed() + 1.0 < _DEADLINE - 3:
            adj_init = _random_adj(target_n, 0.35)
            edges = _try_local_search(target_n, adj_init, min(1.5, _DEADLINE - 3 - _elapsed()))
            if edges is not None:
                best_n = target_n
                best_edges = edges
                continue
    
    return {'n': best_n, 'edges': best_edges}

