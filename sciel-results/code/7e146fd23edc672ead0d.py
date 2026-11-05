import time
import random

_START = time.time()
_DEADLINE = 140  # seconds, leave 10s margin for the 150s limit

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
        
        # Pick vertex that minimizes remaining candidates after inclusion
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
        
        # Branch 1: Include v
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        res = backtrack(candidates & ~(1 << v), current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _paley_graph(p):
    """Construct the Paley graph on p vertices (p must be 1 mod 4)."""
    if p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _paley_edges(p):
    """Return edge list for Paley graph on p vertices."""
    if p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return edges

def _try_local_search(n, adj_init, time_limit, seed=None):
    """Try to find a valid graph on n vertices using local search from initial adj."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    # Copy initial adjacency
    adj = list(adj_init)
    
    # Local search loop
    while time.time() - start < time_limit:
        # Check for K4
        k4 = _has_k4(adj, n)
        if k4:
            # Remove a random edge from the K4
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        # Check for Ind 6
        ind6 = _has_ind_6(adj, n)
        if ind6:
            # Add a random edge between two vertices in the independent set
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        # If no K4 and no Ind 6, we found a valid graph
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def _random_adj(n, p_edge):
    """Create a random adjacency list with edge probability p_edge."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _adj_to_edges(adj, n):
    """Convert adjacency list to edge list."""
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
    
    # Try to find larger graphs from n=18 upward
    # For each n, try multiple strategies with time slices
    
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time allocation: more time for smaller n, less for larger
        if n <= 22:
            time_slice = 4.0
        elif n <= 28:
            time_slice = 3.0
        elif n <= 32:
            time_slice = 2.0
        else:
            time_slice = 1.0
        
        if _elapsed() + time_slice > _DEADLINE - 3:
            time_slice = _DEADLINE - 3 - _elapsed()
            if time_slice < 0.5:
                break
        
        found = False
        
        # Strategy 1: Local search from random initialization with tuned density
        # For K4-free + no independent 6, density around 0.3-0.5 works well
        t_remaining = time_slice
        for attempt in range(3):
            if t_remaining < 0.3:
                break
            alloc = t_remaining / (3 - attempt)
            adj_init = _random_adj(n, 0.35 + 0.05 * attempt)
            edges = _try_local_search(n, adj_init, alloc)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
            t_remaining -= alloc
        
        if found:
            continue
        
        # Strategy 2: Extend from Paley 17 by adding vertices
        if _elapsed() + 1.0 < _DEADLINE - 3:
            paley_adj = _paley_graph(17)
            if paley_adj is not None:
                # Extend to n vertices
                adj_ext = [0] * n
                for i in range(17):
                    adj_ext[i] = paley_adj[i]
                
                # Add new vertices with random connections
                for i in range(17, n):
                    for j in range(i):
                        if random.random() < 0.4:
                            adj_ext[i] |= (1 << j)
                            adj_ext[j] |= (1 << i)
                
                edges = _try_local_search(n, adj_ext, min(2.0, _DEADLINE - 3 - _elapsed()))
                if edges is not None:
                    best_n = n
                    best_edges = edges
                    continue
        
        # Strategy 3: Try with different random seeds
        if _elapsed() + 0.5 < _DEADLINE - 3:
            for seed_val in [42, 123, 999]:
                if _elapsed() > _DEADLINE - 3:
                    break
                adj_init = _random_adj(n, 0.4)
                edges = _try_local_search(n, adj_init, min(1.0, _DEADLINE - 3 - _elapsed()), seed=seed_val)
                if edges is not None:
                    best_n = n
                    best_edges = edges
                    break
    
    return {'n': best_n, 'edges': best_edges}
