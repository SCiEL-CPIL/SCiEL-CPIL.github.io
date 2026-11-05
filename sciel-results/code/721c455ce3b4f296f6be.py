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

def _paley_17_adj():
    """Create Paley graph of order 17."""
    p = 17
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _blow_up_paley(base_adj, base_n, cluster_size, rng):
    """
    Blow up each vertex in the base graph into a cluster of cluster_size vertices.
    Two vertices in different clusters are adjacent if their clusters are adjacent in base.
    Within a cluster, add some random edges to help break independent sets.
    """
    total_n = base_n * cluster_size
    adj = [0] * total_n
    
    # Define cluster membership: vertex v belongs to cluster v // cluster_size
    for i in range(total_n):
        for j in range(i + 1, total_n):
            ci = i // cluster_size
            cj = j // cluster_size
            
            if ci == cj:
                # Same cluster: add edge with some probability to break independence
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            else:
                # Different clusters: edge if clusters are adjacent in base
                if base_adj[ci] & (1 << cj):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    
    return adj

def _affine_graph_36():
    """
    Construct a graph on 36 vertices using AG(2,6) structure.
    Vertices are (x, y) with 0 <= x, y < 6.
    Two vertices are adjacent if their difference is in a chosen set D.
    We want K4-free and alpha <= 5.
    
    For K4-free: avoid having 4 mutually adjacent vertices.
    For alpha <= 5: avoid having 6 mutually non-adjacent vertices.
    
    One approach: use a (36, 15, 4) difference set if it exists, or approximate.
    Since 15*14 = 210 and 35*lambda = 210, lambda=6. So (36,15,6) difference set.
    But we don't have one readily available.
    
    Alternative: use a grid-based adjacency where (x1,y1) ~ (x2,y2) if
    (x1-x2)^2 + (y1-y2)^2 is a quadratic residue mod 6 or similar.
    """
    n = 36
    adj = [0] * n
    
    # Define adjacency based on differences
    # Two vertices (x1,y1) and (x2,y2) are adjacent if (dx, dy) is in S
    # where S is a set of differences.
    
    # For simplicity, use: adjacent if dx != 0 and dy != 0 and (dx*dy) % 6 != 0
    # This is a heuristic; we'll refine with local search.
    
    for x1 in range(6):
        for y1 in range(6):
            v1 = x1 * 6 + y1
            for x2 in range(x1 + 1):
                for y2 in range(6):
                    pass
            for x2 in range(6):
                for y2 in range(6):
                    if x1 == x2 and y1 == y2:
                        continue
                    v2 = x2 * 6 + y2
                    if v1 >= v2:
                        continue
                    dx = (x2 - x1) % 6
                    dy = (y2 - y1) % 6
                    # Adjacent if dx and dy are both non-zero and not both even
                    # This creates a checkerboard-like pattern
                    if dx != 0 and dy != 0:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
    
    return adj

def construct():
    # Start with Paley 17 as baseline
    base_adj = _paley_17_adj()
    best_edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if base_adj[i] & (1 << j):
                best_edges.append([i, j])
    best_n = 17
    
    rng = random.Random(12345)
    
    # Strategy 1: Blow up Paley 17 to get larger graphs
    # Try cluster sizes 2, 3, etc.
    for cluster_size in [2, 3]:
        if _elapsed() > _DEADLINE - 10:
            break
        
        n = 17 * cluster_size
        if n > 40:
            continue
        
        # Try multiple random initializations within clusters
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 5:
                break
            
            adj_init = _blow_up_paley(base_adj, 17, cluster_size, rng)
            time_slice = min(15.0, _DEADLINE - 5 - _elapsed())
            
            edges = _try_local_search(n, adj_init, time_slice, seed=attempt * 100)
            if edges is not None:
                best_n = n
                best_edges = edges
                break
    
    # Strategy 2: Try affine graph on 36 vertices
    if _elapsed() < _DEADLINE - 10 and best_n < 36:
        adj_init = _affine_graph_36()
        time_slice = min(20.0, _DEADLINE - 5 - _elapsed())
        edges = _try_local_search(36, adj_init, time_slice, seed=999)
        if edges is not None:
            best_n = 36
            best_edges = edges
    
    # Strategy 3: Try random graphs on various n values with local search
    for n in [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        if _elapsed() > _DEADLINE - 3:
            break
        if n <= best_n:
            break
        
        time_slice = min(5.0, _DEADLINE - 3 - _elapsed())
        if time_slice < 0.5:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if _elapsed() > _DEADLINE - 3:
                break
            p_edge = 0.3 + 0.05 * attempt
            adj_init = _random_adj(n, p_edge)
            edges = _try_local_search(n, adj_init, time_slice / 3, seed=n * 100 + attempt)
            if edges is not None:
                best_n = n
                best_edges = edges
                break
    
    return {'n': best_n, 'edges': best_edges}
