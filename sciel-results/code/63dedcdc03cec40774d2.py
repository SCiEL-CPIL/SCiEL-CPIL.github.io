"""Construct a dense triangle-free 4-chromatic graph on 16 vertices."""
import time
import random

# EVOLVE-BLOCK-START

def _hamming_weight(x):
    """Return the number of 1-bits in x."""
    w = 0
    while x:
        x &= x - 1
        w += 1
    return w

def _clebsch_edges():
    """
    Return edges of the Clebsch graph (folded 5-cube) on 16 vertices 0..15.
    Vertices are 5-bit strings with MSB=0, i.e. integers 0..15.
    Two vertices are adjacent iff Hamming distance is 1 or 4.
    This graph is triangle-free, 4-chromatic, and has 40 edges.
    """
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            d = _hamming_weight(i ^ j)
            if d == 1 or d == 4:
                edges.append([i, j])
    return edges

def _build_adjacency(edges, n=16):
    """Build a bitset adjacency matrix from an edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free(adj, n=16):
    """Check if graph with bitset adjacency is triangle-free."""
    for u in range(n):
        a = adj[u]
        for v in range(n):
            if not (a >> v) & 1 or v <= u:
                continue
            if adj[u] & adj[v]:
                return False
    return True

def _is_3_colorable(adj, n=16):
    """
    Check 3-colorability via backtracking with MRV heuristic.
    Returns True if the graph is 3-colorable, False otherwise.
    """
    color = [-1] * n
    neighbors = [[] for _ in range(n)]
    for u in range(n):
        a = adj[u]
        while a:
            b = a & (-a)
            v = b.bit_length() - 1
            a ^= b
            if v > u:
                neighbors[u].append(v)
                neighbors[v].append(u)
    
    def backtrack(uncolored, domains):
        if not uncolored:
            return True
        
        best_v = -1
        best_size = 4
        for v in uncolored:
            ds = domains[v]
            s = len(ds)
            if s == 0:
                return False
            if s < best_size:
                best_size = s
                best_v = v
                if s == 1:
                    break
        
        if best_v == -1:
            return True
        
        for c in sorted(domains[best_v]):
            conflict = False
            for nb in neighbors[best_v]:
                if color[nb] == c:
                    conflict = True
                    break
            if conflict:
                continue
            
            color[best_v] = c
            new_domains = dict(domains)
            new_uncolored = set(uncolored)
            new_uncolored.remove(best_v)
            feasible = True
            for nb in neighbors[best_v]:
                if nb in new_uncolored:
                    new_domains[nb] = new_domains[nb] - {c}
                    if not new_domains[nb]:
                        feasible = False
                        break
            
            if feasible:
                if backtrack(frozenset(new_uncolored), new_domains):
                    return True
            
            color[best_v] = -1
        
        return False
    
    initial_domains = {i: {0, 1, 2} for i in range(n)}
    return backtrack(frozenset(range(n)), initial_domains)

def _add_edge_safely(adj, edges, u, v, n=16):
    """
    Try to add edge (u,v) while maintaining triangle-freeness.
    Returns True if edge was added, False otherwise.
    """
    if (adj[u] >> v) & 1:
        return False
    if adj[u] & adj[v]:
        return False
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    edges.append([u, v])
    return True

def _remove_edge(adj, edges, u, v, n=16):
    """Remove edge (u,v) from adjacency and edge list."""
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    for i, e in enumerate(edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            edges.pop(i)
            break

def _simulated_annealing_search(adj, edges, deadline, rng, n=16):
    """
    Simulated annealing: starts from Clebsch graph, accepts edge additions/removals
    that preserve triangle-freeness. Uses a temperature schedule to escape local optima.
    Objective: maximize edges while maintaining 4-chromaticity.
    """
    best_edge_count = len(edges)
    best_edges_snapshot = list(edges)
    best_adj_snapshot = list(adj)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Temperature schedule
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Pick a random move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Try adding a random non-edge
            for _ in range(100):
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                # Check if we can add
                if adj[u] & adj[v]:
                    continue
                # Add temporarily
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                
                new_count = len(edges)
                is_4ch = not _is_3_colorable(adj, n)
                
                if is_4ch:
                    # Accept if better
                    if new_count > best_edge_count:
                        best_edge_count = new_count
                        best_edges_snapshot = list(edges)
                        best_adj_snapshot = list(adj)
                else:
                    # Might reject (3-colorable)
                    # Compute "energy": we prefer 4-chromatic with many edges
                    # Penalty for being 3-colorable
                    old_count = new_count - 1
                    # Accept with probability based on temperature
                    delta = -(50 if old_count >= best_edge_count else 20)  # penalty
                    if rng.random() > min(1.0, exp_neg(delta, T)):
                        # Reject: remove the edge
                        _remove_edge(adj, edges, u, v, n)
                
                break
        elif move_type < 0.9:
            # Try removing a random edge
            if edges:
                idx = rng.randrange(len(edges))
                u, v = edges[idx]
                _remove_edge(adj, edges, u, v, n)
                # Accept with some probability to explore
                if rng.random() > 0.3:
                    # Remove more aggressively
                    pass
        else:
            # Swap: remove one edge, add another
            if edges:
                idx = rng.randrange(len(edges))
                u, v = edges[idx]
                _remove_edge(adj, edges, u, v, n)
                # Try to add a different edge
                for _ in range(50):
                    u2 = rng.randrange(n)
                    v2 = rng.randrange(n)
                    if u2 == v2:
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                    if adj[u2] & adj[v2]:
                        continue
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    edges.append([u2, v2])
                    
                    new_count = len(edges)
                    is_4ch = not _is_3_colorable(adj, n)
                    
                    if is_4ch and new_count > best_edge_count:
                        best_edge_count = new_count
                        best_edges_snapshot = list(edges)
                        best_adj_snapshot = list(adj)
                    break
        
        T *= cooling
    
    return best_edges_snapshot

def exp_neg(x, T):
    """Compute exp(-x/T) safely."""
    import math
    if T <= 0:
        return 0.0
    val = -x / T
    if val < -700:
        return 0.0
    return math.exp(val)

def construct():
    """
    Construct a triangle-free graph on vertices 0..15 that is not 3-colourable.
    Maximize the number of edges. Uses the Clebsch graph (40 edges) as baseline
    and runs simulated annealing to attempt to find a denser 4-chromatic
    triangle-free graph.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with Clebsch graph
    edges = _clebsch_edges()
    adj = _build_adjacency(edges)
    
    # Verify it's valid
    assert _is_triangle_free(adj), "Clebsch graph should be triangle-free"
    assert not _is_3_colorable(adj), "Clebsch graph should not be 3-colorable"
    
    base_count = len(edges)
    
    if time.time() > deadline - 5:
        return edges
    
    # Run simulated annealing
    rng = random.Random(12345)
    best_edges = _simulated_annealing_search(adj, edges, deadline, rng)
    
    return best_edges

# EVOLVE-BLOCK-END
