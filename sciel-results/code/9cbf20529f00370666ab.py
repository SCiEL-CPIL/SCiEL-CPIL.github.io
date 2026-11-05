import time
import random

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 150s kill

def _c6_check(u, v, adj):
    """Check if adding edge (u,v) creates a C6.
    A C6 through (u,v) means a path u-a-b-c-d-v of length 5 in G\\{u,v}.
    Equivalently: is there a path of length 4 from u to v in G - {u,v}?
    """
    # BFS/DFS from u to v with path length exactly 4
    # depth 0: u
    # depth 1: neighbors of u (not v)
    # depth 2: neighbors of depth1 (not u,v, not visited)
    # depth 3: neighbors of depth2
    # depth 4: if v is reachable -> C6 exists
    
    # Use iterative DFS with bitmasks for speed
    n = 40
    # Start: neighbors of u, excluding v
    visited = (1 << u) | (1 << v)
    frontier = [(a, 1) for a in range(n) if (adj[u] >> a) & 1 and a != v]
    
    if not frontier:
        return False
    
    # We track (node, depth, visited_mask)
    # To keep it fast, use a stack
    stack = []
    for a in frontier:
        stack.append((a, 1, visited | (1 << a)))
    
    while stack:
        node, depth, mask = stack.pop()
        
        if depth == 4:
            if (adj[node] >> v) & 1:
                return True
            continue
        
        # Explore neighbors not in mask
        nbrs = adj[node] & ~mask
        while nbrs:
            bit = nbrs & -nbrs
            nb = bit.bit_length() - 1
            nbrs ^= bit
            stack.append((nb, depth + 1, mask | (1 << nb)))
    
    return False

def _try_add(u, v, adj, edges):
    """Try to add edge (u,v). Returns True if added."""
    if (u, v) in edges or (v, u) in edges:
        return False
    if _c6_check(u, v, adj):
        return False
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    edges.add((min(u,v), max(u,v)))
    return True

def _remove(u, v, adj, edges):
    """Remove edge (u,v)."""
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    edges.discard((min(u,v), max(u,v)))

def _local_search(adj, edges, rng, time_limit):
    """Iteratively try to add edges, then do swaps."""
    n = 40
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Phase 1: Try to add any non-edge
        non_edges = [(u, v) for u, v in all_pairs if (u, v) not in edges]
        rng.shuffle(non_edges)
        
        added = False
        for u, v in non_edges:
            if _try_add(u, v, adj, edges):
                added = True
                improved = True
                break
        
        if added:
            continue
        
        # Phase 2: Try edge swaps (remove one, add one)
        edge_list = list(edges)
        rng.shuffle(edge_list)
        
        swapped = False
        for eu, ev in edge_list:
            # Save state
            _remove(eu, ev, adj, edges)
            
            # Try to add a non-edge
            non_edges_curr = [(u, v) for u, v in all_pairs if (u, v) not in edges]
            rng.shuffle(non_edges_curr)
            
            added_back = False
            for u, v in non_edges_curr:
                if _try_add(u, v, adj, edges):
                    added_back = True
                    swapped = True
                    improved = True
                    break
            
            if not added_back:
                # Revert
                adj[eu] |= (1 << ev)
                adj[ev] |= (1 << eu)
                edges.add((min(eu,ev), max(eu,ev)))
            
            if swapped:
                break
    
    return edges

def _seed_graph():
    """Create the 8 disjoint K5 graph."""
    n = 40
    adj = [0] * n
    edges = set()
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u, v = 5*k + a, 5*k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
    return adj, edges

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    best_edges = set()
    best_count = 0
    
    # Start from seed
    adj, edges = _seed_graph()
    best_count = len(edges)
    best_edges = set(edges)
    
    rng = random.Random(42)
    
    # Try multiple restarts with different randomizations
    for seed in range(50):
        if time.time() > _DEADLINE - 5:
            break
        
        # Work on a copy
        adj_copy = adj[:]
        edges_copy = set(edges)
        
        # Randomize edge order for greedy phase
        all_pairs = [(i, j) for i in range(40) for j in range(i+1, 40)]
        rng.shuffle(all_pairs)
        
        # Try adding edges in random order
        for u, v in all_pairs:
            if (u, v) not in edges_copy:
                if not _c6_check(u, v, adj_copy):
                    adj_copy[u] |= (1 << v)
                    adj_copy[v] |= (1 << u)
                    edges_copy.add((u, v))
        
        # Local search
        time_limit = min(time.time() + 20, _DEADLINE - 3)
        edges_copy = _local_search(adj_copy, edges_copy, rng, time_limit)
        
        if len(edges_copy) > best_count:
            best_count = len(edges_copy)
            best_edges = set(edges_copy)
            if best_count >= 110:
                break
    
    # Return as list of [u, v]
    return [[u, v] for u, v in sorted(best_edges)]
