import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 148  # 148s out of 150s budget

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break

        required_min_deg = n - 39

        # Try many random restarts with bitmask-based greedy construction
        for seed in range(5000):
            if time.time() > deadline:
                break
            result = _greedy_bitmask(n, required_min_deg, seed)
            if result is not None:
                return {'n': n, 'edges': result}

        # If greedy didn't work, try local search with bitmask
        result = _local_search_bitmask(n, required_min_deg, deadline)
        if result is not None:
            return {'n': n, 'edges': result}

    # Fallback: n=39, empty graph (min degree 0 >= 39-39=0)
    return {'n': 39, 'edges': []}


def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u,v) to graph represented by adj (list of int bitmasks)
    would create a C4. Returns True if safe.
    
    C4 exists iff there is a path u-x-y-v of length 3 (x!=v, y!=u, x!=y).
    Equivalent: exists x in N(u)\{v} such that N(x) ∩ N(v)\{u} contains a vertex != x.
    """
    nu = adj[u] & ~(1 << v)  # N(u) minus v
    nv = adj[v] & ~(1 << u)  # N(v) minus u
    
    # For each x in nu, check if (adj[x] & nv) has any bit set other than x
    # Iterate through bits of nu
    temp = nu
    while temp:
        # Extract lowest set bit
        lsb = temp & (-temp)
        x = lsb.bit_length() - 1
        # Check if adj[x] & nv has any bit other than x
        if (adj[x] & nv) & ~(1 << x):
            return False
        temp &= temp - 1  # clear lowest set bit
    
    return True


def _greedy_bitmask(n, required_min_deg, seed):
    """
    Greedy edge addition using bitmask-based C4 checking.
    Returns list of edges if min degree requirement is met, else None.
    """
    rng = random.Random(seed)
    adj = [0] * n
    
    # Generate edge order with some bias toward balancing degrees
    # Use a priority-based approach: repeatedly pick the most beneficial edge
    
    # First pass: add edges in random order
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_pairs)
    
    for (u, v) in all_pairs:
        if _can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Check min degree
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    return None


def _local_search_bitmask(n, required_min_deg, deadline):
    """
    Local search using bitmask representation.
    Start from multiple greedy seeds, then try to improve min degree.
    """
    best_adj = None
    best_min_deg = -1
    
    # Try multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        rng = random.Random(seed)
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        
        for (u, v) in all_pairs:
            if _can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        min_deg = min(degrees)
        
        if min_deg > best_min_deg:
            best_min_deg = min_deg
            best_adj = adj[:]
        
        if best_min_deg >= required_min_deg:
            break
    
    if best_adj is None:
        return None
    
    # Local search: try to add more edges to boost min degree
    adj = best_adj[:]
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        if min(degrees) >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return edges
        
        # Find vertex with minimum degree
        min_deg = min(degrees)
        low_verts = [i for i in range(n) if degrees[i] == min_deg]
        
        improved = False
        for u in low_verts:
            # Try to add an edge from u to a non-neighbor
            non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
            # Sort by degree (prefer connecting to low-degree vertices)
            non_neighbors.sort(key=lambda x: degrees[x])
            
            for v in non_neighbors:
                if v <= u:
                    continue
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Final check
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    return None

# EVOLVE-BLOCK-END
