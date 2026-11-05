import numpy as np
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # 145s out of 150s budget
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try polarity graph seed first
        result = _local_search(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found > best_n:
                best_n = n_found
                best_edges = edges_found
            return {'n': n_found, 'edges': edges_found}
        
        # Try greedy construction
        for strategy in ['low_degree_first', 'random_order']:
            if time.time() > deadline:
                break
            
            result = _try_construct(n, required_min_deg, strategy, deadline)
            if result is not None:
                n_found, edges_found = result
                if n_found > best_n:
                    best_n = n_found
                    best_edges = edges_found
                return {'n': n_found, 'edges': edges_found}
    
    return {'n': best_n, 'edges': best_edges}


def _check_c4_fast(adj, u, v):
    """
    Fast check: does adding edge (u,v) create a C4?
    A C4 is created iff there exists a path u-x-y-v of length 3 (x != v, y != u, x != y).
    Equivalently: N(u) \\ {v} and N(v) \\ {u} share a common neighbor in the 2-hop sense.
    More precisely: exists x in N(u), y in N(v) such that x != v, y != u, x != y, and adj[x][y].
    Vectorized: let A = adj[u], B = adj[v]. We need (A & ~B) to have any neighbor in (B & ~A).
    Actually simpler: for each x in N(u)\\{v}, check if any neighbor y of x is in N(v)\\{u} and y != x.
    """
    # Get neighbors of u and v
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]
    
    # Exclude v from nu and u from nv
    mask_u = nu != v
    mask_v = nv != u
    nu_f = nu[mask_u]
    nv_f = nv[mask_v]
    
    # For each x in nu_f, check if adj[x] intersects nv_f (excluding x itself if needed)
    # adj[x] & nv_f should be non-empty, but we need y != x
    # Since x is in nu_f (neighbor of u), and we're checking if x is also neighbor of v... 
    # Actually x could be in nv_f too. If x is in both, then x is a common neighbor.
    # But we need a path u-x-y-v where x != y. So if x is in nv_f, we still need another y.
    
    for x in nu_f:
        # Check neighbors of x that are in nv_f and not equal to x
        neighbors_x = np.where(adj[x])[0]
        # We need y in neighbors_x such that y in nv_f and y != x
        # Use set intersection for speed
        if len(nv_f) == 0:
            continue
        # Check if any neighbor of x is in nv_f and != x
        for y in neighbors_x:
            if y == x:
                continue
            # Check if y is in nv_f
            # This is O(deg(x)) per x, total O(deg(u)*avg_deg)
            if y in set(nv_f):
                return True
    return False


def _check_c4_vectorized(adj, u, v):
    """
    Fully vectorized C4 check.
    Adding (u,v) creates C4 iff there exist x,y with:
    adj[u,x] & adj[x,y] & adj[y,v] & (x != v) & (y != u) & (x != y)
    
    Let's compute: for all x,y: adj[u,x] * adj[x,y] * adj[y,v]
    Then exclude x==v, y==u, x==y cases.
    """
    # A[x,y] = adj[u,x] * adj[x,y] * adj[y,v]
    # = (adj[u] * adj[x,y] * adj[v]) summed over y... no.
    # For each x: sum over y of adj[u,x] * adj[x,y] * adj[y,v]
    # = adj[u,x] * (adj[x] * adj[v]).sum()
    
    # Compute M[x] = (adj[x] & adj[v]).sum() for all x
    # Then C4 exists if any x with adj[u,x] & M[x] > 0, excluding x==v
    # But also need to handle x==y case: if x is common neighbor of u and v, 
    # then path u-x-x-v is not valid (x==y). 
    # Actually if x is common neighbor, adj[u,x] & adj[x,x] & adj[x,v] - but adj[x,x]=0.
    # So x==y case is automatically excluded since no self-loops.
    
    # Wait: path u-x-y-v with x==y would be u-x-x-v, which requires adj[x,x]=1, impossible.
    # So we just need: exists x != v such that adj[u,x] and (adj[x] & adj[v]).sum() > 0
    # But we also need y != u. If y == u, then adj[x,u] & adj[u,v]... 
    # Path u-x-u-v: requires adj[x,u] and adj[u,v]. But adj[u,v] is 0 (we're checking if we CAN add it).
    # So y==u case: adj[x,y]=adj[x,u] and adj[y,v]=adj[u,v]=0. So automatically excluded.
    
    # So the check simplifies to: exists x != v such that adj[u,x] and (adj[x] & adj[v]).sum() > 0
    
    # Compute common = adj[u] & ~adj[v] (exclude v from consideration... actually just compute)
    # For each x where adj[u,x] is True and x != v:
    #   check if (adj[x] & adj[v]).sum() > 0
    
    # Vectorized:
    # Let B = adj[v] (bool array)
    # For each x: count_x = (adj[x] & B).sum()
    # We need: exists x with adj[u,x] & (x != v) & (count_x > 0)
    
    # count for all x at once:
    # counts = (adj & adj[v]).sum(axis=1)  # this is (adj * adj[v][:,None]).sum(axis=1)
    # Actually: adj[x] & adj[v] is element-wise, sum gives count of common neighbors of x and v
    
    common_with_v = (adj & adj[v]).sum(axis=1)  # for each x, # of common neighbors of x and v
    
    # Now check: is there an x with adj[u,x] & (x != v) & (common_with_v[x] > 0)?
    # Note: if x == v, adj[u,v] might be 0 (we're about to add it), so it's fine.
    # But to be safe:
    mask = adj[u].copy()
    mask[v] = False  # exclude x == v
    # Also need common_with_v[x] > 0
    return np.any(mask & (common_with_v > 0))


def _try_construct(n, required_min_deg, strategy, deadline):
    """Greedy edge addition with fast C4 check."""
    adj = np.zeros((n, n), dtype=bool)
    
    # Build priority queue of edges to try
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    if strategy == 'random_order':
        random.shuffle(all_pairs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        degrees = adj.sum(axis=1)
        
        if degrees.min() >= required_min_deg:
            break
        
        # Sort by degree sum (lower degree vertices first)
        if strategy == 'low_degree_first':
            all_pairs.sort(key=lambda p: degrees[p[0]] + degrees[p[1]])
        
        for (u, v) in all_pairs:
            if time.time() > deadline:
                break
            if adj[u, v]:
                continue
            if not _check_c4_vectorized(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                improved = True
                break  # restart to re-prioritize
    
    degrees = adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    return None


def _polarity_seed(n):
    """
    Create a polarity graph seed. For q = n, if n is a prime power,
    the polarity graph of PG(2,q) has q+1 vertices and is C4-free.
    For general n, we use a greedy construction biased toward high degree.
    """
    adj = np.zeros((n, n), dtype=bool)
    rng = np.random.RandomState(42)
    
    # Try to build a dense C4-free graph
    # Start with empty, add edges greedily prioritizing low-degree vertices
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(pairs)
    
    for (u, v) in pairs:
        if adj[u, v]:
            continue
        if not _check_c4_vectorized(adj, u, v):
            adj[u, v] = True
            adj[v, u] = True
    
    return adj


def _local_search(n, required_min_deg, deadline):
    """
    Local search: start with a seed, then try edge additions/removals to maximize min degree.
    """
    # Start with polarity/greedy seed
    adj = _polarity_seed(n)
    degrees = adj.sum(axis=1)
    
    # If already good enough, return
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    
    # Local search: try to add edges to boost min degree
    # Focus on vertices with lowest degree
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        if degrees.min() >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return (n, edges)
        
        # Find vertex with minimum degree
        min_deg = degrees.min()
        low_verts = np.where(degrees == min_deg)[0]
        
        # For each low-degree vertex, try to add an edge
        improved = False
        for u in low_verts:
            if time.time() > deadline:
                break
            
            # Try all non-neighbors of u, sorted by their degree (prefer connecting to low-degree)
            non_neighbors = np.where(~adj[u])[0]
            non_neighbors = non_neighbors[non_neighbors != u]
            
            # Sort by degree (prefer connecting to low-degree vertices to balance)
            non_neighbors = sorted(non_neighbors, key=lambda x: degrees[x])
            
            for v in non_neighbors:
                if v <= u:
                    continue
                if adj[u, v]:
                    continue
                if not _check_c4_vectorized(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try edge swaps: remove an edge from a high-degree vertex, add to low-degree
            # This is more complex; for now, try random restarts
            break
    
    # Final check
    degrees = adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    
    return None

# EVOLVE-BLOCK-END
