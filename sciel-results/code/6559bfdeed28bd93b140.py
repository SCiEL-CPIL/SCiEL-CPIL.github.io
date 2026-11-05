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
    deadline = time.time() + 140  # 140s out of 150s budget
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies and seeds
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            result = _try_construct_fast(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # If not found with fast method, try local search
        result = _local_search_fast(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
    
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}


def _check_c4_fast(adj, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    A C4 is created if there exists a path of length 3 from u to v not using (u,v).
    That is, exists x,y such that u-x, x-y, y-v are edges, with x != v, y != u, x != y.
    """
    # For each common neighbor... no, we need paths of length 3.
    # u - x - y - v
    # x is neighbor of u (x != v)
    # y is neighbor of x (y != u, y != v)
    # v is neighbor of y
    
    # Efficient check: 
    # For each x in N(u) \ {v}, for each y in N(x) \ {u, v}, check if y in N(v)
    
    # Even better: count common neighbors of (u, v) after considering paths.
    # Actually, the number of C4s created by adding (u,v) equals the number of
    # paths of length 3 from u to v in G.
    
    # A path u-x-y-v exists iff there is x in N(u)\{v} and y in N(x)\{u,v} with y in N(v).
    
    # Let's compute: for each x in N(u) \ {v}, let S_x = N(x) \ {u, v}.
    # If S_x intersects N(v), then C4 exists.
    
    # More efficient: 
    # Let A = N(u) \ {v}
    # Let B = N(v) \ {u}
    # A C4 exists iff there is x in A, y in B, such that x-y is an edge and x != y.
    # Wait, that's not quite right. The path is u-x-y-v. So x in N(u), y in N(v), and x-y is an edge.
    # And x != v, y != u, x != y.
    
    # So: for x in N(u) \ {v}, for y in N(v) \ {u}, if x != y and adj[x,y], then C4.
    
    # This is O(deg(u) * deg(v)) which could be large.
    
    # Better: For each x in N(u) \ {v}, check if N(x) \ {u} intersects N(v) \ {u, x}.
    # i.e., is there y in N(x) \ {u, v} such that y in N(v)?
    
    # Let's use numpy for speed:
    # neighbors_u = np.where(adj[u])[0]
    # neighbors_v = np.where(adj[v])[0]
    # For each x in neighbors_u where x != v:
    #     common = neighbors of x that are also in neighbors_v, excluding u and v and x
    
    # Actually, the most efficient way with numpy:
    # A C4 is formed iff there exists x != u,v such that:
    #   adj[u,x] and adj[x, y] and adj[y, v] for some y != u,v,x
    # This is equivalent to: the number of length-2 paths from x to v (via y) is > 0
    # for some x in N(u)\{v}, where the intermediate y is not u or v.
    
    # Let's think differently. Adding edge (u,v) creates a C4 iff there is a path u-x-y-v
    # of length 3 in the current graph.
    # This happens iff there exists x such that x is adjacent to u, and x has a neighbor y
    # (y != u, y != v) that is adjacent to v.
    
    # So: for x in N(u) \ {v}:
    #     if (N(x) \ {u, v}) intersects N(v): return True
    
    # Using numpy:
    # N_u = set of neighbors of u
    # N_v = set of neighbors of v
    # For x in N_u - {v}:
    #     N_x = set of neighbors of x
    #     if (N_x - {u, v}) & N_v: return True
    
    # Let's implement with numpy arrays for speed:
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]
    
    for x in nu:
        if x == v:
            continue
        nx = np.where(adj[x])[0]
        # Check if any neighbor of x (excluding u and v) is in nv
        # mask: nx != u and nx != v and adj[nx, v]
        # But adj[nx, v] is just checking if nx is in nv
        # So: is there y in nx such that y != u, y != v, and y in nv?
        # i.e., (nx \ {u, v}) ∩ nv != empty
        
        # Use numpy:
        # candidates = nx[(nx != u) & (nx != v)]
        # if len(np.intersect1d(candidates, nv)) > 0: return True
        
        # Even faster: 
        # Check if any element of nx (excluding u, v) has adj[y, v] = True
        # adj[nx, v] gives a boolean array. We need to exclude u and v from nx.
        
        mask = (nx != u) & (nx != v)
        if np.any(adj[nx[mask], v]):
            return True
    
    return False


def _try_construct_fast(n, required_min_deg, deadline, seed):
    """
    Greedy construction with efficient C4 checking.
    """
    random.seed(seed)
    rng = np.random.RandomState(seed)
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Start with a random permutation of vertices
    perm = list(range(n))
    random.shuffle(perm)
    
    # Try adding edges in random order, prioritizing low-degree vertices
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We'll use a priority queue approach: always add the edge that connects
    # the two lowest-degree vertices (among those that don't create C4)
    
    max_iterations = 5000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        min_deg = degrees.min()
        
        if min_deg >= required_min_deg:
            break
        
        # Find the best edge to add: one that doesn't create C4 and maximally helps
        # Try edges involving the minimum-degree vertex first
        min_deg_vertices = np.where(degrees == min_deg)[0]
        
        added = False
        # Shuffle to add randomness
        random.shuffle(min_deg_vertices.tolist())
        
        for u in min_deg_vertices:
            if time.time() > deadline:
                break
            
            # Try adding edges from u to other vertices, preferring those with low degree
            candidates = []
            for v in range(n):
                if v != u and not adj[u, v]:
                    candidates.append(v)
            
            # Sort candidates by degree (lower first)
            candidates.sort(key=lambda v: degrees[v])
            
            for v in candidates:
                if time.time() > deadline:
                    break
                
                if adj[u, v]:
                    continue
                
                if not _check_c4_fast(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            # No more edges can be added without creating C4
            # Try a different random restart
            break
    
    degrees = adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    return None


def _local_search_fast(n, required_min_deg, deadline):
    """
    Local search: start with a greedy solution, then try edge swaps to improve min degree.
    """
    # First, get an initial solution
    for seed in range(10):
        result = _try_construct_fast(n, required_min_deg, deadline, seed)
        if result is not None:
            return result
    
    # Try to build from scratch with a different strategy:
    # Start with a high-degree subgraph and add edges carefully
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Use a different heuristic: build a graph that is close to complete
    # but remove edges to avoid C4s
    
    # Start with complete graph and remove edges that create C4s
    # This is expensive, so use a heuristic instead
    
    # Try: start with a random sparse graph and add edges
    for seed in range(50):
        if time.time() > deadline:
            break
        
        random.seed(seed + 1000)
        adj = np.zeros((n, n), dtype=bool)
        
        # Random initial edges
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.3:
                    if not _check_c4_fast(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
        
        # Now try to add more edges to increase min degree
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            degrees = adj.sum(axis=1)
            min_deg = degrees.min()
            
            if min_deg >= required_min_deg:
                break
            
            min_deg_vertices = np.where(degrees == min_deg)[0]
            random.shuffle(min_deg_vertices.tolist())
            
            added = False
            for u in min_deg_vertices:
                if time.time() > deadline:
                    break
                
                candidates = [v for v in range(n) if v != u and not adj[u, v]]
                candidates.sort(key=lambda v: degrees[v])
                
                for v in candidates:
                    if time.time() > deadline:
                        break
                    if not _check_c4_fast(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        added = True
                        break
                
                if added:
                    break
            
            if not added:
                break
        
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
