import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 108

def _can_add_edge(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    A C6 is formed if there is a path of length 4 from u to v in G \ {u, v}.
    Path: u - a - b - c - d - v
    This is equivalent to checking if (A^4)[u, v] > 0 in the graph with u, v removed from adjacency?
    No, simple matrix power counts walks, not paths. Walks can repeat vertices.
    However, if there is a walk of length 4, does it imply a path?
    Not necessarily. But for sparse graphs, it's a good heuristic?
    Actually, we need an exact check.
    
    Let's use a fast iterative BFS/DFS with bitsets or numpy.
    Given the small size (40 vertices), bitsets are very fast in Python if implemented correctly.
    The previous implementation used a list of tuples in a stack.
    
    Let's try a vectorized check using numpy.
    We want to know if there is a path u -> ... -> v of length 4.
    Let A be the adjacency matrix (40x40 boolean).
    We remove u and v from consideration as internal nodes.
    
    Actually, the standard efficient check for C6 in this context (adding an edge) is:
    Does there exist a path of length 4 between u and v in G - {u, v}?
    
    Let's stick to the bitset DFS but optimize it.
    Or use NumPy matrix multiplication.
    (A^4)[u][v] counts the number of walks of length 4 from u to v.
    If this count is > 0, there is a walk.
    Does a walk of length 4 imply a C6?
    A walk u-a-b-c-d-v.
    If all vertices are distinct, it's a path of length 5 (which with edge u-v forms C6).
    If vertices repeat, e.g., u-a-u-a-d-v? No, u is not in internal nodes.
    Internal nodes a,b,c,d must be distinct from u and v.
    They can repeat each other.
    Example: u-a-b-a-d-v.
    This is a walk. Does it form a C6?
    The cycle would be u-a-b-a-d-v-u. This is not a simple cycle.
    So matrix power is not exact.
    
    However, for random graphs, the probability of such "fake" walks is low?
    No, we need exactness.
    
    Let's optimize the Python bitset DFS.
    The previous code:
    stack = []
    ...
    while stack:
        node, depth, mask = stack.pop()
        ...
    
    This is O(2^N) worst case but usually fast.
    
    Alternative:
    Use NumPy to compute reachable sets in parallel?
    
    Let's try to optimize the Python code first.
    Instead of a stack of (node, depth, mask), we can do level-by-level BFS.
    Level 1: Neighbors of u (excluding v)
    Level 2: Neighbors of Level 1 (excluding u, v, Level 1)
    Level 3: Neighbors of Level 2 (excluding u, v, Level 1, Level 2)
    Level 4: Neighbors of Level 3 (excluding u, v, Level 1, Level 2, Level 3)
    Check if v is in Neighbors of Level 4.
    
    This is much faster than DFS with masks because we don't carry the mask for every node, just the sets of nodes at each depth.
    Since depth is only 4, this is very efficient.
    """
    # adj is a list of integers (bitmasks)
    
    # Depth 1
    # Neighbors of u, excluding v
    level1 = adj[u] & ~(1 << v)
    if not level1:
        return False
        
    # Depth 2
    # Union of neighbors of all nodes in level1
    # Exclude u, v, and level1 nodes
    # Note: We must exclude level1 nodes to ensure distinctness?
    # Path u-a-b-c-d-v. a,b,c,d distinct.
    # So b cannot be a. c cannot be a or b. d cannot be a, b, c.
    # So yes, we must exclude previous levels.
    
    excluded = (1 << u) | (1 << v) | level1
    level2 = 0
    temp = level1
    while temp:
        bit = temp & -temp
        node = bit.bit_length() - 1
        temp ^= bit
        level2 |= adj[node]
    level2 &= ~excluded
    
    if not level2:
        return False
        
    # Depth 3
    excluded |= level2
    level3 = 0
    temp = level2
    while temp:
        bit = temp & -temp
        node = bit.bit_length() - 1
        temp ^= bit
        level3 |= adj[node]
    level3 &= ~excluded
    
    if not level3:
        return False
        
    # Depth 4
    excluded |= level3
    level4 = 0
    temp = level3
    while temp:
        bit = temp & -temp
        node = bit.bit_length() - 1
        temp ^= bit
        level4 |= adj[node]
    level4 &= ~excluded
    
    # Check if v is in level4
    if level4 & (1 << v):
        return True
        
    return False

def _greedy_with_local_search(n=40, seed=0, deadline=None):
    rng = random.Random(seed)
    adj = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if not _can_add_edge(u, v, adj):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add((u, v))
            
    # Local Search
    # We maintain a list of non-edges to try
    # And a list of edges to try swapping
    
    improved = True
    max_iter = 2000  # Increased iterations due to faster check
    
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Phase 1: Try to add
        non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
        rng.shuffle(non_edges)
        
        added = False
        for u, v in non_edges:
            if not _can_add_edge(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
                added = True
                improved = True
                break
                
        if added:
            continue
            
        # Phase 2: Try to swap
        # Remove one edge, try to add one non-edge
        edge_list = list(edges)
        rng.shuffle(edge_list)
        
        swapped = False
        # Limit swap attempts per iteration to save time
        for u, v in edge_list[:10]: # Only try first 10 shuffled edges
            # Remove u, v
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.remove((u, v))
            
            # Try to add a non-edge
            non_edges_curr = [(x, y) for x, y in all_edges if (x, y) not in edges]
            rng.shuffle(non_edges_curr)
            
            added_back = False
            for x, y in non_edges_curr:
                if not _can_add_edge(x, y, adj):
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    edges.add((x, y))
                    added_back = True
                    swapped = True
                    improved = True
                    break
                    
            if not added_back:
                # Revert swap
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
            
            if swapped:
                break
                
    return list(edges)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = []
    best_count = 0
    
    # Run multiple seeds
    for seed in range(1000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges = _greedy_with_local_search(40, seed, _DEADLINE)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 110:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]
