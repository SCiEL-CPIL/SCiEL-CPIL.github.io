"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_regular_induced(adj, mask, n):
    """Check if the induced subgraph on vertices in mask is regular.
    adj[i] is a bitmask of neighbors of vertex i.
    mask is a bitmask of the vertex subset.
    Returns True if regular (all vertices in the subset have same degree within subset).
    """
    count = bin(mask).count('1')
    if count < 6:
        return False
    # Compute degree of each vertex in the induced subgraph
    first_deg = -1
    m = mask
    while m:
        # extract lowest set bit
        lb = m & (-m)
        idx = lb.bit_length() - 1
        deg = bin(adj[idx] & mask).count('1')
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
        m ^= lb
    return True

def _has_regular_subgraph(adj, n):
    """Check if any subset of 6+ vertices induces a regular subgraph."""
    # Check all subsets of size 6 through n
    # For efficiency, we can check sizes 6, 7, ..., n
    # But that's a lot. Let's be smart: check size 6 first (most likely to find one)
    # Actually we need to check ALL sizes >= 6
    # Optimize: if a subset S of size k >= 6 is regular, then... 
    # Let's just iterate through all masks with popcount >= 6
    # For n up to ~15, this is feasible with early termination
    
    # Generate all masks with popcount >= 6
    # For n <= 12, total masks = 4096, manageable
    # For larger n, we need smarter approach
    
    # Strategy: check all subsets of size 6 first (C(n,6) subsets)
    # If none found, check size 7, etc.
    
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_induced(adj, mask, n):
                return True
    return False

def _check_valid(adj, n):
    """Verify the graph has no regular induced subgraph on 6+ vertices."""
    return not _has_regular_subgraph(adj, n)

def _random_graph(n, p=0.5):
    """Generate a random graph with edge probability p."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _greedy_construct(n, seed=42):
    """Try to construct a valid graph greedily by adding edges one at a time."""
    rng = random.Random(seed)
    adj = [0] * n
    # Try adding edges in random order, keeping the graph valid
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edge_list)
    
    for (u, v) in edge_list:
        # Tentatively add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if _has_regular_subgraph(adj, n):
            # Remove edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
    return adj

def _try_local_search(adj, n, max_steps=500):
    """Try to improve a graph by flipping edges to remove regular subgraphs."""
    adj = adj[:]
    for step in range(max_steps):
        if not _has_regular_subgraph(adj, n):
            return adj
        # Find a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        # Flip edge (u, v)
        has_edge = (adj[u] >> v) & 1
        if has_edge:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s kill)
    
    best_n = 5
    best_edges = []
    
    # Start with known valid small graphs and try to extend
    # Try various approaches
    
    # Approach 1: Random graphs with different n and p values
    for n in range(6, 16):
        if time.time() > deadline:
            break
        for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline:
                break
            for trial in range(200):
                if time.time() > deadline:
                    break
                adj = _random_graph(n, p)
                if not _has_regular_subgraph(adj, n):
                    if n > best_n:
                        best_n = n
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (adj[i] >> j) & 1:
                                    edges.append([i, j])
                        best_edges = edges
                    break  # Found valid graph for this n, move to larger n
    
    # Approach 2: Greedy construction
    for n in range(best_n + 1, 16):
        if time.time() > deadline:
            break
        for seed in range(50):
            if time.time() > deadline:
                break
            adj = _greedy_construct(n, seed)
            if not _has_regular_subgraph(adj, n):
                if n > best_n:
                    best_n = n
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_edges = edges
                break
    
    # Approach 3: Local search from random starts
    for n in range(max(best_n + 1, 8), 16):
        if time.time() > deadline:
            break
        for trial in range(50):
            if time.time() > deadline:
                break
            adj = _random_graph(n, 0.5)
            adj = _try_local_search(adj, n, max_steps=200)
            if not _has_regular_subgraph(adj, n):
                if n > best_n:
                    best_n = n
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_edges = edges
                break
    
    # If we couldn't find anything better than 5, return the trivial one
    if best_n <= 5:
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
