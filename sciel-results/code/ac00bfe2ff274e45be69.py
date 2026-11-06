"""K4-free graph on 18 vertices minimizing max triangle-free induced subgraph size."""
import time
import random
import itertools
from collections import defaultdict

N = 18
DEADLINE_MARGIN = 10  # seconds before hard limit to stop

def _time_left(deadline):
    return deadline - time.time()

def _is_k4_free(adj):
    """Check if graph with adjacency bitmask list is K4-free."""
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                for k in range(N):
                    if common & (1 << k):
                        common2 = adj[i] & adj[j] & adj[k]
                        for l in range(N):
                            if common2 & (1 << l):
                                return False
    return True

def _max_triangle_free(adj):
    """Find the largest triangle-free induced subgraph size via branch and bound."""
    # Use bitmask for vertex set
    all_mask = (1 << N) - 1
    
    # Precompute: for each vertex, its neighbors
    # Triangle-free means no three vertices form a triangle
    # A set S is triangle-free if for every pair u,v in S that are adjacent, 
    # there's no w in S adjacent to both u and v
    
    # Branch and bound: try to find if there exists a triangle-free set of size > best
    best_size = 0
    best_set = 0
    
    def has_triangle_in_set(mask):
        """Check if the set of vertices in mask contains a triangle."""
        # Get list of vertices
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        n_v = len(verts)
        if n_v < 3:
            return False
        
        # Check all triples
        for i in range(n_v):
            for j in range(i+1, n_v):
                u, v = verts[i], verts[j]
                if adj[u] & (1 << v):
                    # Check if any w in verts adjacent to both u and v
                    common = adj[u] & adj[v]
                    for k in range(n_v):
                        w = verts[k]
                        if w != u and w != v and (common & (1 << w)):
                            return True
        return False
    
    def is_triangle_free(mask):
        return not has_triangle_in_set(mask)
    
    # Branch and bound to find maximum triangle-free subset
    # We want to maximize |S| such that S is triangle-free
    
    # Use a recursive search with pruning
    def search(available, current_mask, current_size):
        nonlocal best_size, best_set
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_mask
        
        if available == 0:
            return
        
        # Upper bound: current_size + popcount(available)
        if current_size + bin(available).count('1') <= best_size:
            return
        
        # Try adding vertices
        m = available
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            new_mask = current_mask | lsb
            if is_triangle_free(new_mask):
                # Remove v and all vertices that would create a triangle with v and any two in current_mask
                # Actually, just remove v from available
                new_available = available & ~(1 << v)
                # Also, we can prune: if adding v creates potential issues
                # For efficiency, just recurse
                search(new_available, new_mask, current_size + 1)
            m ^= lsb
    
    search(all_mask, 0, 0)
    return best_size

def _score(adj):
    """Return the max triangle-free induced subgraph size."""
    return _max_triangle_free(adj)

def _local_search(adj, deadline):
    """Local search: try to reduce the max triangle-free subgraph size."""
    best_adj = list(adj)
    best_score = _score(best_adj)
    
    while _time_left(deadline) > 0:
        improved = False
        # Try adding edges
        for i in range(N):
            for j in range(i+1, N):
                if not (best_adj[i] & (1 << j)):
                    # Try adding edge (i,j)
                    new_adj = list(best_adj)
                    new_adj[i] |= (1 << j)
                    new_adj[j] |= (1 << i)
                    if _is_k4_free(new_adj):
                        s = _score(new_adj)
                        if s < best_score:
                            best_adj = new_adj
                            best_score = s
                            improved = True
        # Try removing edges
        for i in range(N):
            for j in range(i+1, N):
                if best_adj[i] & (1 << j):
                    new_adj = list(best_adj)
                    new_adj[i] &= ~(1 << j)
                    new_adj[j] &= ~(1 << i)
                    if _is_k4_free(new_adj):
                        s = _score(new_adj)
                        if s < best_score:
                            best_adj = new_adj
                            best_score = s
                            improved = True
        if not improved:
            break
    
    return best_adj, best_score

def _random_k4_free_graph():
    """Generate a random K4-free graph."""
    # Start with empty graph, add edges that don't create K4
    adj = [0] * N
    order = list(range(N))
    random.shuffle(order)
    
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < 0.5:
                # Check if adding (i,j) creates a K4
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                if not _is_k4_free(adj):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
    return adj

def _complete_tripartite():
    """Complete tripartite K6,6,6."""
    adj = [0] * N
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    return adj

def _octahedral_extension():
    """Try a different structure: 3 groups of 6, but not complete tripartite."""
    adj = [0] * N
    # Add edges within groups sparsely, complete between groups
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 150 - DEADLINE_MARGIN  # 140s effective
    
    best_adj = None
    best_score = 999
    
    # Try several initial structures
    initial_graphs = []
    
    # 1. Complete tripartite
    initial_graphs.append(_complete_tripartite())
    
    # 2. Random K4-free graphs
    for _ in range(5):
        initial_graphs.append(_random_k4_free_graph())
    
    # 3. Octahedral-like
    initial_graphs.append(_octahedral_extension())
    
    for init_adj in initial_graphs:
        if _time_left(deadline) < 5:
            break
        adj, score = _local_search(init_adj, deadline)
        if score < best_score:
            best_score = score
            best_adj = adj
    
    # Convert to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
