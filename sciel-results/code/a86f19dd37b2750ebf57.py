import time
import random

def _is_3_colorable(adj, n):
    """
    Checks if the graph defined by adjacency bitmasks 'adj' on 'n' vertices 
    is 3-colorable.
    """
    colors = [-1] * n
    
    # Order vertices by degree (descending) to prune faster
    # Pre-compute degrees
    degrees = [adj[i].bit_count() for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map original index to position in order for quick access?
    # Or just iterate through order.
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        
        # Determine forbidden colors
        forbidden = 0
        neighbors = adj[v]
        while neighbors:
            lsb = neighbors & -neighbors
            u = lsb.bit_length() - 1
            c = colors[u]
            if c != -1:
                forbidden |= (1 << c)
            neighbors ^= lsb
        
        available = 7 ^ forbidden
        
        while available:
            lsb = available & -available
            c = lsb.bit_length() - 1
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
            available ^= lsb
            
        return False
    
    return backtrack(0)

def _get_clebsch_edges():
    """
    Generates the edges of the Clebsch graph (16 vertices, 48 edges).
    Vertices are 0..15 (4-bit binary).
    Edge exists if Hamming distance is 2.
    """
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            xor = u ^ v
            if xor.bit_count() == 2:
                edges.append([u, v])
    return edges

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    # 1. Initialize with Clebsch graph (guaranteed 48 edges, 4-chromatic)
    clebsch_edges = _get_clebsch_edges()
    best_count = len(clebsch_edges)
    best_edges = clebsch_edges
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Shuffle for randomization
    # We will run multiple random greedy constructions
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        edge_count = 0
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a triangle
            # Triangle exists if there is a common neighbor
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        # Only check 3-colorability if we might beat the best
        # If edge_count <= best_count, skip (unless we want to explore, but max edges is goal)
        if edge_count > best_count:
            if not _is_3_colorable(adj, n):
                best_count = edge_count
                # Reconstruct edges
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                best_edges = edges
                
        # Optional: If we have time, try local search on the best found?
        # Since greedy produces maximal graphs, we can't add edges.
        # We can try to "swap" edges to find a different maximal graph with more edges?
        # This is complex. The random restarts are usually sufficient for n=16.
        
    return best_edges
