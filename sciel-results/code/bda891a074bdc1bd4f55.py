"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _has_c4(n, adj):
    """Check if graph with adjacency sets has a C4."""
    for i in range(n):
        for j in range(i + 1, n):
            if j in adj[i]:
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return True
    return False

def _min_degree(n, adj):
    return min(len(adj[i]) for i in range(n))

def _try_construct(n, seed=0, max_restarts=50):
    """Try to build a C4-free graph on n vertices with min degree >= n-39."""
    min_deg_needed = n - 39
    if min_deg_needed < 1:
        min_deg_needed = 1
    
    random.seed(seed)
    
    for restart in range(max_restarts):
        # Start with a random sparse graph and greedily add edges
        adj = [set() for _ in range(n)]
        
        # Try to build a high-min-degree C4-free graph
        # Strategy: start with a random regular-ish graph, then add edges greedily
        
        # First, create a base graph: random edges ensuring some connectivity
        # We need min degree >= min_deg_needed
        # Let's try a greedy approach: randomly order vertices and add edges
        
        # Start with empty graph
        # Randomly permute vertices
        verts = list(range(n))
        random.shuffle(verts)
        
        # For each vertex, try to connect to enough others
        # We'll use a greedy edge-addition approach
        
        # Generate all possible edges
        all_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                all_edges.append((i, j))
        
        random.shuffle(all_edges)
        
        # Greedily add edges if they don't create a C4
        for u, v in all_edges:
            # Check if adding (u,v) creates a C4
            # A C4 is created if there exist x, y such that x~u, x~v, y~u, y~v
            # i.e., |N(u) ∩ N(v)| >= 2 after adding the edge
            common = adj[u] & adj[v]
            if len(common) < 2:
                adj[u].add(v)
                adj[v].add(u)
        
        # Check min degree
        md = _min_degree(n, adj)
        if md >= min_deg_needed:
            # Verify no C4
            if not _has_c4(n, adj):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best = None
    
    # Try from largest n down
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        
        min_deg_needed = n - 39
        if min_deg_needed < 1:
            continue
        
        # Try multiple seeds
        seed = 0
        while time.time() < deadline:
            result = _try_construct(n, seed=seed, max_restarts=10)
            if result is not None:
                if best is None or n > best['n']:
                    best = result
                    return best
            seed += 1
    
    # Fallback: return the trivial 39-vertex empty graph
    if best is None:
        return {'n': 39, 'edges': []}
    
    return best
# EVOLVE-BLOCK-END
