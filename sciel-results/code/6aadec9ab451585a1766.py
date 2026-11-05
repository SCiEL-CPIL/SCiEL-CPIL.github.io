"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Strategy:
    # 1. Try a few high-quality structured initializations (e.g., based on Petersen/Hoffman-Singleton ideas)
    # 2. Refine using local search (edge swaps) to maximize edges while maintaining K2,3-free property
    
    # Known good base: The 5-disjoint K4 gives 30 edges.
    # We can do better. A 3-regular graph on 20 vertices has 30 edges.
    # A 4-regular graph on 20 vertices has 40 edges.
    # Let's try to build a 4-regular or 5-regular graph.
    
    # Initialization 1: Random 4-regular graph
    edges1 = _random_regular_graph(n, 4, seed=42)
    if edges1:
        edges1 = _local_search(edges1, n, deadline)
        if len(edges1) > best_score:
            best_score = len(edges1)
            best_edges = edges1
            
    # Initialization 2: Random 5-regular graph
    edges2 = _random_regular_graph(n, 5, seed=123)
    if edges2:
        edges2 = _local_search(edges2, n, deadline)
        if len(edges2) > best_score:
            best_score = len(edges2)
            best_edges = edges2
            
    # Initialization 3: Greedy construction with random restarts
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
        edges = _greedy_construct(n, seed)
        edges = _local_search(edges, n, deadline)
        if len(edges) > best_score:
            best_score = len(edges)
            best_edges = edges
            
    return best_edges

def _random_regular_graph(n, k, seed):
    """Generate a random k-regular graph on n vertices."""
    random.seed(seed)
    if k >= n or (n * k) % 2 != 0:
        return []
    
    # Configuration model
    half_edges = list(range(n * k))
    random.shuffle(half_edges)
    
    adj = {i: set() for i in range(n)}
    edges = []
    
    for i in range(0, n * k, 2):
        u = half_edges[i] // k
        v = half_edges[i + 1] // k
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
        else:
            # Retry if loop or multi-edge
            return [] # Simplified: just fail and try another seed
    
    # Check if it's actually k-regular
    for i in range(n):
        if len(adj[i]) != k:
            return []
            
    return edges

def _greedy_construct(n, seed):
    """
    Greedy construction of K2,3-free graph.
    Start empty, add edges in random order if they don't create K2,3.
    """
    random.seed(seed)
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    adj = {i: set() for i in range(n)}
    edges = []
    
    for u, v in all_edges:
        if _can_add_edge(u, v, adj, n):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            
    return edges

def _can_add_edge(u, v, adj, n):
    """Check if adding edge (u,v) creates a K2,3."""
    # A K2,3 is formed if there are 3 common neighbors of u and v.
    # Common neighbors of u and v:
    common = adj[u].intersection(adj[v])
    if len(common) >= 3:
        return False
    return True

def _local_search(edges, n, deadline):
    """
    Local search: try to add edges, or swap edges to increase density.
    Since we want to maximize edges, we primarily try to add missing edges.
    If we can't add any, we might try swaps (remove one, add two).
    """
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    current_edges = len(edges)
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    # First pass: Try to add any missing edge that doesn't violate K2,3
    missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u,v) not in edge_set]
    random.shuffle(missing_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u, v in missing_edges:
            if (u, v) in edge_set:
                continue
            if _can_add_edge(u, v, adj, n):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                current_edges += 1
                improved = True
                if time.time() > deadline - 5:
                    break
        
    # Convert back to list
    return [[u, v] for u, v in edge_set]

# EVOLVE-BLOCK-END
