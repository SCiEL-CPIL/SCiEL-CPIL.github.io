"""Rewritten program for K2,3-free graph on 20 vertices."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try multiple random seeds with greedy construction
    for seed in range(500):
        if time.time() > deadline - 10.0:
            break
        
        edges = _greedy_construct(n, all_edges, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges
    
    # If we still have time, try local search improvements
    if time.time() < deadline - 20.0 and best_edges:
        improved = _local_search(n, best_edges, all_edges, deadline - 15.0)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)
    
    return best_edges

def _greedy_construct(n, all_edges, seed):
    """
    Greedy construction of K2,3-free graph.
    Start empty, add edges in random order if they don't create K2,3.
    """
    random.seed(seed)
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edges = []
    
    # Shuffle edges for random order
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    for u, v in shuffled_edges:
        if _can_add_edge(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return edges

def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u,v) would create a K2,3.
    A K2,3 exists if there are two vertices with three common neighbors.
    
    When we add edge (u,v), we need to check:
    1. For any pair (x,y) where x != u,v and y != u,v: 
       if x and y both become adjacent to u or v in a way that creates 3 common neighbors
    2. Specifically, after adding (u,v):
       - u and v now have each other as neighbors
       - For any other vertex w, if w is adjacent to both u and v, 
         then u and v share w as a common neighbor
       - We need to ensure no two vertices have >= 3 common neighbors
    
    The key insight: adding edge (u,v) can only create new K2,3 if:
    - There exist vertices a, b such that |N(a) ∩ N(b)| >= 3 after the addition
    - The new edge (u,v) adds u to N(v) and v to N(u)
    - So we need to check pairs that include u or v
    
    More precisely, after adding (u,v):
    - For any vertex w != u,v:
      - If w is adjacent to u, then N(u) gains v, and N(w) gains nothing new from this edge
      - If w is adjacent to v, then N(v) gains u, and N(w) gains nothing new
    - The common neighbors of u and v are: N(u) ∩ N(v) (before adding) plus potentially each other
      Actually, after adding (u,v), N(u) includes v and N(v) includes u, but u is not in N(u) 
      (simple graph), so common neighbors of u and v are N(u) ∩ N(v) where both now include 
      the other... wait, no. In a simple graph, N(u) is the set of neighbors of u. After adding 
      edge (u,v), v ∈ N(u) and u ∈ N(v). Common neighbors of u and v are vertices w such that 
      w ∈ N(u) AND w ∈ N(v). Since u ∉ N(u) and v ∉ N(v), u is not a common neighbor of u and v, 
      and v is not a common neighbor of u and v. So common neighbors of u and v after adding the 
      edge are exactly N_old(u) ∩ N_old(v) (the common neighbors before adding the edge).
    
    - For any vertex w != u,v:
      - Common neighbors of u and w: N(u) ∩ N(w). After adding (u,v), N(u) gains v. 
        So new common neighbors = old common neighbors ∪ {v if v ∈ N(w)}
      - Common neighbors of v and w: N(v) ∩ N(w). After adding (u,v), N(v) gains u. 
        So new common neighbors = old common neighbors ∪ {u if u ∈ N(w)}
    
    So we need to check:
    1. |N_old(u) ∩ N_old(v)| < 3 (common neighbors of u and v)
    2. For each w != u,v:
       - |N_old(u) ∩ N_old(w)| + (1 if v ∈ N_old(w) else 0) < 3
       - |N_old(v) ∩ N_old(w)| + (1 if u ∈ N_old(w) else 0) < 3
    """
    # Check 1: common neighbors of u and v
    common_uv = adj[u] & adj[v]
    if len(common_uv) >= 3:
        return False
    
    # Check 2: for each w != u,v
    for w in range(len(adj)):
        if w == u or w == v:
            continue
        
        # Common neighbors of u and w after adding (u,v)
        common_uw = adj[u] & adj[w]
        if v in adj[w]:
            common_uw = common_uw | {v}
        if len(common_uw) >= 3:
            return False
        
        # Common neighbors of v and w after adding (u,v)
        common_vw = adj[v] & adj[w]
        if u in adj[w]:
            common_vw = common_vw | {u}
        if len(common_vw) >= 3:
            return False
    
    return True

def _local_search(n, edges, all_edges, deadline):
    """
    Try to improve the graph by swapping edges.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    improved = True
    current_edges = [list(e) for e in edges]
    
    while improved and time.time() < deadline:
        improved = False
        # Try adding each missing edge
        for u, v in all_edges:
            if time.time() > deadline:
                return current_edges
            
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            
            if _can_add_edge(adj, u, v):
                # Add the edge
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u,v), max(u,v)))
                current_edges.append([u, v])
                improved = True
    
    return current_edges
# EVOLVE-BLOCK-END
