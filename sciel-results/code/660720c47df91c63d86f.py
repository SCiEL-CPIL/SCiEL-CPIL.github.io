"""K3,3-free dense graph on 20 vertices via greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Adjacency as set of frozensets for fast lookup
    # adj[u] = set of neighbors of u
    
    def is_k33_free(adj):
        """Check if graph is K3,3-free by checking all pairs of 3-sets."""
        for a1, a2, a3 in itertools.combinations(range(n), 3):
            common = adj[a1] & adj[a2] & adj[a3]
            if len(common) >= 3:
                return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if u == v:
            return False
        if v in adj[u]:
            return False
        
        # Optimized check:
        # K3,3 formed by A={u,x,y} and B={v,p,q}
        # Requires:
        # 1. x,y in N(v)
        # 2. p,q in N(u)
        # 3. x,y not in {u,v}
        # 4. p,q not in {u,v,x,y}
        # 5. Edges (x,p), (x,q), (y,p), (y,q) exist.
        
        # Get neighbors of v (excluding u)
        nv = [w for w in adj[v] if w != u]
        # Get neighbors of u (excluding v)
        nu = [w for w in adj[u] if w != v]
        
        # Iterate over pairs (x,y) from N(v)
        for x, y in itertools.combinations(nv, 2):
            # For each pair (x,y), we need to find a pair (p,q) from N(u)
            # such that p,q are not x,y and are connected to both x and y.
            # Pre-filter candidates for p,q: must be in N(u) AND N(x) AND N(y)
            candidates = []
            for w in nu:
                if w == x or w == y:
                    continue
                if w in adj[x] and w in adj[y]:
                    candidates.append(w)
            
            if len(candidates) >= 2:
                return False
        
        return True
    
    # Greedy construction
    def greedy_construct():
        adj = [set() for _ in range(n)]
        edges = []
        
        # Try adding edges in order of decreasing "safety"
        # Start with all possible edges, try to add them greedily
        for u in range(n):
            for v in range(u+1, n):
                if can_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
        
        return adj, edges
    
    def rebuild_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Local search: try to add more edges, and if stuck, remove some and retry
    def local_search(adj, edges):
        nonlocal deadline
        
        # Phase 1: Greedy addition
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() >= deadline - 1:
                    break
                for v in range(u+1, n):
                    if v not in adj[u] and can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        improved = True
                        break
                if improved:
                    break
            
        # Rebuild edges to ensure consistency
        edges = rebuild_edges(adj)
        current_count = len(edges)
        
        # Phase 2: Edge removal and re-optimization
        if time.time() < deadline - 10:
            # Try removing each edge and see if we can gain more
            # We do this for a limited number of attempts to save time
            attempts = 0
            max_attempts = 50 # Limit attempts to avoid timeout
            
            for u in range(n):
                if time.time() >= deadline - 5 or attempts >= max_attempts:
                    break
                for v in list(adj[u]):
                    if v <= u:
                        continue
                    if time.time() >= deadline - 5 or attempts >= max_attempts:
                        break
                    
                    attempts += 1
                    
                    # Save state
                    adj_copy = [set(s) for s in adj]
                    
                    # Remove edge (u,v)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    # Try to add more edges (greedy)
                    imp = True
                    while imp and time.time() < deadline - 5:
                        imp = False
                        for u2 in range(n):
                            if time.time() >= deadline - 5:
                                break
                            for v2 in range(u2+1, n):
                                if v2 not in adj[u2] and can_add_edge(adj, u2, v2):
                                    adj[u2].add(v2)
                                    adj[v2].add(u2)
                                    imp = True
                                    break
                            if imp:
                                break
                    
                    new_edges = rebuild_edges(adj)
                    
                    if len(new_edges) > current_count:
                        # Accept change
                        current_count = len(new_edges)
                        edges = new_edges
                    else:
                        # Revert
                        adj = adj_copy
        
        return adj, edges
    
    # Run greedy construction with multiple random orderings
    best_edges = None
    best_adj = None
    
    # Deterministic greedy first
    adj, edges = greedy_construct()
    adj, edges = local_search(adj, edges)
    if best_edges is None or len(edges) > len(best_edges):
        best_edges = edges[:]
        best_adj = [set(s) for s in adj]
    
    # Try random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() >= deadline - 10:
            break
        
        # Random edge ordering
        all_edges = [[u, v] for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        adj = [set() for _ in range(n)]
        edges = []
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        adj, edges = local_search(adj, edges)
        
        if best_edges is None or len(edges) > len(best_edges):
            best_edges = edges[:]
            best_adj = [set(s) for s in adj]
    
    return best_edges
# EVOLVE-BLOCK-END
