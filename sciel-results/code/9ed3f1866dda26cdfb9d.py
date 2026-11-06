import random
import time

def construct():
    n = 20
    best_edges = 0
    best_graph = None
    start_time = time.time()
    timeout = 110  # Safety margin
    
    def has_k33(adj):
        # Check if graph has K3,3
        # For every 3 vertices u1, u2, u3, check if common neighbors >= 3
        # This is O(N^4) which is 20^4 = 160000, very fast.
        for u1 in range(n):
            for u2 in range(u1+1, n):
                for u3 in range(u2+1, n):
                    common = adj[u1] & adj[u2] & adj[u3]
                    if len(common) >= 3:
                        return True, (u1, u2, u3, common)
        return False, None

    def count_violations(adj):
        # Returns list of (u1, u2, u3, common_set)
        violations = []
        for u1 in range(n):
            for u2 in range(u1+1, n):
                for u3 in range(u2+1, n):
                    common = adj[u1] & adj[u2] & adj[u3]
                    if len(common) >= 3:
                        violations.append((u1, u2, u3, common))
        return violations

    def remove_edge(adj, u, v):
        adj[u].remove(v)
        adj[v].remove(u)

    def add_edge(adj, u, v):
        adj[u].add(v)
        adj[v].add(u)

    def is_add_safe(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        # As derived:
        # There exist distinct b1, b2 in N(u) \ {v} such that 
        # |N(v) & N(b1) & N(b2) \ {u, v, b1, b2}| >= 2
        # Note: b1, b2 must be in N(u).
        # And a1, a2 must be in N(v) & N(b1) & N(b2).
        # Also a1, a2 must be distinct from u, v, b1, b2.
        
        # Optimization: 
        # Candidates for b1, b2 are in N(u).
        # For each pair (b1, b2) in N(u):
        #   if b1 == v or b2 == v: continue (v is not in N(u) yet, but if it were, we handle it)
        #   Actually v is not in N(u) because we are adding it.
        #   So b1, b2 are in N(u).
        #   common = N(v) & N(b1) & N(b2)
        #   common -= {u, v, b1, b2}
        #   if len(common) >= 2: return False
        
        for b1 in adj[u]:
            for b2 in adj[u]:
                if b1 == b2: continue
                if b1 == v or b2 == v: continue # v is not in adj[u] anyway
                common = adj[v] & adj[b1] & adj[b2]
                common -= {u, v, b1, b2}
                if len(common) >= 2:
                    return False
        return True

    def local_search(p):
        # Initialize
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    add_edge(adj, i, j)
        
        # Phase 1: Remove edges to fix K3,3
        # We will iterate until no violations
        while True:
            violations = count_violations(adj)
            if not violations:
                break
            
            # Count edge frequencies in violations
            # A violation (u1, u2, u3, C) involves edges:
            # (u1, c), (u2, c), (u3, c) for c in C.
            # Also edges between u1,u2,u3 are not part of K3,3 definition (bipartite).
            # So only edges between {u1,u2,u3} and C are relevant.
            edge_counts = {}
            for u1, u2, u3, C in violations:
                for c in C:
                    edge_counts[(u1, c)] = edge_counts.get((u1, c), 0) + 1
                    edge_counts[(u2, c)] = edge_counts.get((u2, c), 0) + 1
                    edge_counts[(u3, c)] = edge_counts.get((u3, c), 0) + 1
            
            # Pick edge with max count
            if not edge_counts:
                break
                
            max_count = -1
            best_edge = None
            for e, cnt in edge_counts.items():
                if cnt > max_count:
                    max_count = cnt
                    best_edge = e
            
            if best_edge:
                remove_edge(adj, best_edge[0], best_edge[1])
        
        # Phase 2: Add edges
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        if is_add_safe(adj, i, j):
                            add_edge(adj, i, j)
                            improved = True
        
        # Count edges
        edge_count = sum(len(s) for s in adj) // 2
        return adj, edge_count

    # Try multiple restarts with different p
    for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for _ in range(10):
            if time.time() - start_time > timeout:
                break
            
            adj, edge_count = local_search(p)
            if edge_count > best_edges:
                best_edges = edge_count
                best_graph = [list(s) for s in adj]
                # Convert to list of pairs
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                best_graph = edges
                
        if time.time() - start_time > timeout:
            break

    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_graph
