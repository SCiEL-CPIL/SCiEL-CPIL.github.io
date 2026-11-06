import random
import time
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_edges = 0
    best_graph = None
    
    def is_k33_free(adj):
        # Check if any 3 vertices have 3 common neighbors
        # adj[u] is a set of neighbors
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common_uv = adj[u] & adj[v]
                    for w in range(v + 1, n):
                        if w in adj[u] and w in adj[v]:
                            # u, v, w are a triangle? No, K3,3 is bipartite.
                            # The condition is for ANY 3 vertices, not necessarily adjacent.
                            pass
        # Correct check:
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = adj[u] & adj[v] & adj[w]
                    if len(common) >= 3:
                        return False
        return True

    def count_common_violations(adj, u, v):
        # Check if adding edge (u, v) creates a K3,3
        # New common neighbors for triples involving u or v
        # Actually, just check all triples after hypothetical addition?
        # Or check specific condition:
        # Adding (u,v) makes v a neighbor of u and u a neighbor of v.
        # For any pair (a,b), if v is in N(a) and N(b), then v becomes a common neighbor of (u,a,b) if u is in N(a) and N(b)?
        # No.
        # Let's just simulate the addition and check the condition for triples involving u or v.
        # Triples involving u: (u, a, b). Common neighbors = N(u) & N(a) & N(b).
        # N(u) gains v. So new common = old common + {v} if v in N(a) and v in N(b).
        # So we need to ensure that for all a < b, if v in N(a) and v in N(b), then old common size < 2.
        # Similarly for triples involving v: (v, a, b).
        
        # Let's be simpler and safer: just check all triples. n=20, 1140 triples.
        # But we can optimize.
        
        # Check triples (u, a, b) for all a,b != u
        for a in range(n):
            if a == u: continue
            for b in range(n):
                if b == u or b == a: continue
                # Check if v is in N(a) and N(b)
                if v in adj[a] and v in adj[b]:
                    # v will be a common neighbor of u, a, b
                    # Current common neighbors of u, a, b
                    common = adj[u] & adj[a] & adj[b]
                    # If v is already in common, it doesn't increase count. But v is not in N(u) yet.
                    # So if v in N(a) and N(b), it adds 1 to the count.
                    # So we need len(common) < 2.
                    if len(common) >= 2:
                        return False
        # Check triples (v, a, b)
        for a in range(n):
            if a == v: continue
            for b in range(n):
                if b == v or b == a: continue
                if u in adj[a] and u in adj[b]:
                    common = adj[v] & adj[a] & adj[b]
                    if len(common) >= 2:
                        return False
        return True

    def try_add_edge(adj, u, v):
        if u == v: return False
        if v in adj[u]: return False
        if not is_k33_free_after_add(adj, u, v):
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    def is_k33_free_after_add(adj, u, v):
        # Optimized check
        # Triples containing u: (u, a, b)
        # New common neighbor is v.
        # v is common to u,a,b iff v in N(a) and v in N(b) (and v in N(u) which is true after add).
        # So if v in N(a) and v in N(b), then v is in N(u) & N(a) & N(b).
        # We need total count < 3.
        # Existing count = len(N(u) & N(a) & N(b)).
        # If v is already in that set, count doesn't change. But v is not in N(u) yet.
        # So if v in N(a) and N(b), it adds 1.
        # So we need len(N(u) & N(a) & N(b)) < 2 whenever v in N(a) and v in N(b).
        
        # Also, triples containing v: (v, a, b)
        # u is new neighbor of v.
        # If u in N(a) and u in N(b), it adds 1 to len(N(v) & N(a) & N(b)).
        
        # What about triples not containing u or v?
        # Their neighbor sets don't change. So they remain valid.
        
        for a in range(n):
            if a == u or a == v: continue
            for b in range(n):
                if b == u or b == v or b == a: continue
                
                # Check triple (u, a, b)
                if v in adj[a] and v in adj[b]:
                    common = adj[u] & adj[a] & adj[b]
                    if len(common) >= 2:
                        return False
                
                # Check triple (v, a, b)
                if u in adj[a] and u in adj[b]:
                    common = adj[v] & adj[a] & adj[b]
                    if len(common) >= 2:
                        return False
        return True

    def local_search(adj):
        # Try to add any missing edge
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        if is_k33_free_after_add(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            improved = True
        return adj

    def greedy_random(adj, order):
        for u, v in order:
            if v not in adj[u]:
                if is_k33_free_after_add(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)

    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = [set() for _ in range(n)]
        
        # Random edge order
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        greedy_random(adj, edges)
        
        # Local search to fill gaps
        local_search(adj)
        
        # Count edges
        e_count = sum(len(s) for s in adj) // 2
        if e_count > best_edges:
            best_edges = e_count
            best_graph = [list(s) for s in adj]
            
        # If we hit a known good threshold, we can stop early?
        # ex(20, K3,3) is likely around 55-60.
        if best_edges >= 60:
            break

    # Convert best_graph to list of edges
    if best_graph:
        edges_out = []
        for u in range(n):
            for v in best_graph[u]:
                if u < v:
                    edges_out.append([u, v])
        return edges_out
    else:
        # Fallback to K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]

# Ensure function name matches
# The prompt asks to keep the function names and signatures.
# The current program has `construct`.
