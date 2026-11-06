import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_edges = []
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_c4_free(adj, u, v):
        # Check if adding (u, v) creates a 4-cycle
        # A 4-cycle is u-v-w-x-u
        # We need to check if there exist w, x such that (v,w), (w,x), (x,u) are in adj
        # And w != x, w != u, w != v, x != u, x != v
        # Also need to ensure we don't count triangles or other shapes
        # Actually, if (v,w) and (w,x) and (x,u) exist, and (u,v) is being added,
        # we have a cycle u-v-w-x-u.
        # We must ensure u != w, u != x, v != w, v != x, w != x.
        
        for w in adj[v]:
            if w == u: continue
            for x in adj[w]:
                if x == u or x == v: continue
                if x in adj[u]:
                    # Found a 4-cycle u-v-w-x-u
                    # Wait, is it possible that w=x? No, simple graph.
                    # Is it possible that this is a triangle?
                    # If w=x, then u-v-w-u is a triangle. But we check x in adj[u].
                    # If w=x, then (v,w) and (w,u) exist. Adding (u,v) makes a triangle.
                    # Triangles are not 4-cycles.
                    # So we must ensure the cycle is length 4.
                    # The path v-w-x-u has length 3.
                    # If w=x, path is v-w-u, length 2.
                    # So we must check w != x.
                    if w != x:
                        return False
        return True

    for _ in range(10000): # Many restarts
        if time.time() > deadline: break
        
        random.shuffle(all_edges)
        adj = [set() for _ in range(n)]
        edges = []
        
        for u, v in all_edges:
            if is_c4_free(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
        
        if len(edges) > len(best_edges):
            best_edges = edges
            # Optional: Early stop if we reach a very high number?
            # ex(20, C4) is likely < 100. If we get > 90, it's good.

    # Convert best_edges to required format
    # Color can be anything, e.g., 0
    return [[u, v, 0] for u, v in best_edges]
