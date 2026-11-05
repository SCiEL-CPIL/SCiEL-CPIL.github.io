import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    TIME_LIMIT = 10.0
    
    # Fallback: 8 disjoint K4s (n=32)
    def get_fallback():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        return {'n': n, 'edges': edges}

    def check_c5_free(n, adj):
        # adj is list of sets
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                # BFS from u to v, depth 4
                # Actually, just check if there is a path of length 4
                # We can do a BFS from u up to depth 4, and see if v is reached.
                # But we must ensure the path doesn't use (u,v).
                # Since we check dist(u,v) == 4, and (u,v) is an edge, dist is 1.
                # Wait, if (u,v) is an edge, dist is 1.
                # A C5 is formed if there is a path of length 4 between u and v.
                # This path cannot use the edge (u,v).
                # So we need to check if there is a path of length 4 in G - {(u,v)}.
                
                # Optimization: 
                # If we just run BFS on the full graph, dist(u,v) will be 1.
                # We need to ignore the edge (u,v).
                
                # Let's do a DFS/BFS specifically for path length 4.
                # Or simpler: 
                # For each edge (u,v), check if there exist distinct a,b,c such that 
                # (u,a), (a,b), (b,c), (c,v) are edges.
                # This is O(E * n^2) which is too slow for n=33?
                # E ~ 33^2/2 = 544. n^2 = 1089. 544 * 1089 ~ 500k. This is fine.
                
                # But we can do better.
                # Precompute squares of adjacency matrix?
                # Let's stick to the iterative check for now, but optimize.
                
                pass # Placeholder, will implement in loop

    def has_c5(n, adj):
        # adj: list of sets
        # Check all edges (u,v)
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                # Check if there is a path u-a-b-c-v
                # a must be neighbor of u, not v
                # c must be neighbor of v, not u
                # b must be neighbor of a and c
                # a, b, c distinct, a != c, a != v, c != u
                
                # Iterate a in adj[u] \ {v}
                # Iterate c in adj[v] \ {u}
                # Check if adj[a] & (adj[c] \ {u, v, a, c}) is not empty?
                # b must be in adj[a] AND adj[c].
                # b must not be u, v, a, c.
                
                # This is O(deg(u) * deg(v) * (deg(a)+deg(c)))?
                # We can precompute intersection of adj[a] and adj[c]?
                
                # Let's try a simpler check:
                # For each u, for each v in adj[u]:
                #   candidates_b = set()
                #   for a in adj[u] - {v}:
                #       candidates_b |= adj[a]
                #   for c in adj[v] - {u}:
                #       if c in candidates_b and c != u and c != v:
                #           # Found path u-a-b-c-v? 
                #           # Need to ensure a, b, c distinct and path valid.
                #           # Actually, if c is in adj[a] for some a, then u-a-c is path length 2.
                #           # We need u-a-b-c-v.
                #           # So b must be in adj[a] and adj[c].
                #           # So we need to check if adj[a] & adj[c] contains a vertex b 
                #           # such that b not in {u, v, a, c}.
                
                # This is getting complicated to inline.
                # Let's use a standard C5 detection algorithm.
                # For n=33, we can even do:
                # For all 5-tuples? 33^5 is too big.
                
                # Let's use the property: 
                # A graph is C5-free iff for every edge (u,v), the distance between u and v 
                # in G - {(u,v)} is not 4.
                
                # We can compute this by running BFS from u in G - {(u,v)} and checking if v is at depth 4.
                # But doing this for every edge is expensive.
                
                # Alternative: 
                # Iterate all pairs (u, v) that are NOT edges.
                # If dist(u, v) == 4, does it imply C5? No.
                # If (u,v) is an edge and there is a path of length 4, it's a C5.
                
                # Let's just do the BFS for each edge.
                # n=33, E ~ 100-500.
                # BFS is O(n+E).
                # Total O(E * (n+E)).
                # 500 * 500 = 250k. Very fast.
                
                # Implementation:
                for u in range(n):
                    for v in adj[u]:
                        if v <= u: continue
                        # BFS from u to v, ignoring edge (u,v)
                        visited = {u: 0}
                        queue = [(u, 0)]
                        found = False
                        while queue:
                            curr, d = queue.pop(0)
                            if d > 4: continue
                            for nb in adj[curr]:
                                if nb == v and curr == u: continue # Ignore edge (u,v)
                                if nb == u and curr == v: continue # Ignore edge (v,u)
                                if nb not in visited:
                                    visited[nb] = d + 1
                                    if nb == v:
                                        if d + 1 == 4:
                                            return True
                                        # If d+1 < 4, it's a smaller cycle, not C5
                                        # But we only care if C5 exists.
                                        # If dist < 4, no C5 via this path?
                                        # Wait, if there is a path of length 2, and edge (u,v), it's C3.
                                        # If path length 3, C4.
                                        # If path length 4, C5.
                                        # If path length 5, C6.
                                        # So we only return True if we find v at depth 4.
                                        # If we find v at depth 2 or 3, we continue searching for depth 4?
                                        # No, BFS finds shortest path.
                                        # If shortest path is 2, there is no path of length 4 that is "simple" in the sense of being the shortest?
                                        # Actually, if there is a path of length 2, there might ALSO be a path of length 4.
                                        # Example: u-a-v and u-b-c-d-v.
                                        # If u-a-v exists, dist is 2.
                                        # But u-b-c-d-v is a C5 if u-v is an edge.
                                        # So BFS stopping at first visit is NOT sufficient.
                                        # We need to check if there is ANY path of length 4.
                                        
                                        # So we must explore up to depth 4.
                                        pass
                        # If we finished BFS without finding v at depth 4, this edge doesn't form a C5.
                        # But the above logic is flawed because BFS marks visited.
                        # We need to allow revisiting? No, simple path.
                        # So we need to check all simple paths of length 4.
                        # This is harder.
                        
                return False

    # This C5 check is tricky to implement correctly and fast in a few lines.
    # Let's use a different approach: 
    # We know the fallback is valid.
    # We can try to construct a valid graph on n=33 using a known method or random search with a robust C5 checker.
    
    # Robust C5 Checker:
    # For each edge (u,v), check if there is a path of length 4.
    # We can do this by:
    # 1. Remove edge (u,v).
    # 2. Check if dist(u,v) == 4.
    # But dist is shortest path.
    # If there is a path of length 4, dist <= 4.
    # If dist == 4, then yes.
    # If dist < 4, there might still be a path of length 4.
    # Example: u-a-v (len 2) and u-b-c-d-v (len 4).
    # Dist is 2. But C5 exists.
    # So we need to check if there is a path of length 4.
    
    # Algorithm for "is there a path of length k between u and v":
    # DP or BFS with depth tracking?
    # Since k=4 is small, we can do:
    # For each u, v:
    #   For each a in N(u):
    #     For each b in N(a):
    #       If b == v: continue
    #       For each c in N(b):
    #         If c == v: 
    #            if a != v and b != v and c != v and a != c: # Simple path?
    #               return True
    #         # Actually, we need u-a-b-c-v.
    #         # So c must be neighbor of v.
    #         # So:
    #   For each a in N(u) \ {v}:
    #     For each b in N(a) \ {u, v}:
    #       For each c in N(b) \ {u, v, a}:
    #         If c in N(v) and c != a:
    #            # Check if path u-a-b-c-v is simple.
    #            # u, a, b, c, v distinct.
    #            # a != v, b != u, b != v, c != u, c != v, c != a.
    #            # b != a.
    #            return True
    #   return False
    
    # This is O(deg(u) * deg(a) * deg(b)).
    # Average degree d. O(d^3).
    # If d=10, 1000 per edge. 500 edges -> 500k. OK.

    def has_c5_fast(n, adj):
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                # Check for path u-a-b-c-v
                for a in adj[u]:
                    if a == v: continue
                    for b in adj[a]:
                        if b == u or b == v: continue
                        for c in adj[b]:
                            if c == u or c == v or c == a: continue
                            if c in adj[v]:
                                # Path u-a-b-c-v exists
                                # Check distinctness:
                                # u, a, b, c, v
                                # a != v (checked)
                                # b != u, v (checked)
                                # c != u, v, a (checked)
                                # a != b (implied by b in adj[a] and simple graph? No, a-b is edge, so a!=b)
                                # So yes, simple path.
                                return True
        return False

    def alpha_less_than_9(n, adj):
        # Branch and bound for max independent set
        # Or just check if there is an independent set of size 9.
        # Greedy + Backtracking
        
        # Order vertices by degree (descending)
        # This is a heuristic for speed.
        
        # Simple recursive check
        def find_independent_set(vertices, size):
            if size == 0:
                return True
            if not vertices:
                return False
            if len(vertices) < size:
                return False
            
            # Pick vertex v
            # Heuristic: pick vertex with highest degree in induced subgraph?
            # Or just first one.
            v = next(iter(vertices))
            
            # Option 1: Include v
            remaining = vertices - {v} - adj[v]
            if find_independent_set(remaining, size - 1):
                return True
            
            # Option 2: Exclude v
            remaining = vertices - {v}
            if find_independent_set(remaining, size):
                return True
                
            return False

        return not find_independent_set(set(range(n)), 9)

    # Search for n=33
    best_result = get_fallback()
    
    for attempt in range(100):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        n = 33
        # Initialize with random edges
        # We want a graph that is likely C5-free and has low alpha.
        # Start with empty graph? Alpha is 33. Too high.
        # Start with random graph p=0.5?
        
        adj = [set() for _ in range(n)]
        edges = []
        
        # Random initialization
        p = 0.5
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        # Check validity
        if has_c5_fast(n, adj):
            continue # Restart or repair
            
        if not alpha_less_than_9(n, adj):
            # Try to repair
            # Find independent set of size 9
            # Add edge between two vertices in it
            
            # For simplicity, just restart if invalid
            continue
            
        if not has_c5_fast(n, adj) and alpha_less_than_9(n, adj):
            return {'n': n, 'edges': edges}
            
    return best_result
