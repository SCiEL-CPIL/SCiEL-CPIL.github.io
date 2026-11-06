import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_c4_free_after_add(adj, u, v):
        # Check if adding (u, v) creates a C4
        # C4 is u-x-y-v-u
        # So we need x in N(u) and y in N(v) such that (x, y) is an edge
        # Note: x != v, y != u is guaranteed if u, v not adjacent yet
        # Also x != y? If x == y, then u-x-v-u is a triangle, not C4.
        # The problem asks for C4-free. Triangles are allowed.
        # So we only care about x != y.
        
        # Optimization: Iterate over smaller neighborhood
        if len(adj[u]) > len(adj[v]):
            u, v = v, u
            
        for x in adj[u]:
            if x == v: continue
            # Check if any neighbor of v is connected to x
            # We can just check if x is connected to any node in adj[v]
            # But we need to ensure the cycle length is 4.
            # u-x-y-v-u.
            # If x is in adj[v], then u-x-v-u is a triangle.
            # We need y in adj[v] such that (x,y) in E.
            # If x is in adj[v], then y=x gives triangle.
            # So we need y != x.
            
            # Simple check:
            for y in adj[v]:
                if y == x:
                    continue
                if y in adj[x]:
                    return False
        return True

    def try_construct(n, min_deg):
        # Random restart loop
        while time.time() < deadline:
            adj = [set() for _ in range(n)]
            success = True
            
            # Randomly shuffle vertices to break symmetry
            order = list(range(n))
            random.shuffle(order)
            
            # We want to ensure min degree.
            # Greedy: always try to add edge to the vertex with currently lowest degree
            # But we need to be careful not to create C4s.
            
            # A better heuristic:
            # 1. Generate a random graph with slightly higher density.
            # 2. Remove edges to break C4s, prioritizing keeping degree high.
            
            # Let's try the "Start dense, sparsify" approach.
            # Target average degree ~ min_deg + 2
            p = (min_deg + 2) / (n - 1)
            
            # Initialize
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
                        
            # Sparsify
            # Find C4s and remove edges
            # To be fast, we can just check all pairs?
            # n=47, pairs ~ 1100.
            # Checking C4 for each pair is O(n^2) or less.
            # Total O(n^4) ~ 47^4 ~ 5M ops. Fast enough.
            
            # However, removing an edge might create new C4s? No, removing edges cannot create cycles.
            # So we just need to remove enough edges to destroy all C4s.
            
            # We want to keep degrees high.
            # So we should remove edges from high-degree vertices if possible,
            # or rather, we should avoid removing edges from low-degree vertices.
            
            # Strategy:
            # While there is a C4:
            #   Find a C4.
            #   Pick an edge (u,v) in the C4.
            #   Prefer (u,v) where min(deg(u), deg(v)) is maximized?
            #   Or prefer (u,v) where deg(u) > min_deg and deg(v) > min_deg.
            #   If we remove (u,v), deg(u) and deg(v) drop.
            #   If they drop below min_deg, it's bad.
            
            # Let's use a simple heuristic:
            # While True:
            #   Find all C4s.
            #   If none, break.
            #   For each C4, for each edge in it:
            #      If deg(u) > min_deg and deg(v) > min_deg:
            #          Remove (u,v)
            #          Break out of loops
            #   If no edge removed, fail this restart.
            
            # Finding all C4s is expensive if done naively.
            # We can just check if graph is C4-free.
            # If not, find one.
            
            while True:
                # Find a C4
                c4_found = None
                for i in range(n):
                    if time.time() > deadline:
                        return None
                    for j in adj[i]:
                        if j > i: continue
                        # Common neighbors of i and j form triangles, not C4s directly.
                        # C4: i-a-b-j-i
                        # We need a in N(i), b in N(j), a != b, (a,b) in E.
                        # And a != j, b != i.
                        
                        # To find one C4 efficiently:
                        # Iterate a in N(i)
                        # Iterate b in N(j)
                        # If a != b and (a,b) in E, then i-a-b-j-i is a C4.
                        
                        # Optimization:
                        # If len(N(i)) * len(N(j)) is small, do it.
                        # n=47, deg ~ 10. 100 checks.
                        
                        for a in adj[i]:
                            if a == j: continue
                            for b in adj[j]:
                                if b == i or b == a: continue
                                if b in adj[a]:
                                    c4_found = (i, a, b, j)
                                    break
                            if c4_found: break
                        if c4_found: break
                    if c4_found: break
                
                if not c4_found:
                    # Check min degree
                    min_d = min(len(adj[i]) for i in range(n))
                    if min_d >= min_deg:
                        # Success
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        # C4-free but degree too low. Fail.
                        break
                
                # Remove an edge from c4_found
                # c4 is (u, v, w, x) -> edges (u,v), (v,w), (w,x), (x,u)
                u, v, w, x = c4_found
                candidates = [(u,v), (v,w), (w,x), (x,u)]
                
                # Pick candidate where both degrees are > min_deg
                best_candidate = None
                best_score = -1
                for a, b in candidates:
                    if len(adj[a]) > min_deg and len(adj[b]) > min_deg:
                        # Score: sum of degrees? We want to remove from high degree nodes
                        # to preserve low degree nodes?
                        # Actually, we want to keep degrees >= min_deg.
                        # If we remove from high degree nodes, we have more "slack".
                        score = len(adj[a]) + len(adj[b])
                        if score > best_score:
                            best_score = score
                            best_candidate = (a, b)
                
                if not best_candidate:
                    # Cannot remove any edge without violating min degree constraint
                    # in this restart.
                    break
                
                a, b = best_candidate
                adj[a].remove(b)
                adj[b].remove(a)
                
            # End while True (C4 removal)
            
    # Try n=47
    result = try_construct(47, 8)
    if result:
        return result
        
    # Try n=46
    result = try_construct(46, 7)
    if result:
        return result
        
    # Fallback to seed
    return {'n': 39, 'edges': []}
