import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 110
    best_n = 0
    best_edges = []
    
    # Try n=47 first
    for n in [47, 46]:
        if time.time() > deadline:
            break
            
        max_deg_bar = 38
        
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            # Initialize bar_G as empty graph (so G is complete)
            # G complete has many C4s.
            # We need to add edges to bar_G to break C4s in G.
            # Start with random bar_G with low degree?
            # Or start with empty bar_G and add edges?
            
            # Better: Start with a random bar_G that satisfies degree constraint.
            # We want G to be C4-free.
            # G is C4-free <=> bar_G has no induced 2K2? No, that's not sufficient.
            # Let's just use the local search on bar_G.
            
            bar_adj = [set() for _ in range(n)]
            
            # Initial random bar_G
            # Ensure delta(bar_G) is not too high?
            # We need Delta(bar_G) <= 38.
            # Start with empty bar_G.
            
            # Check G (which is complete) for C4.
            # It has many.
            
            # Loop: Find C4 in G, add edge to bar_G to break it.
            # Stop when G is C4-free.
            
            found_c4 = True
            while found_c4 and time.time() < deadline:
                found_c4 = False
                # Find a C4 in G
                # G edges are (u,v) where (u,v) NOT in bar_adj
                # We need 4 vertices u,v,x,y such that
                # (u,v), (v,x), (x,y), (y,u) are NOT in bar_adj
                # and (u,x), (v,y) can be anything.
                
                # Randomly sample 4 vertices
                # This might be slow.
                # Better: Iterate over all 4-tuples?
                # n=47, 4-tuples ~ 180k.
                # Check each.
                
                for quad in combinations(range(n), 4):
                    u, v, x, y = quad
                    # Check if u-v-x-y-u is a C4 in G
                    # Edges in G: (u,v), (v,x), (x,y), (y,u)
                    if (v not in bar_adj[u] and 
                        x not in bar_adj[v] and 
                        y not in bar_adj[x] and 
                        u not in bar_adj[y]):
                        # Found C4 in G
                        found_c4 = True
                        # Pick an edge to add to bar_G
                        # Randomly pick one of the 4 edges
                        edge_to_add = random.choice([(u,v), (v,x), (x,y), (y,u)])
                        a, b = edge_to_add
                        # Check if adding (a,b) to bar_G violates Delta(bar_G) <= 38
                        if len(bar_adj[a]) < max_deg_bar and len(bar_adj[b]) < max_deg_bar:
                            bar_adj[a].add(b)
                            bar_adj[b].add(a)
                            break # Restart quad search? Or continue?
                            # Restarting is safer to ensure we don't miss new C4s
                        else:
                            # Try another edge
                            # If all fail, we are stuck?
                            # Try all 4
                            added = False
                            for e in [(u,v), (v,x), (x,y), (y,u)]:
                                a, b = e
                                if len(bar_adj[a]) < max_deg_bar and len(bar_adj[b]) < max_deg_bar:
                                    bar_adj[a].add(b)
                                    bar_adj[b].add(a)
                                    added = True
                                    break
                            if not added:
                                # Cannot break this C4 without violating degree
                                # Try to remove an edge from bar_G to make space?
                                # Or just fail this attempt
                                break 
                    if found_c4:
                        break
                if not found_c4:
                    # G is C4-free
                    # Extract edges of G
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if v not in bar_adj[u]:
                                edges.append([u, v])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    break # Next n or attempt
                    
            if best_n == 47:
                break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=1
        return {'n': 1, 'edges': []}
