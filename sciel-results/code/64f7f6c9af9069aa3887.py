import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_n = 39
    best_edges = []
    
    for n in range(46, 38, -1):
        if time.time() > deadline:
            break
            
        target_min_deg = n - 39
        
        # Try multiple restarts for each n
        found = False
        attempts = 0
        while not found and time.time() < deadline:
            attempts += 1
            # Initialize random graph
            # Start with p=0.5
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = adj[j][i] = True
            
            # Remove C4s
            # We need a fast C4 detection/removal
            # Iterate until no C4s
            max_iter = 1000
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                c4_found = False
                # Find a C4
                # Iterate over pairs of edges (u,v) and (w,x) that share a vertex?
                # Or just check all 4-tuples? n=46, 4-tuples is ~46^4/24 ~ 1.5M. Too slow in Python if done naively every step.
                
                # Better: Iterate over pairs of vertices (u,v) that are NOT connected.
                # If we add (u,v), does it create a C4?
                # This is the "Add" check.
                # For "Remove", we can find a C4 by checking common neighbors.
                # A C4 is u-w-x-v-u.
                # So w and x are common neighbors of u and v? No.
                # u and v are opposite vertices. w and x are opposite.
                # (u,w), (w,x), (x,v), (v,u) ? No, (v,u) is not an edge in C4 definition usually, but in cycle it is.
                # In a simple graph C4, edges are (u,w), (w,x), (x,v), (v,u).
                # So u and v are connected. w and x are connected.
                # And u,x not connected, w,v not connected.
                
                # Let's find a C4 by iterating u < v.
                # For each pair (u,v) with edge, check if there is w < x such that
                # (u,w), (w,x), (x,v) exist and (u,x) and (w,v) do not exist?
                # Actually, any 4-cycle is enough.
                
                # Simpler C4 finding:
                # For each pair of vertices (u, v) that are adjacent:
                #   For each common neighbor w of u and v? No, that's a triangle.
                #   For C4, we need u-w-x-v.
                #   So w is neighbor of u, x is neighbor of v.
                #   And (w,x) is an edge.
                #   And w != v, x != u.
                
                c4_edges = None
                for u in range(n):
                    for v in range(u+1, n):
                        if not adj[u][v]:
                            continue
                        # Find w in N(u), x in N(v) such that (w,x) is edge
                        # and w != v, x != u
                        # To avoid duplicates, we can just find one.
                        for w in range(n):
                            if w == u or w == v: continue
                            if not adj[u][w]: continue
                            for x in range(n):
                                if x == u or x == v: continue
                                if x == w: continue
                                if not adj[v][x]: continue
                                if adj[w][x]:
                                    # Found C4: u-w-x-v-u
                                    # Edges: (u,w), (w,x), (x,v), (v,u)
                                    c4_edges = [(u,w), (w,x), (x,v), (v,u)]
                                    break
                            if c4_edges:
                                break
                        if c4_edges:
                            break
                    if c4_edges:
                        break
                
                if not c4_edges:
                    break
                
                # Remove one edge from c4_edges
                # Strategy: Remove the edge between the two highest degree vertices
                # to preserve degrees of lower degree vertices
                degs = [sum(adj[i]) for i in range(n)]
                # Sort edges by sum of degrees of endpoints, descending
                # We want to remove the edge with highest endpoint degrees
                c4_edges.sort(key=lambda e: degs[e[0]] + degs[e[1]], reverse=True)
                e_to_remove = c4_edges[0]
                adj[e_to_remove[0]][e_to_remove[1]] = False
                adj[e_to_remove[1]][e_to_remove[0]] = False
            
            # Check min degree
            degs = [sum(adj[i]) for i in range(n)]
            min_deg = min(degs)
            
            if min_deg >= target_min_deg:
                # Success
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # If min degree is too low, try to add edges
            # This is complex to implement efficiently in the loop.
            # Let's just restart if it fails.
            # Or, we can try a "repair" phase.
            # For now, just restart.
            
            # If attempts > 100, maybe lower p or change strategy?
            # For n=46, p=0.5 might be too dense or too sparse?
            # Let's try different p values in restarts.
            # But simple restarts first.

    # If we didn't find 46 or 47, return best known or 39
    # Actually, we should return the best n we found.
    # But the loop breaks on success.
    # If it doesn't succeed, we return 39.
    
    return {'n': 39, 'edges': []}
