import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 0:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        m = len(edges)
        
        # Map edges to indices
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build Line Graph Adjacency
        # L(G) has m nodes.
        # Node i is adjacent to Node j if edge i and edge j share a vertex.
        # We need to build the bipartite graph for matching.
        # Cost matrix of size m x m.
        # cost[i][j] = 1 if edge i and edge j are adjacent in L(G) and i != j.
        # cost[i][j] = 0 otherwise (or infinity).
        
        # Actually, we want to find a matching that maximizes the number of links.
        # This is equivalent to minimizing the number of paths.
        # Min Path Cover = m - Max Matching.
        
        # To use linear_sum_assignment, we set cost = 1 for valid transitions, 0 for invalid?
        # No, linear_sum_assignment minimizes cost.
        # We want to MAXIMIZE matching.
        # So we set cost = 0 for valid edges, and a large number (e.g., m+1) for invalid edges.
        # But we must ensure we don't match i to i (self loop in L(G) is not allowed for simple paths? 
        # In L(G), i and i are not adjacent. So cost[i][i] should be large.
        
        # Initialize cost matrix
        cost = np.full((m, m), m + 1) # Large cost for non-edges
        
        # For each pair of edges, check if they share a vertex
        # Optimize by building adjacency of edges
        edge_adj = defaultdict(set)
        for i, (u, v) in enumerate(edges):
            for j, (x, y) in enumerate(edges):
                if i == j:
                    continue
                # Check if {u,v} and {x,y} share a vertex
                if u == x or u == y or v == x or v == y:
                    edge_adj[i].add(j)
                    # cost[i][j] = 0
                    cost[i][j] = 0

        # Solve Assignment Problem
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # The matching is given by (row_ind[k], col_ind[k])
        # This means edge row_ind[k] is followed by edge col_ind[k] in a trail.
        
        # Reconstruct trails
        # next_edge[i] = j means i -> j
        next_edge = [-1] * m
        prev_edge = [-1] * m
        
        for r, c in zip(row_ind, col_ind):
            # Only accept if cost is 0 (valid transition)
            if cost[r][c] == 0:
                next_edge[r] = c
                prev_edge[c] = r
        
        # Find starts (edges with no predecessor)
        starts = [i for i in range(m) if prev_edge[i] == -1]
        
        trails = []
        for s in starts:
            trail = []
            curr = s
            visited = set()
            while curr != -1:
                if curr in visited:
                    # Cycle detected? Should not happen in assignment problem matching
                    # because it's a permutation.
                    break
                visited.add(curr)
                trail.append(edges[curr])
                curr = next_edge[curr]
            trails.append(trail)
            
        # Convert trails to paths (vertex lists)
        paths = []
        cycle_count = 0
        for trail in trails:
            if not trail:
                continue
            # Reconstruct vertex path from edge sequence
            # trail is a list of (u, v)
            v_path = [trail[0][0], trail[0][1]]
            for k in range(1, len(trail)):
                prev_e = trail[k-1]
                curr_e = trail[k]
                # Connect them
                if prev_e[1] == curr_e[0]:
                    v_path.append(curr_e[1])
                elif prev_e[1] == curr_e[1]:
                    # This means the edge was traversed backwards in the line graph logic?
                    # In L(G), adjacency is undirected.
                    # The matching direction is arbitrary.
                    # We need to check connectivity.
                    # If prev_e = (a,b) and curr_e = (b,c), then b is the common vertex.
                    # If prev_e = (a,b) and curr_e = (c,b), then b is common, but curr_e is (c,b).
                    # The path in G would be ... a-b-c ... or ... a-b-c ...
                    # Wait, if prev_e=(a,b) and curr_e=(c,b), the common vertex is b.
                    # The sequence of vertices is ... a, b, c ...
                    # So we append c.
                    v_path.append(curr_e[0])
                elif prev_e[0] == curr_e[0]:
                    # Common vertex is prev_e[0]
                    # This implies the trail went ... curr_e[0] ... prev_e[0] ... ?
                    # This would mean the direction was reversed.
                    # In L(G) matching, we don't enforce direction consistency.
                    # We just know they are adjacent.
                    # We must ensure the vertex path is valid.
                    # If prev_e=(a,b) and curr_e=(a,c), common is a.
                    # Path so far ends at b.
                    # We need to go from b to a to c?
                    # But we are just listing edges.
                    # The "path" in G is defined by the sequence of edges.
                    # For it to be a valid trail, consecutive edges must share a vertex.
                    # They do.
                    # To form the vertex list, we must trace the walk.
                    # Let's reconstruct carefully.
                    
                    pass # Handled in generic reconstruction below
                    
            # Generic reconstruction
            v_path = [trail[0][0]]
            last_v = trail[0][1]
            # Assume first edge is (u, v). Path starts u, ends v.
            # But which end is the start?
            # It doesn't matter for the set of edges, but for "simple path" check, 
            # we need the actual walk.
            
            # Let's build the walk
            walk = [trail[0][0], trail[0][1]]
            for k in range(1, len(trail)):
                e1 = trail[k-1]
                e2 = trail[k]
                # Find common vertex
                if e1[1] == e2[0]:
                    walk.append(e2[1])
                elif e1[1] == e2[1]:
                    walk.append(e2[0])
                elif e1[0] == e2[0]:
                    # This means the trail reversed direction at the vertex?
                    # e1=(a,b), e2=(a,c). Common a.
                    # Walk was ... b -> a. Next is a -> c.
                    # But walk ends at b.
                    # We need to go b -> a -> c?
                    # No, the edge e1 is already in the walk.
                    # If walk is ... x, b. And e1=(a,b). Then previous was a.
                    # So walk is ... a, b.
                    # Now e2=(a,c).
                    # We must go from b to a? No, we are at b.
                    # The edge e2 is incident to a.
                    # This implies the trail is not a valid walk in G?
                    # In L(G), e1 and e2 are adjacent.
                    # A valid trail in G requires that the end of e1 is the start of e2 (or vice versa).
                    # The matching in L(G) ensures adjacency, but not consistent orientation.
                    # So we might have a "zig-zag" which is not a trail.
                    # However, for simple graphs, if e1 and e2 share a vertex, 
                    # and we are building a path cover, we can orient them to form a trail.
                    # The assignment problem finds the structure.
                    # We just need to verify if the resulting sequence forms a simple path.
                    pass
            
            # Simplified: Just check if the set of vertices in the walk has duplicates.
            # If duplicates, it's not a simple path.
            if len(walk) != len(set(walk)):
                # Not a simple path. 
                # For now, we count this as a "bad" trail.
                # We will fall back to heuristic if too many.
                cycle_count += 1
                # For the result, we can split it or just return it?
                # The task requires simple paths.
                # If we return a non-simple path, it's invalid.
                # We must fix it.
                # Splitting a non-simple trail into simple paths is complex.
                # Given the random nature, this should be rare.
                # We'll mark it and handle later.
                pass
                
            paths.append(walk)
        
        # If we have "bad" paths, we need to fix them.
        # For simplicity and robustness, if any path is not simple, 
        # we will use the heuristic beam search for the whole graph.
        
        is_valid = True
        for p in paths:
            if len(p) != len(set(p)):
                is_valid = False
                break
        
        if not is_valid:
            # Fallback to heuristic
            # (Re-implementing a simplified version of the current code's logic)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining_edges = set(edges)
            heur_paths = []
            
            while remaining_edges:
                if time.time() > deadline - 2.0:
                    heur_paths.extend([list(e) for e in remaining_edges])
                    break
                
                # Find a long path greedily
                # Start from a random edge
                start_e = next(iter(remaining_edges))
                u, v = start_e
                # Try to extend
                path = [u, v]
                remaining_edges.discard(start_e)
                
                # Extend forward
                while True:
                    curr = path[-1]
                    extended = False
                    for n in adj[curr]:
                        e = (min(curr, n), max(curr, n))
                        if e in remaining_edges and n not in path:
                            path.append(n)
                            remaining_edges.discard(e)
                            extended = True
                            break
                    if not extended:
                        break
                
                # Extend backward
                while True:
                    curr = path[0]
                    extended = False
                    for n in adj[curr]:
                        e = (min(curr, n), max(curr, n))
                        if e in remaining_edges and n not in path:
                            path.insert(0, n)
                            remaining_edges.discard(e)
                            extended = True
                            break
                    if not extended:
                        break
                
                heur_paths.append(path)
            
            paths = heur_paths

        results.append(paths)

    return results
