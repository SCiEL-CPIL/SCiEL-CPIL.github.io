"""Rewritten program for strong edge colouring optimization."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate graph
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, time_limit):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict sets for edges
        # Two edges (u,v) and (x,y) conflict if:
        # 1. They share a vertex (u==x, u==y, v==x, v==y)
        # 2. They are joined by an edge (u==x and v in adj[x], etc.)
        # Actually, for induced matching, edges e1=(u1,v1) and e2=(u2,v2) are incompatible if:
        # - They share a vertex
        # - There is an edge between any vertex of e1 and any vertex of e2
        # Let's precompute for each edge e, the set of edges that conflict with it.
        
        edge_index = {e: i for i, e in enumerate(edges)}
        m = len(edges)
        conflicts = [set() for _ in range(m)]
        
        for i, e1 in enumerate(edges):
            u1, v1 = e1
            # Check all other edges
            for j, e2 in enumerate(edges):
                if i == j: continue
                u2, v2 = e2
                # Check shared vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
                # Check if joined by an edge
                # u1 connected to u2 or v2?
                if u2 in adj[u1] or v2 in adj[u1]:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
                # u1 connected to v2 or v1? (already checked u1-u2, u1-v2)
                # v1 connected to u2 or v2?
                if u2 in adj[v1] or v2 in adj[v1]:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
        
        # Greedy initialization
        # Sort edges by degree or just use given order
        # We want to assign colours.
        # colour[i] is the colour index for edge i.
        # colours is a list of sets of edge indices for each colour.
        
        def greedy_coloring(order):
            col_of_edge = [-1] * m
            colours = [] # list of sets
            for i in order:
                placed = False
                # Try to place in existing colours
                for c_idx, c_set in enumerate(colours):
                    # Check if i conflicts with any edge in c_set
                    ok = True
                    for j in c_set:
                        if j in conflicts[i]:
                            ok = False
                            break
                    if ok:
                        col_of_edge[i] = c_idx
                        c_set.add(i)
                        placed = True
                        break
                if not placed:
                    new_c = len(colours)
                    col_of_edge[i] = new_c
                    colours.append({i})
            return col_of_edge, colours

        # Initial order: maybe by degree?
        degs = [len(conflicts[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: -degs[x])
        col_of_edge, colours = greedy_coloring(order)
        best_num_colours = len(colours)
        best_col_of_edge = col_of_edge[:]
        best_colours = [s.copy() for s in colours]
        
        # Local Search
        # Operations:
        # 1. Move an edge from colour A to colour B if possible.
        # 2. Merge two colours if possible.
        
        rng = random.Random(42)
        
        def can_place(edge_idx, colour_set, col_of_edge, colours):
            # Check if edge_idx can be added to colour_set
            # It must not conflict with any edge in colour_set
            for j in colour_set:
                if j in conflicts[edge_idx]:
                    return False
            return True

        def try_move(edge_idx, target_colour_idx, col_of_edge, colours):
            current_colour_idx = col_of_edge[edge_idx]
            if current_colour_idx == target_colour_idx:
                return False
            
            # Check if we can move edge_idx to target_colour_idx
            # Remove from current
            current_set = colours[current_colour_idx]
            target_set = colours[target_colour_idx]
            
            # Temporarily remove from current
            current_set.remove(edge_idx)
            
            # Check if can add to target
            if can_place(edge_idx, target_set, col_of_edge, colours):
                target_set.add(edge_idx)
                col_of_edge[edge_idx] = target_colour_idx
                
                # Clean up: if current_set is empty, remove it? 
                # For simplicity in this loop, we might leave empty sets or handle carefully.
                # Better to keep indices consistent. Let's not delete colours immediately to avoid index shifting issues.
                # We will clean up at the end of a successful move or periodically.
                return True
            else:
                # Revert
                current_set.add(edge_idx)
                return False

        def try_merge(c1_idx, c2_idx, col_of_edge, colours):
            # Try to merge c2_idx into c1_idx
            if c1_idx == c2_idx:
                return False
            
            c1_set = colours[c1_idx]
            c2_set = colours[c2_idx]
            
            # Check if union is an independent set in conflict graph
            # i.e., no edge in c1_set conflicts with any edge in c2_set
            for e1 in c1_set:
                for e2 in c2_set:
                    if e2 in conflicts[e1]:
                        return False
            
            # Perform merge
            for e in c2_set:
                col_of_edge[e] = c1_idx
            c1_set.update(c2_set)
            c2_set.clear()
            return True

        # Run local search loop
        # We use a simple randomized local search with restarts if stuck.
        # Given time constraints, we will just run moves/merges randomly until time runs out.
        
        while time.time() < time_limit:
            # Randomly pick an operation
            op = rng.choice(['move', 'merge'])
            
            if op == 'move':
                # Pick a random edge
                e_idx = rng.randrange(m)
                current_c = col_of_edge[e_idx]
                # Pick a random target colour
                # Only consider non-empty colours?
                active_colours = [i for i in range(len(colours)) if colours[i]]
                if len(active_colours) <= 1:
                    continue
                target_c = rng.choice(active_colours)
                if target_c != current_c:
                    try_move(e_idx, target_c, col_of_edge, colours)
                    
            elif op == 'merge':
                active_colours = [i for i in range(len(colours)) if colours[i]]
                if len(active_colours) < 2:
                    continue
                c1, c2 = rng.sample(active_colours, 2)
                try_merge(c1, c2, col_of_edge, colours)
            
            # Check for improvement
            current_num = len([c for c in colours if c])
            if current_num < best_num_colours:
                best_num_colours = current_num
                best_col_of_edge = col_of_edge[:]
                best_colours = [s.copy() for s in colours]
                
            # Optional: Periodic cleanup of empty colours to keep list small
            # But indices matter. Let's just leave them.
            
            # If we reach the theoretical lower bound (max degree + 1?), we could stop early?
            # Lower bound for strong chromatic index is at least delta+1.
            # Actually, it's at least delta+1 for any graph? No, strong chromatic index >= delta+1.
            # For d=4, LB=5. For d=5, LB=6. For d=6, LB=7.
            # But we want to minimize.
            pass

        # Return the best solution found
        # We need to return the list of colours in sorted edge order
        # The colours are 0-indexed.
        # We should compress the colour indices to be 0..k-1
        final_colours = best_col_of_edge[:]
        # Map old colour indices to new compact ones
        colour_map = {}
        next_c = 0
        for i in range(m):
            c = final_colours[i]
            if c not in colour_map:
                colour_map[c] = next_c
                next_c += 1
            final_colours[i] = colour_map[c]
        
        return final_colours

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Allocate time roughly proportional to complexity, or just split evenly
    # 115 seconds total. 3 instances. ~38s each.
    time_per_instance = (deadline - time.time()) / 3.0
    
    for spec in specs:
        edges = instance(*spec)
        # Solve
        result = solve_instance(edges, time.time() + time_per_instance)
        out.append(result)
        
    return out
# EVOLVE-BLOCK-END
