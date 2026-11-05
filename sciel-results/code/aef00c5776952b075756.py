import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit

    n = 20
    num_colours = 20

    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d and one of 3 orderings.
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    # Total: C(20,4) * 3 = 14535 cycles.
    
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    # Precompute for each edge (u,v) the list of cycles that contain it.
    # This allows us to quickly check constraints when adding an edge.
    # edge_index: (u, v) with u < v -> list of cycle indices
    # Also, for each cycle, we store the 4 edges as tuples.
    
    # Map edge to list of cycle indices
    edge_to_cycles = {}
    for i, cycle in enumerate(cycles):
        for e in cycle:
            u, v = e
            if u > v:
                u, v = v, u
            key = (u, v)
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(i)

    def is_valid(adj):
        """
        Checks if the current graph is valid.
        adj: n x n matrix, -1 for no edge, else colour.
        """
        for cycle in cycles:
            e1, e2, e3, e4 = cycle
            c1 = adj[e1[0], e1[1]]
            c2 = adj[e2[0], e2[1]]
            c3 = adj[e3[0], e3[1]]
            c4 = adj[e4[0], e4[1]]
            
            if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                continue # Not a cycle in the graph
            
            if len({c1, c2, c3, c4}) < 4:
                return False
        return True

    def can_colour(adj, u, v):
        """
        Checks if edge (u,v) can be added with some colour.
        Returns the colour if possible, else None.
        """
        # 1. Proper edge colouring constraint:
        # Colour of (u,v) must be different from colours of all edges incident to u or v.
        forbidden_proper = set()
        for w in range(n):
            if w != u and adj[u, w] != -1:
                forbidden_proper.add(adj[u, w])
            if w != v and adj[w, v] != -1:
                forbidden_proper.add(adj[w, v])
                
        # 2. 4-cycle rainbow constraint:
        # For every cycle containing (u,v), the other 3 edges must have distinct colours,
        # and the new colour must be distinct from them.
        
        forbidden_cycle = set()
        
        # Get list of cycles containing (u,v)
        key = (min(u,v), max(u,v))
        if key not in edge_to_cycles:
            cycle_indices = []
        else:
            cycle_indices = edge_to_cycles[key]
            
        for idx in cycle_indices:
            cycle = cycles[idx]
            # Find the other 3 edges
            other_edges = []
            for e in cycle:
                eu, ev = e
                if (eu == u and ev == v) or (eu == v and ev == u):
                    continue
                other_edges.append(e)
            
            # Check if all 3 other edges exist
            valid_cycle = True
            colours = []
            for e in other_edges:
                c = adj[e[0], e[1]]
                if c == -1:
                    valid_cycle = False
                    break
                colours.append(c)
            
            if not valid_cycle:
                continue
                
            # Check if the 3 existing edges have distinct colours
            if len(set(colours)) < 3:
                # If the existing edges in this cycle already have a conflict,
                # we cannot fix it by choosing a new colour for (u,v).
                # So this edge (u,v) cannot be added.
                return None
            
            # The 3 colours are distinct. The new colour must be different from all 3.
            forbidden_cycle.update(colours)
        
        all_forbidden = forbidden_proper.union(forbidden_cycle)
        
        # Find a colour not in all_forbidden
        for c in range(num_colours):
            if c not in all_forbidden:
                return c
                
        return None

    # Strategy:
    # 1. Start with empty graph.
    # 2. Generate list of all potential edges.
    # 3. Shuffle them.
    # 4. For each edge, try to add it.
    #    - Check if there is a valid colour.
    #    - If yes, add it with that colour.
    #    - If no, skip it.
    # 5. This produces a maximal valid graph (with respect to the order).
    # 6. To maximize density, we can try multiple random orders (restarts).
    
    best_edges = 0
    best_adj = None
    
    # Number of restarts limited by time
    # Each restart: O(E * Cost_Check)
    # E ~ 190. Cost_Check: iterating neighbour cycles. 
    # We can do many restarts.
    
    # Heuristic for order: 
    # Random order is good.
    
    for attempt in range(1000): # Try up to 1000 restarts
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = np.full((n, n), -1, dtype=int)
        
        # Create list of edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                edges.append((u, v))
                
        random.shuffle(edges)
        
        # Greedy Add
        for u, v in edges:
            c = can_colour(adj, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
                
        # Count edges
        edge_count = int(np.sum(adj != -1)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
            
            # If we reached a very high density, we can stop?
            # Max edges in C4-free graph is ~ n^2/4 = 100? 
            # But we have colour constraints.
            # If we get > 80, it's very good.
            if best_edges >= 90:
                # Might be hard to beat, but let's run a few more
                pass

    # If best_adj is None, fallback
    if best_adj is None:
        adj = np.full((n, n), -1, dtype=int)
        # Add a star
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Final verification (optional but good for safety)
    # Check all 4-cycles in best_adj
    # This is O(14535) checks. Fast.
    valid = True
    for cycle in cycles:
        e1, e2, e3, e4 = cycle
        c1 = best_adj[e1[0], e1[1]]
        c2 = best_adj[e2[0], e2[1]]
        c3 = best_adj[e3[0], e3[1]]
        c4 = best_adj[e4[0], e4[1]]
        
        # All must be present (not -1)
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            continue # Not a cycle in the graph
            
        # Check distinctness
        if len({c1, c2, c3, c4}) < 4:
            valid = False
            break
            
    if not valid:
        # This should not happen if can_colour is correct.
        # But if it does, we need to return a valid graph.
        # We can just return the star (which is valid).
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
