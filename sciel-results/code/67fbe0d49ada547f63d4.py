"""Construct a dense edge-coloured graph on 20 vertices with rainbow 4-cycles."""
import time
import itertools
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 20
    num_colors = 20

    # Strategy: Start with a complete graph and try to find a valid coloring.
    # This is NP-hard in general, but for n=20 we can use a heuristic approach.
    
    # Alternative strategy: Build up the graph incrementally.
    # Start with no edges, add edges one by one, choosing colors that don't violate the rainbow 4-cycle constraint.
    
    # For efficiency, let's try a random greedy approach with multiple restarts.
    
    def check_rainbow_4cycles(edges, color_map):
        """Check if all 4-cycles have rainbow colors. color_map is dict (u,v)->color with u<v"""
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for (u, v), c in color_map.items():
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all 4-cycles and check rainbow
        for a in range(n):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[b]:
                    if c <= b or c == a:
                        continue
                    for d in adj[c]:
                        if d <= c or d == b:
                            continue
                        if d in adj[a] and d > a:
                            # Found 4-cycle a-b-c-d-a
                            colors = [color_map[(min(a,b),max(a,b))],
                                      color_map[(min(b,c),max(b,c))],
                                      color_map[(min(c,d),max(c,d))],
                                      color_map[(min(d,a),max(d,a))]]
                            if len(set(colors)) != 4:
                                return False
        return True

    def try_random_construction():
        """Try to build a valid edge-colored graph using random greedy."""
        # Order edges randomly
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        color_map = {}
        for u, v in all_edges:
            # Try each color
            found = False
            colors = list(range(num_colors))
            random.shuffle(colors)
            for c in colors:
                # Tentatively add edge
                color_map[(u, v)] = c
                # Check if this creates any non-rainbow 4-cycle
                # Only need to check 4-cycles that include this new edge
                valid = True
                # A 4-cycle containing edge (u,v) has form u-v-x-y-u or v-u-x-y-v
                # i.e., u-v, v-x, x-y, y-u
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if (min(v,x), max(v,x)) in color_map:
                        for y in range(n):
                            if y == u or y == v or y == x:
                                continue
                            if (min(x,y), max(x,y)) in color_map and (min(y,u), max(y,u)) in color_map:
                                # Check 4-cycle u-v-x-y-u
                                c1 = color_map[(min(u,v), max(u,v))]
                                c2 = color_map[(min(v,x), max(v,x))]
                                c3 = color_map[(min(x,y), max(x,y))]
                                c4 = color_map[(min(y,u), max(y,u))]
                                if len(set([c1, c2, c3, c4])) != 4:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    found = True
                    break
                else:
                    del color_map[(u, v)]
            # If no color works, skip this edge
            
        # Convert to list format
        result = []
        for (u, v), c in color_map.items():
            result.append([u, v, c])
        return result

    # Try multiple random constructions
    best_edges = []
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        edges = try_random_construction()
        if len(edges) > len(best_edges):
            best_edges = edges
            # If we get a very dense graph, we might be done
            if len(edges) > 150:
                break
        attempts += 1
        if attempts % 10 == 0:
            # Periodic check
            pass

    return best_edges
# EVOLVE-BLOCK-END
