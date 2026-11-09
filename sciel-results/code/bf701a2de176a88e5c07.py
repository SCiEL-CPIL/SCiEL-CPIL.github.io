"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin
    
    n = 20
    colors = list(range(20))
    
    # We will attempt to construct a graph by adding edges one by one.
    # We maintain the current set of edges and their colors.
    # We check constraints only for 4-cycles involving the new edge.
    
    # To speed up, we can pre-calculate or maintain a structure.
    # For n=20, checking 4-cycles is feasible if done smartly.
    
    # Strategy: 
    # 1. Start with an empty graph.
    # 2. Generate all possible edges (u, v) with u < v.
    # 3. Shuffle them.
    # 4. For each edge, try to assign a color.
    #    - Try all 20 colors.
    #    - For each color, check if adding this edge with this color creates a non-rainbow 4-cycle.
    #    - A 4-cycle involving (u,v) is u-x-v-y-u.
    #    - We need to check all x, y such that (u,x), (x,v), (v,y), (y,u) exist.
    #    - Actually, simpler: just check all 4-cycles in the current graph + new edge.
    #    - Since we add edges one by one, we only need to check 4-cycles that include the new edge.
    
    edges = []
    edge_color = {} # (u,v) -> c
    adj = {i: [] for i in range(n)} # i -> list of (neighbor, color)
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_possible_edges)
    
    def is_valid_4cycle(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) == 4

    def check_new_edge(u, v, c):
        # Check if adding edge (u,v) with color c violates any 4-cycle constraint.
        # A 4-cycle involving (u,v) is of the form u-x-v-y-u.
        # The edges are (u,x), (x,v), (v,y), (y,u).
        # We need to find all pairs (x, y) such that:
        # 1. (u,x) exists
        # 2. (x,v) exists
        # 3. (v,y) exists
        # 4. (y,u) exists
        # And x != y, x != u, x != v, y != u, y != v.
        
        # Get neighbors of u and v
        # Note: adj stores (neighbor, color)
        
        # For each neighbor x of u:
        for x, cx in adj[u]:
            if x == v: continue
            # Check if (x, v) exists
            # We need to find the color of (x, v)
            # Let's use a lookup dict for O(1)
            key_xv = (min(x, v), max(x, v))
            if key_xv not in edge_color:
                continue
            cxv = edge_color[key_xv]
            
            # Now we have path u-x-v.
            # We need to find y such that (v, y) and (y, u) exist.
            # y must be a common neighbor of v and u (excluding x, u, v).
            # Actually, y just needs to be connected to v and u.
            
            for y, cy in adj[v]:
                if y == u or y == x: continue
                # Check if (y, u) exists
                key_yu = (min(y, u), max(y, u))
                if key_yu not in edge_color:
                    continue
                cyu = edge_color[key_yu]
                
                # We have a 4-cycle: u-x-v-y-u
                # Colors: cx (u-x), cxv (x-v), cy (v-y), cyu (y-u)
                # Wait, the new edge is (u,v) with color c.
                # The cycle is u-x-v-y-u.
                # The edges are:
                # (u, x) : cx
                # (x, v) : cxv
                # (v, y) : cy
                # (y, u) : cyu
                # Is (u,v) part of this cycle?
                # The cycle is u-x-v-y-u. The edges are (u,x), (x,v), (v,y), (y,u).
                # The edge (u,v) is NOT in this cycle.
                # So adding (u,v) does not affect this cycle's validity.
                # 
                # We only need to check cycles that INCLUDE the new edge (u,v).
                # A cycle including (u,v) is u-v-x-y-u.
                # Edges: (u,v) [c], (v,x) [cxv?], (x,y) [?], (y,u) [?].
                # Let's re-evaluate.
                # Cycle: u - v - x - y - u.
                # Edges:
                # 1. (u, v) : c (new)
                # 2. (v, x) : must exist
                # 3. (x, y) : must exist
                # 4. (y, u) : must exist
                
                # So we need to find x, y such that:
                # (v, x) exists
                # (x, y) exists
                # (y, u) exists
                # And x != u, y != v, x != y.
                
                # Let's iterate over neighbors x of v (x != u)
                # and neighbors y of u (y != v)
                # such that (x, y) exists.
                
                pass # This logic is inside the loop below

        # Correct Logic for check_new_edge(u, v, c):
        # Find all 4-cycles containing edge (u,v).
        # Cycle: u - v - x - y - u
        # Edges: (u,v) [c], (v,x) [c_vx], (x,y) [c_xy], (y,u) [c_yu]
        
        # Iterate x in neighbors of v (x != u)
        for x, c_vx in adj[v]:
            if x == u: continue
            # Iterate y in neighbors of u (y != v)
            for y, c_yu in adj[u]:
                if y == v or y == x: continue
                # Check if (x, y) exists
                key_xy = (min(x, y), max(x, y))
                if key_xy not in edge_color: continue
                c_xy = edge_color[key_xy]
                
                # Check rainbow
                if not is_valid_4cycle(c, c_vx, c_xy, c_yu):
                    return False
        
        return True

    for u, v in all_possible_edges:
        if time.time() > DEADLINE:
            break
            
        # Try to find a valid color
        found = False
        # Shuffle colors to add randomness
        shuffled_colors = colors[:]
        random.shuffle(shuffled_colors)
        
        for c in shuffled_colors:
            if check_new_edge(u, v, c):
                # Add edge
                edges.append([u, v, c])
                edge_color[(min(u,v), max(u,v))] = c
                adj[u].append((v, c))
                adj[v].append((u, c))
                found = True
                break
        
        # If no color works, we skip this edge.
        # This results in a maximal graph (with respect to the order).

    return edges

# EVOLVE-BLOCK-END
