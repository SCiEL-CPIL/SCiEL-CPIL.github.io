"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin under 150s kill
    
    n = 20
    num_colors = 20
    
    # Pre-compute all 4-cycles in K_n
    # A 4-cycle is defined by 4 distinct vertices {a,b,c,d} and the cycle order.
    # There are 3 distinct 4-cycles on any set of 4 vertices:
    # 1. a-b-c-d-a (edges: ab, bc, cd, da)
    # 2. a-b-d-c-a (edges: ab, bd, dc, ca)
    # 3. a-c-b-d-a (edges: ac, cb, bd, da)
    
    # We need to efficiently check if a set of 4 edges forms a rainbow cycle.
    # Strategy: 
    # 1. Represent the graph as a dictionary of edges: (u,v) -> color
    # 2. For each move, we need to check if the move creates a violation.
    #    A violation occurs if there is a 4-cycle where two edges have the same color.
    #    
    # Instead of checking ALL 4-cycles every time (expensive), we can check locally.
    # When adding edge (u,v) with color c:
    #    It participates in 4-cycles of the form u-x-v-y-u.
    #    For any x, y such that (u,x), (x,v), (v,y), (y,u) exist, we have a 4-cycle.
    #    We must ensure c is distinct from colors of (u,x), (x,v), (v,y), (y,u).
    #    
    # This local check is feasible.
    
    # Pre-compute potential 4-cycle structures?
    # Let's use a simple Simulated Annealing approach.
    # State: 
    #   edges: set of (u,v)
    #   colors: dict (u,v) -> c
    
    # Objective: Maximize |E| - lambda * |Violations|
    # We want to maximize |E|, so we prioritize adding edges.
    # We define a score function: score = |E| - 10 * |Violations|
    # If Violations == 0, score = |E|.
    
    # Initialize with a random sparse graph
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # We will use a greedy construction first to get a good starting point,
    # then refine with local search.
    
    # Greedy Phase:
    # Try to add edges in a random order, assigning a color that doesn't create an immediate violation.
    # If no color works, skip the edge.
    
    random.seed(42)
    edge_order = all_possible_edges[:]
    random.shuffle(edge_order)
    
    current_edges = set()
    current_colors = {}
    
    # Helper to get color of edge (u,v)
    def get_color(u, v):
        if u > v: u, v = v, u
        return current_colors.get((u, v), None)
    
    # Helper to check if adding (u,v) with color c creates a violation
    # We only need to check 4-cycles containing (u,v).
    # A 4-cycle containing (u,v) is u-x-v-y-u.
    # Edges: (u,v), (v,x), (x,y), (y,u)
    # Wait, the cycle is u-v-x-y-u? No.
    # Cycle: u - v - x - y - u.
    # Edges: (u,v), (v,x), (x,y), (y,u).
    # We are adding (u,v). The other edges (v,x), (x,y), (y,u) must exist.
    # The colors must be distinct.
    
    def is_valid_addition(u, v, c):
        # Check all 4-cycles involving edge (u,v)
        # Form: u - v - x - y - u
        # x and y are distinct from u, v and each other.
        # Edges required: (v,x), (x,y), (y,u)
        
        # Iterate over possible x
        for x in range(n):
            if x == u or x == v: continue
            if (v, x) not in current_edges and (x, v) not in current_edges:
                # Normalize key for lookup
                key_vx = (v, x) if v < x else (x, v)
                if key_vx not in current_edges:
                    continue
            
            # Iterate over possible y
            for y in range(n):
                if y == u or y == v or y == x: continue
                
                key_yu = (y, u) if y < u else (u, y)
                if key_yu not in current_edges:
                    continue
                
                key_xy = (x, y) if x < y else (y, x)
                if key_xy not in current_edges:
                    continue
                
                # We have a 4-cycle: u-v-x-y-u
                # Edges: (u,v) [new, color c], (v,x), (x,y), (y,u)
                # Get colors of existing edges
                c_vx = get_color(v, x)
                c_xy = get_color(x, y)
                c_yu = get_color(y, u)
                
                # All must be distinct and distinct from c
                if c == c_vx or c == c_xy or c == c_yu:
                    return False
                if c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                    # This cycle was already invalid before we added (u,v)?
                    # If so, the graph already has a violation. 
                    # In our greedy phase, we assume current graph is valid.
                    # So if current graph is valid, these colors are already distinct.
                    pass
        
        return True

    # Greedy Construction
    for u, v in edge_order:
        if time.time() - t0 > DEADLINE - 10:
            break
        # Try to find a valid color
        valid_color = None
        # Shuffle colors to break symmetry
        colors_list = list(range(num_colors))
        random.shuffle(colors_list)
        for c in colors_list:
            if is_valid_addition(u, v, c):
                valid_color = c
                break
        
        if valid_color is not None:
            current_edges.add((u, v) if u < v else (v, u))
            current_colors[(u, v) if u < v else (v, u)] = valid_color
    
    # Local Search / Simulated Annealing Phase
    # We try to improve the number of edges.
    # Moves:
    # 1. Try to add a missing edge with some color.
    # 2. If that fails, try to recolor existing edges to make space.
    # 3. If that fails, remove a "blocking" edge and try again.
    
    # For simplicity and speed, we will do a hill-climbing on the number of edges.
    # We iterate through missing edges and try to add them.
    # If we can't add one directly, we might try to recolor its neighbors.
    
    # To be efficient, we limit the number of iterations.
    iterations = 0
    max_iterations = 5000
    
    # List of missing edges
    missing_edges = [e for e in all_possible_edges if e not in current_edges]
    random.shuffle(missing_edges)
    
    # We'll use a simple restart strategy if stuck
    best_edge_count = len(current_edges)
    
    # Helper to check if a specific edge (u,v) can be added with color c
    # This is the same as is_valid_addition but we need to be careful about 
    # existing violations if we allow them (we don't in this greedy phase).
    
    # Let's do a more aggressive local search:
    # 1. Pick a random missing edge.
    # 2. Try all colors.
    # 3. If none work, pick a random existing edge that is part of a conflicting 4-cycle 
    #    and remove it, then retry.
    
    # To speed up, we pre-compute for each edge, which other edges form 4-cycles with it?
    # That's complex. Let's stick to the direct check but optimize the inner loop.
    
    # Optimization: 
    # The check `is_valid_addition` is O(n^2) per edge addition attempt.
    # With 190 edges, and 20 colors, that's 190 * 20 * 400 = 1.5M operations per full pass.
    # This is fast enough in Python if we do a few passes.
    
    passes = 0
    while time.time() - t0 < DEADLINE - 5 and passes < 20:
        passes += 1
        progress = False
        random.shuffle(missing_edges)
        for u, v in missing_edges:
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            
            found = False
            colors_list = list(range(num_colors))
            random.shuffle(colors_list)
            for c in colors_list:
                if is_valid_addition(u, v, c):
                    current_edges.add((u, v) if u < v else (v, u))
                    current_colors[(u, v) if u < v else (v, u)] = c
                    missing_edges.remove((u, v))
                    progress = True
                    found = True
                    break
            
            if found:
                # If we added an edge, we might have invalidated some other missing edges?
                # No, adding an edge only creates more constraints, so other missing edges
                # that were valid might become invalid. But we re-check them in the next pass.
                # The current edge is now in current_edges.
                continue
        
        if not progress:
            # No more edges can be added directly.
            # Try to remove an edge to unlock others?
            # This is risky and complex. For now, if no progress, break or try recoloring.
            
            # Let's try a simple recoloring heuristic:
            # Find an edge that is "blocking" many additions.
            # This is hard to measure. 
            # Alternative: Random restart of the greedy phase with different seeds?
            # We already did that. 
            # Let's try to remove a random edge and see if we can add more.
            
            if current_edges:
                # Pick a random edge to remove
                rem_u, rem_v = random.choice(list(current_edges))
                del current_colors[(rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)]
                current_edges.remove((rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u))
                # Add it back to missing list
                missing_edges.append((rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u))
                # Don't set progress=True to allow further iterations in this pass?
                # Actually, removing an edge reduces the count. We should only do this 
                # if we expect to add more than 1 back.
                # For simplicity, let's just break if no progress in a full pass.
                # But we just removed an edge. Let's try to add it back or others.
                # The loop continues to the next missing edge.
            else:
                break
                
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            # Save best state?
            # We are modifying in place. We should keep a copy of the best.
            best_edges_copy = set(current_edges)
            best_colors_copy = dict(current_colors)

    # If we didn't save best state, use current
    if 'best_edges_copy' not in locals():
        best_edges_copy = set(current_edges)
        best_colors_copy = dict(current_colors)
    else:
        # Restore best state
        current_edges = best_edges_copy
        current_colors = best_colors_copy

    # Convert to output format
    result = []
    for u, v in current_edges:
        c = current_colors[(u, v) if u < v else (v, u)]
        result.append([u, v, c])
    
    return result

# EVOLVE-BLOCK-END
