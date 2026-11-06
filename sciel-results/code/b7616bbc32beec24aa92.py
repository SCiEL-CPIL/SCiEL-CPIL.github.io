import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    n = 20
    colors = list(range(n))
    
    # State representation:
    # edge_color[u][v] = color or -1 if no edge
    edge_color = [[-1 for _ in range(n)] for _ in range(n)]
    
    def add_edge(u, v, c):
        edge_color[u][v] = c
        edge_color[v][u] = c

    def remove_edge(u, v):
        edge_color[u][v] = -1
        edge_color[v][u] = -1

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if edge_color[u][v] != -1:
                    edges.append([u, v, edge_color[u][v]])
        return edges

    def count_edges():
        return sum(1 for u in range(n) for v in range(u+1, n) if edge_color[u][v] != -1)

    def find_violations():
        """
        Returns a list of (u, a, b, v) where u-a-b-v-u is a 4-cycle 
        and the colors are not all distinct.
        We check for duplicate colors.
        """
        violations = []
        # To speed up, we can just check a sample or all.
        # With n=20, total 4-cycles is manageable if we are smart.
        # A 4-cycle is defined by 4 vertices.
        # Let's iterate over pairs of disjoint edges? No, that's complex.
        # Iterate over u < a < b < v ? No, cycles can be any order.
        # A 4-cycle is a set of 4 vertices {u, a, b, v} such that
        # edges (u,a), (a,b), (b,v), (v,u) exist.
        # Note: (u,v) is an edge in the cycle.
        
        # Optimization: Precompute adjacency lists?
        # Let's just iterate u, a, b, v.
        # u != a, a != b, b != v, v != u.
        # Also need edges to exist.
        
        for u in range(n):
            for a in range(n):
                if a == u: continue
                if edge_color[u][a] == -1: continue
                for b in range(n):
                    if b == u or b == a: continue
                    if edge_color[a][b] == -1: continue
                    for v in range(n):
                        if v == u or v == a or v == b: continue
                        if edge_color[b][v] == -1: continue
                        if edge_color[v][u] == -1: continue
                        
                        # Check colors
                        c1 = edge_color[u][a]
                        c2 = edge_color[a][b]
                        c3 = edge_color[b][v]
                        c4 = edge_color[v][u]
                        
                        if len({c1, c2, c3, c4}) < 4:
                            # Record violation. 
                            # We can break early if we just need one, but for local search 
                            # knowing all helps? No, fixing one might fix others or create new.
                            # Let's return the first few to avoid overhead.
                            violations.append((u, a, b, v))
                            if len(violations) > 100:
                                return violations
        return violations

    def try_fix(violation):
        u, a, b, v = violation
        # The cycle edges are (u,a), (a,b), (b,v), (v,u)
        # One of these edges has a color that conflicts.
        # Strategy: Randomly pick one of the 4 edges and try to change its color.
        # If no color works, try removing the edge.
        
        edges_in_cycle = [(u, a), (a, b), (b, v), (v, u)]
        random.shuffle(edges_in_cycle)
        
        for u2, v2 in edges_in_cycle:
            # Try to recolor edge (u2, v2)
            # We need to ensure that changing its color doesn't create new violations
            # involving this edge.
            
            # For simplicity in this first pass of structural change, 
            # let's just pick a random new color and check if it reduces violations locally?
            # Or better: pick a color that is not used by any other edge in any 4-cycle containing (u2, v2).
            
            # This is expensive to check fully. 
            # Heuristic: Pick a random color from available ones.
            # Then verify if the specific violation is fixed and no *new* immediate violations 
            # on this edge are created? 
            # Actually, just changing the color and re-evaluating is safer.
            
            # Let's collect forbidden colors for edge (u2, v2) based on current graph state.
            # A color c is forbidden for (u2, v2) if there exists a path u2-x-y-v2 
            # such that {c(color(u2,v2)), c(color(u2,x)), c(color(x,y)), c(color(y,v2))} has duplicates.
            
            forbidden = set()
            
            # Check all paths u2-x-y-v2
            for x in range(n):
                if x == u2 or x == v2: continue
                if edge_color[u2][x] == -1: continue
                c_ux = edge_color[u2][x]
                
                for y in range(n):
                    if y == u2 or y == v2 or y == x: continue
                    if edge_color[x][y] == -1: continue
                    if edge_color[y][v2] == -1: continue
                    c_xy = edge_color[x][y]
                    c_yv = edge_color[y][v2]
                    
                    # If we set color(u2, v2) = c, the set is {c, c_ux, c_xy, c_yv}
                    # We need all 4 distinct.
                    # So c must be distinct from c_ux, c_xy, c_yv.
                    # Also c_ux, c_xy, c_yv must be distinct from each other?
                    # If they are not, then no c can make the set distinct.
                    # But wait, the violation definition is about the 4-cycle.
                    # If c_ux, c_xy, c_yv already have duplicates, then ANY color for (u2,v2) 
                    # will result in a non-rainbow cycle? 
                    # Yes, because the other 3 edges already fail to be distinct.
                    # In that case, we should probably remove one of those edges, not recolor (u2,v2).
                    
                    if len({c_ux, c_xy, c_yv}) < 3:
                         # This path is "broken" regardless of (u2,v2)'s color.
                         # We can't fix it by recoloring (u2,v2).
                         # We should return False or signal to remove another edge?
                         # For now, let's assume we can't fix via recoloring (u2,v2) in this iteration.
                         return False 
                         
                    forbidden.add(c_ux)
                    forbidden.add(c_xy)
                    forbidden.add(c_yv)
            
            available = [c for c in colors if c not in forbidden]
            if available:
                new_c = random.choice(available)
                # Check if this change actually fixes the specific violation we found?
                # It should, by construction of forbidden set derived from all paths.
                # But we should verify it doesn't break something else? 
                # The forbidden set covers all 4-cycles containing (u2, v2).
                # So this change is "safe" for all 4-cycles involving (u2, v2).
                
                # However, it might not fix the *original* violation if the original violation 
                # was caused by duplicates among the OTHER 3 edges. 
                # But we checked that above (return False if other 3 have duplicates).
                # So if we get here, the other 3 are distinct, and we pick a 4th distinct.
                # So the specific cycle (u,a,b,v) will be rainbow.
                
                add_edge(u2, v2, new_c)
                return True
        
        # If no edge could be recolored safely, remove a random edge from the cycle
        u2, v2 = random.choice(edges_in_cycle)
        remove_edge(u2, v2)
        return True

    # Initialize with complete graph K_n
    # Assign random colors
    for u in range(n):
        for v in range(u+1, n):
            add_edge(u, v, random.choice(colors))
    
    best_edge_count = count_edges()
    best_state = [row[:] for row in edge_color]
    
    while time.time() < deadline:
        violations = find_violations()
        if not violations:
            # Valid graph found!
            # Try to add more edges? 
            # Since we started with K_n and only remove/recolor, 
            # we are already at max possible density for this specific search trajectory.
            # But we can try to "add back" removed edges?
            # For now, if valid, we can try to restart with different random init 
            # or just break if we are dense.
            # Actually, if we are valid, we can try to add missing edges.
            missing = []
            for u in range(n):
                for v in range(u+1, n):
                    if edge_color[u][v] == -1:
                        missing.append((u, v))
            
            if missing:
                # Try to add one
                random.shuffle(missing)
                added = False
                for u, v in missing:
                    # Try all colors
                    success = False
                    for c in colors:
                        # Check if adding (u,v) with c creates violation
                        # We can use a similar logic: check all paths u-x-y-v
                        forbidden = set()
                        valid = True
                        for x in range(n):
                            if x == u or x == v: continue
                            if edge_color[u][x] == -1: continue
                            c_ux = edge_color[u][x]
                            for y in range(n):
                                if y == u or y == v or y == x: continue
                                if edge_color[x][y] == -1: continue
                                if edge_color[y][v] == -1: continue
                                c_xy = edge_color[x][y]
                                c_yv = edge_color[y][v]
                                
                                if len({c_ux, c_xy, c_yv}) < 3:
                                    valid = False
                                    break
                                forbidden.add(c_ux)
                                forbidden.add(c_xy)
                                forbidden.add(c_yv)
                        if not valid:
                            break
                        if c in forbidden:
                            continue
                        # If we reach here, c is a candidate
                        # We should verify it doesn't break existing cycles?
                        # The forbidden set logic ensures no new 4-cycle containing (u,v) is bad.
                        # Existing cycles don't involve (u,v), so they are unaffected.
                        add_edge(u, v, c)
                        added = True
                        success = True
                        break
                    if added:
                        break
                if not added:
                    # No edge could be added. We are at a local optimum for this structure.
                    # Restart?
                    pass
            else:
                # Complete graph and valid? Unlikely but possible.
                pass
            
            # Update best
            current_count = count_edges()
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_state = [row[:] for row in edge_color]
            
            # Restart to explore other basins
            # Re-randomize colors?
            for u in range(n):
                for v in range(u+1, n):
                    if edge_color[u][v] != -1:
                        edge_color[u][v] = random.choice(colors)
                        edge_color[v][u] = edge_color[u][v]
            continue

        # If violations exist, try to fix one
        v = random.choice(violations)
        try_fix(v)
        
        # Update best
        current_count = count_edges()
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_state = [row[:] for row in edge_color]

    # Restore best state
    for i in range(n):
        for j in range(n):
            edge_color[i][j] = best_state[i][j]
            
    return get_edges()

