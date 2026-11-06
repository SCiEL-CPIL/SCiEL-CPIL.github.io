import random
import itertools
import time

def construct():
    n = 20
    # Potential edges: (u, v) for u < v
    # Potential colors: 0..19
    
    # We will try multiple random orderings of the potential colored edges
    # and greedily add them if they don't violate the rainbow 4-cycle constraint.
    
    # Precompute all 4-cycles (sets of 4 edge pairs) to speed up checking?
    # Or just check on the fly.
    # On the fly: When adding edge (u, v) with color c, 
    # we need to check if it forms a 4-cycle with any existing 3 edges.
    # A 4-cycle involving (u, v) is u-x-y-v-u.
    # Edges: (u,v), (v,x), (x,y), (y,u).
    # We need (v,x), (x,y), (y,u) to exist in the current graph.
    # And their colors must be distinct from c and each other.
    
    # Data structures:
    # graph: dict mapping (u, v) to color, or None.
    # We use (min(u,v), max(u,v)) as key.
    
    def try_construct(order_seed):
        random.seed(order_seed)
        graph = {} # key: (u, v) with u<v, value: color
        # List of all possible (u, v, c)
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                for c in range(n): # 20 colors
                    candidates.append((u, v, c))
        
        random.shuffle(candidates)
        
        # To speed up, we can prioritize edges that are "central" or just random.
        # We iterate through candidates and add if valid.
        
        # Helper to check validity of adding (u, v, c)
        def can_add(u, v, c):
            # Check all 4-cycles containing edge (u, v)
            # Cycle: u - x - y - v - u
            # Edges: (u, v), (v, x), (x, y), (y, u)
            # Note: (u, v) is the one we are adding.
            # We need (v, x), (x, y), (y, u) to be in graph.
            
            # Iterate x, y
            # x != u, v
            # y != u, v, x
            for x in range(n):
                if x == u or x == v:
                    continue
                # Edge (v, x) must exist
                key_vx = (min(v, x), max(v, x))
                if key_vx not in graph:
                    continue
                c_vx = graph[key_vx]
                if c_vx == c:
                    # Adjacent edges in 4-cycle must have different colors?
                    # Yes, rainbow requires all 4 distinct.
                    # So if c_vx == c, invalid.
                    return False
                
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    # Edge (x, y) must exist
                    key_xy = (min(x, y), max(x, y))
                    if key_xy not in graph:
                        continue
                    c_xy = graph[key_xy]
                    if c_xy == c or c_xy == c_vx:
                        return False
                    
                    # Edge (y, u) must exist
                    key_yu = (min(y, u), max(y, u))
                    if key_yu not in graph:
                        continue
                    c_yu = graph[key_yu]
                    if c_yu == c or c_yu == c_vx or c_yu == c_xy:
                        return False
                    
                    # If we reach here, we found a 4-cycle u-v-x-y-u (or similar)
                    # with colors c, c_vx, c_xy, c_yu.
                    # We checked they are all distinct.
                    # Wait, the condition is: if such a cycle EXISTS in the graph,
                    # the colors MUST be distinct.
                    # So if we are adding (u,v,c), and the other 3 edges are present,
                    # we MUST ensure c, c_vx, c_xy, c_yu are distinct.
                    # If they are not distinct, we CANNOT add (u,v,c).
                    
                    # My logic above:
                    # if c_vx == c: return False. (Correct)
                    # if c_xy == c or c_xy == c_vx: return False. (Correct)
                    # if c_yu == c or c_yu == c_vx or c_yu == c_xy: return False. (Correct)
                    
                    # If they are distinct, this specific cycle is fine.
                    # But we must check ALL such cycles.
                    # If ANY cycle formed by adding (u,v,c) has a color conflict, we cannot add it.
                    # So if we find ANY valid cycle (where the other 3 edges exist),
                    # we must ensure the colors are distinct.
                    # If they are distinct, we continue to check other x, y.
                    # If we find a cycle where the other 3 edges exist BUT colors are not distinct, we return False.
                    
                    # Wait, my logic `if c_yu ... return False` handles the "not distinct" case.
                    # If they ARE distinct, we do nothing and continue.
                    # This is correct.
            return True

        for u, v, c in candidates:
            if (u, v) in graph:
                continue # Already have an edge between u and v (with some color)
            # We can only have one edge between u and v.
            # So if (u,v) is in graph, we skip this candidate.
            
            if can_add(u, v, c):
                graph[(u, v)] = c
        
        # Convert to list
        result = []
        for (u, v), c in graph.items():
            result.append([u, v, c])
        return result

    # Run multiple trials with different seeds
    best_edges = []
    start_time = time.time()
    trial = 0
    while time.time() - start_time < 100: # Leave 20s buffer
        res = try_construct(trial)
        if len(res) > len(best_edges):
            best_edges = res
        trial += 1
        
    return best_edges
