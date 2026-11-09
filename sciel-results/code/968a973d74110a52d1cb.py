import numpy as np
import time
import random

N = 20
C = 20
DEADLINE = time.time() + 110.0

def _get_forbidden_colors(u, v, adj):
    """
    Returns a set of colors forbidden for edge (u, v).
    If the 3 other edges in any 4-cycle have duplicate colors, return all colors (blocked).
    """
    forbidden = set()
    # 4-cycle: u-v-x-y-u
    # x is neighbor of v, y is neighbor of u, and (x,y) is an edge
    # We iterate over all x, y
    for x in range(N):
        if x == u or x == v:
            continue
        if adj[v, x] == -1:
            continue
        for y in range(N):
            if y == u or y == v or y == x:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            # If the existing 3 edges don't have distinct colors, 
            # this 4-cycle can never be rainbow, so (u,v) cannot be added with ANY color.
            if c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                return set(range(C))
            
            # Otherwise, c_uv just needs to be different from these three
            forbidden.add(c_vx)
            forbidden.add(c_xy)
            forbidden.add(c_yu)
            
    return forbidden

def _is_recolor_valid(u, v, new_c, adj):
    """
    Check if changing color of edge (u,v) to new_c keeps all 4-cycles rainbow.
    (u,v) must already exist in adj.
    """
    for x in range(N):
        if x == u or x == v:
            continue
        if adj[v, x] == -1:
            continue
        for y in range(N):
            if y == u or y == v or y == x:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            if new_c == c_vx or new_c == c_xy or new_c == c_yu:
                return False
            if c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                # This cycle was already invalid? 
                # Wait, if the graph is valid, this shouldn't happen for existing cycles.
                # But if we are recoloring, we assume the graph was valid before.
                # If c_vx == c_xy, then the cycle u-v-x-y-u was NOT rainbow before either?
                # No, if (u,v) existed, the cycle existed. If it was valid, colors were distinct.
                # So c_vx, c_xy, c_yu were distinct from old_c(u,v) and each other.
                # So c_vx != c_xy etc. must hold.
                return False 
    return True

def _try_build(seed):
    rng = random.Random(seed)
    adj = -np.ones((N, N), dtype=int)
    
    # 1. Greedy Construction
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > DEADLINE - 5.0:
            break
        if adj[u, v] != -1:
            continue
            
        forb = _get_forbidden_colors(u, v, adj)
        if len(forb) < C:
            available = [c for c in range(C) if c not in forb]
            c = rng.choice(available)
            adj[u, v] = c
            adj[v, u] = c
            
    # 2. Local Search
    # We try to add missing edges and recolor existing edges to unblock others.
    # We repeat this process for a fixed number of iterations or until no improvement.
    
    best_count = np.sum(adj != -1) // 2
    no_improve_count = 0
    max_no_improve = 50 # Break if stuck for 50 iterations
    
    for _ in range(1000):
        if time.time() > DEADLINE - 5.0:
            break
        
        improved = False
        
        # Phase 2a: Try to add missing edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u, v] == -1]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > DEADLINE - 5.0:
                break
            if adj[u, v] != -1:
                continue
                
            forb = _get_forbidden_colors(u, v, adj)
            if len(forb) < C:
                available = [c for c in range(C) if c not in forb]
                c = rng.choice(available)
                adj[u, v] = c
                adj[v, u] = c
                improved = True
        
        if improved:
            no_improve_count = 0
            continue
        
        # Phase 2b: Try to recolor existing edges to unblock additions
        # We pick a random existing edge and try to change its color.
        # If the new color allows adding at least one missing edge, we keep it.
        # Actually, a simpler heuristic: just try to recolor an edge to a random valid color.
        # If it doesn't immediately allow an addition, it might still help later?
        # Better: Try to recolor an edge that is "constraining" a missing edge.
        
        # For now, let's just try random recoloring of a random edge.
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u, v] != -1]
        if not existing:
            break
            
        rng.shuffle(existing)
        
        # Try to recolor up to 20 random edges per iteration
        for u, v in existing[:20]:
            if time.time() > DEADLINE - 5.0:
                break
            
            old_c = adj[u, v]
            # Get forbidden colors for (u,v) assuming it's not there? No, we need to find a color 
            # that is valid for (u,v) given the rest of the graph.
            # The constraints on (u,v)'s color come from 4-cycles containing (u,v).
            # These are the same constraints as when adding (u,v).
            forb = _get_forbidden_colors(u, v, adj)
            
            # But wait, _get_forbidden_colors checks if the OTHER 3 edges form a valid path.
            # If they do, it returns the colors of those 3 edges.
            # So any color NOT in forb is valid for (u,v).
            
            if len(forb) < C:
                available = [c for c in range(C) if c not in forb]
                if old_c not in available:
                    # This shouldn't happen if the graph is valid, but just in case
                    continue
                    
                # Pick a new color different from old_c
                if len(available) > 1:
                    new_c = rng.choice([c for c in available if c != old_c])
                    # Check if this recoloring is valid (it should be, by definition of forb)
                    # But we must ensure it doesn't break other cycles?
                    # No, the only cycles involving (u,v) are the ones considered in _get_forbidden_colors.
                    # Changing (u,v) only affects cycles containing (u,v).
                    # Since new_c is not in forb, it is distinct from the other 3 edges in all such cycles.
                    # And the other 3 edges are distinct from each other (graph was valid).
                    # So the new cycle is rainbow.
                    
                    # Apply recoloring
                    adj[u, v] = new_c
                    adj[v, u] = new_c
                    improved = True
                    # We might break here to re-evaluate missing edges immediately?
                    # Or continue recoloring?
                    # Let's break to re-run the "add missing" phase.
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        else:
            no_improve_count = 0
            
    # Convert to list of lists for output
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u, v] != -1:
                result.append([u, v, int(adj[u, v])])
                
    return result

def construct():
    best_edges = []
    best_count = 0
    start_time = time.time()
    
    # Run multiple restarts
    seed = 0
    while time.time() < DEADLINE - 10.0:
        cand = _try_build(seed)
        if len(cand) > best_count:
            best_count = len(cand)
            best_edges = cand
            # Optional: if we reach a very high density, we might stop?
            # Max is 190. If we get > 150, it's good.
            # But keep going to try for more.
        seed += 1
        
    return best_edges
