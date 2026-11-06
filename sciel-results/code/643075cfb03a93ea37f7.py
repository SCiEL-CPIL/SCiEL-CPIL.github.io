import random
import time
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Build incidence
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    colors = [-1] * n
    
    def propagate(v, c):
        """Assign color c to vertex v. Returns False on contradiction."""
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            c0 = sum(1 for u in e if colors[u] == 0)
            c1 = sum(1 for u in e if colors[u] == 1)
            
            if c0 == 5 or c1 == 5:
                return False
            
            if c1 == 4:
                unassigned = [u for u in e if colors[u] == -1]
                if unassigned and not propagate(unassigned[0], 0):
                    return False
            elif c0 == 4:
                unassigned = [u for u in e if colors[u] == -1]
                if unassigned and not propagate(unassigned[0], 1):
                    return False
        
        return True
    
    def backtrack():
        # Find the most constrained edge (fewest unassigned vertices)
        best_v = -1
        best_constraint = 999
        best_edge_idx = -1
        best_forced = -1
        
        for e_idx in range(m):
            e = edges[e_idx]
            unassigned = 0
            c0 = 0
            c1 = 0
            for u in e:
                if colors[u] == -1:
                    unassigned += 1
                elif colors[u] == 0:
                    c0 += 1
                else:
                    c1 += 1
            
            if unassigned == 0:
                if c0 == 5 or c1 == 5:
                    return False
                continue
            
            if c1 == 4 or c0 == 4:
                if unassigned == 1:
                    best_constraint = 0
                    best_v = [u for u in e if colors[u] == -1][0]
                    best_edge_idx = e_idx
                    best_forced = 0 if c1 == 4 else 1
                    break
                elif unassigned < best_constraint:
                    best_constraint = unassigned
                    best_v = e[0]
                    best_edge_idx = e_idx
                    best_forced = 0 if c1 == 4 else 1
            elif unassigned < best_constraint:
                best_constraint = unassigned
                best_v = e[0]
                best_edge_idx = e_idx
                best_forced = -1
        
        if best_v == -1:
            for v in range(n):
                if colors[v] == -1:
                    best_v = v
                    best_constraint = 999
                    break
            if best_v == -1:
                return True
        
        if best_forced != -1:
            e = edges[best_edge_idx]
            for u in e:
                if colors[u] == -1:
                    best_v = u
                    break
            
            old = colors[:]
            if propagate(best_v, best_forced):
                if backtrack():
                    return True
            colors[:] = old
            
            old = colors[:]
            if propagate(best_v, 1 - best_forced):
                if backtrack():
                    return True
            colors[:] = old
        else:
            if best_edge_idx != -1:
                e = edges[best_edge_idx]
                for u in e:
                    if colors[u] == -1:
                        best_v = u
                        break
            else:
                best_deg = -1
                for v in range(n):
                    if colors[v] == -1:
                        deg = len(vertex_edges[v])
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
            
            old = colors[:]
            if propagate(best_v, 0):
                if backtrack():
                    return True
            colors[:] = old
            
            old = colors[:]
            if propagate(best_v, 1):
                if backtrack():
                    return True
            colors[:] = old
        
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with complete 5-uniform on 9 vertices: 126 edges
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    def try_remove_edges(edges, n, time_limit):
        """Greedy edge removal: remove edge if hypergraph remains non-2-colorable."""
        edges = [list(e) for e in edges]
        changed = True
        # Shuffle to get different local minima
        random.shuffle(edges)
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(edges)):
                if time.time() > time_limit:
                    break
                remaining = edges[:i] + edges[i+1:]
                # Check if remaining is non-2-colorable
                if not is_2_colorable(remaining, n):
                    edges = remaining
                    changed = True
                    break
        return edges
    
    # Phase 1: Reduce the complete 9-vertex hypergraph
    if time.time() < deadline - 40:
        reduced = try_remove_edges(best_edges, best_n, time_limit=20.0)
        # Ensure reduced is valid (non-2-colorable and non-empty)
        if len(reduced) > 0 and not is_2_colorable(reduced, best_n) and len(reduced) < best_count:
            best_count = len(reduced)
            best_edges = reduced
    
    # Phase 2: Try to build a smaller one on n=10 using random restarts
    # We look for a set of edges that is non-2-colorable.
    # Strategy: Start with a random subset, check if non-2-colorable.
    # If yes, try to minimize.
    # This is expensive. We'll rely on the n=9 reduction mostly.
    
    # If we have time, try a few random subsets on n=9 to see if we can find a smaller non-2-colorable core
    if time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            # Pick a random subset of the best_edges found so far
            # Or pick random edges from K9
            all_9 = list(combinations(range(9), 5))
            random.shuffle(all_9)
            
            # Try to find a small non-2-colorable set by adding edges
            current = []
            for e in all_9:
                if len(current) >= best_count:
                    break
                candidate = current + [list(e)]
                # We want to find a point where it becomes non-2-colorable
                # But checking every step is slow.
                # Instead, let's just try to reduce the current best further with different shuffles
                pass
            
            # Alternative: Just run try_remove_edges again with a different shuffle
            reduced = try_remove_edges(best_edges, best_n, time_limit=5.0)
            if len(reduced) > 0 and not is_2_colorable(reduced, best_n) and len(reduced) < best_count:
                best_count = len(reduced)
                best_edges = reduced
    
    return {'n': best_n, 'edges': best_edges}
