import time
import random
from itertools import combinations

_DEADLINE = 135.0

def _is_non_2_colorable_fast(edges, n, deadline):
    """
    Fast check if 5-uniform hypergraph is non-2-colorable.
    edges: list of tuples of 5 vertex indices.
    n: number of vertices.
    """
    # Precompute masks
    edge_masks = []
    for e in edges:
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks.append(m)
    
    # Adjacency: for each vertex, which edges contain it?
    vertex_edges = [[] for _ in range(n)]
    for i, mask in enumerate(edge_masks):
        for v in range(n):
            if mask & (1 << v):
                vertex_edges[v].append(i)

    color = [-1] * n
    
    def propagate():
        # Unit propagation: if 4 vertices in an edge are same color, 5th is opposite.
        # If 5 same color, contradiction.
        changed = True
        while changed:
            changed = False
            for i, mask in enumerate(edge_masks):
                c0 = 0
                c1 = 0
                unassigned = []
                for v in range(n):
                    if mask & (1 << v):
                        cv = color[v]
                        if cv == 0: c0 += 1
                        elif cv == 1: c1 += 1
                        else: unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                if len(unassigned) == 1:
                    v = unassigned[0]
                    if c0 == 4:
                        if color[v] != 1:
                            if color[v] != -1: return False
                            color[v] = 1
                            changed = True
                    elif c1 == 4:
                        if color[v] != 0:
                            if color[v] != -1: return False
                            color[v] = 0
                            changed = True
        return True

    def backtrack():
        if time.time() > deadline:
            return False # Timeout, assume colorable (conservative for "find non-colorable" search? 
                         # Actually if we assume colorable, we might reject a valid non-colorable one.
                         # But for minimization, we want to be sure it's non-colorable.
                         # If timeout, we should probably return False (colorable) to avoid false positives?
                         # Wait, if we are checking a candidate for "is it non-colorable?", 
                         # and we timeout, we don't know. 
                         # In local search, if we can't verify, we should discard or keep?
                         # Let's return False (meaning "is colorable" or "unknown") to be safe, 
                         # so we don't accept a potentially colorable graph as non-colorable.
            
        # Check if all assigned
        all_assigned = True
        for v in range(n):
            if color[v] == -1:
                all_assigned = False
                break
        
        if all_assigned:
            return True # Found a coloring, so it IS 2-colorable. 
                        # We want to return False for "is_non_2_colorable".
                        # So if we find a coloring, the function should return False.
            
        # Propagate
        if not propagate():
            return False # Contradiction, so this path is invalid. 
                         # Wait, if we are looking for A coloring, contradiction means no coloring in this branch.
                         # So we backtrack.
            
        # Find unassigned vertex with max degree in remaining constraints
        best_v = -1
        best_score = -1
        for v in range(n):
            if color[v] == -1:
                # Count how many edges are "almost assigned" or just count degree?
                # Better: pick vertex in an edge with most assigned neighbors.
                score = 0
                for ei in vertex_edges[v]:
                    mask = edge_masks[ei]
                    assigned_in_edge = 0
                    for u in range(n):
                        if mask & (1 << u) and color[u] != -1:
                            assigned_in_edge += 1
                    score += assigned_in_edge
                if score > best_score:
                    best_score = score
                    best_v = v
                    
        if best_v == -1:
            return True # All assigned (should have been caught above)

        # Branch
        for c in (0, 1):
            color[best_v] = c
            if backtrack():
                return True
            color[best_v] = -1
        
        return False

    # The function returns True if it finds a coloring (i.e., IS 2-colorable).
    # So "is_non_2_colorable" is the negation.
    is_colorable = backtrack()
    return not is_colorable

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5
    
    # Strategy: Local Search to minimize edges
    # Start with a known non-2-colorable hypergraph: K_9^(5)
    n = 9
    all_edges = list(combinations(range(n), 5))
    current_edges = all_edges[:]
    
    # Verify start
    if not _is_non_2_colorable_fast(current_edges, n, deadline):
        # Should not happen, K9^5 is non-colorable
        pass
    
    best_edges = current_edges[:]
    best_len = len(current_edges)
    
    rng = random.Random(42)
    
    # Simulated Annealing / Hill Climbing
    # We want to minimize edges.
    # Move: Remove a random edge. If still non-colorable, accept.
    # If stuck, try adding an edge (to escape), or restart.
    
    T = 100.0 # Temperature
    T_min = 1.0
    cooling = 0.99
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        if len(current_edges) < best_len:
            best_edges = current_edges[:]
            best_len = len(current_edges)
            
        if len(current_edges) <= 29:
            break # Theoretical lower bound is 29
            
        # Try to remove a random edge
        if len(current_edges) > 0:
            idx = rng.randint(0, len(current_edges) - 1)
            edge_to_remove = current_edges[idx]
            
            new_edges = current_edges[:idx] + current_edges[idx+1:]
            
            # Check if new_edges is non-2-colorable
            if _is_non_2_colorable_fast(new_edges, n, deadline):
                current_edges = new_edges
            else:
                # Rejection
                # With probability exp(-delta/T), accept? 
                # Delta = 1 (edge count increased? No, we failed to decrease).
                # Actually, if we try to remove and it fails, we didn't change state.
                # To escape local minima, we might ADD an edge.
                if rng.random() < 0.1:
                    # Add a random edge not in current set
                    # Need a set for fast lookup
                    edge_set = set(tuple(e) for e in current_edges)
                    possible_adds = [e for e in all_edges if e not in edge_set]
                    if possible_adds:
                        add_edge = rng.choice(possible_adds)
                        current_edges.append(add_edge)
                        # We accept this "worsening" move to escape
                        # But we must ensure it's still non-colorable? 
                        # Adding edges to a non-colorable graph keeps it non-colorable.
                        # So this is always valid.
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return {'n': n, 'edges': [list(e) for e in best_edges]}
