"""Rewritten program for few edges in 5-uniform hypergraphs without Property B."""
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    if not edges:
        return False  # Empty hypergraph is 2-colorable
    
    edge_set = [tuple(sorted(e)) for e in edges]
    m = len(edge_set)
    
    # Build incidence: for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: assignment[v] = -1 (unassigned), 0 or 1
    # edge_status[i] = (count0, count1, unassigned_count)
    
    def backtrack(assignment, edge_count0, edge_count1, edge_unassigned):
        # Check if any edge is already monochromatic (all 0 or all 1)
        for i in range(m):
            if edge_count0[i] == 5 or edge_count1[i] == 5:
                return False  # This edge is monochromatic, so this coloring is invalid... 
                # Wait, we're checking if there EXISTS a proper 2-coloring.
                # A proper 2-coloring requires NO edge to be monochromatic.
                # So if an edge is monochromatic, this partial assignment cannot lead to a proper coloring.
        
        # Find first unassigned vertex
        idx = -1
        for i in range(n):
            if assignment[i] == -1:
                idx = i
                break
        if idx == -1:
            # All assigned, check if all edges are non-monochromatic
            for i in range(m):
                if edge_count0[i] == 5 or edge_count1[i] == 5:
                    return False
            return True
        
        for color in [0, 1]:
            new_assignment = assignment[:]
            new_assignment[idx] = color
            
            # Update edge counts
            new_edge_count0 = edge_count0[:]
            new_edge_count1 = edge_count1[:]
            new_edge_unassigned = edge_unassigned[:]
            
            valid = True
            for ei in vertex_edges[idx]:
                e = edge_set[ei]
                if color == 0:
                    new_edge_count0[ei] += 1
                else:
                    new_edge_count1[ei] += 1
                new_edge_unassigned[ei] -= 1
                
                # Check if this edge is now monochromatic
                if new_edge_count0[ei] == 5 or new_edge_count1[ei] == 5:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Unit propagation: if an edge has 4 of one color and 1 unassigned, force the last
            changed = True
            while changed and valid:
                changed = False
                for ei in range(m):
                    if new_edge_unassigned[ei] == 0:
                        if new_edge_count0[ei] == 5 or new_edge_count1[ei] == 5:
                            valid = False
                            break
                        continue
                    if new_edge_unassigned[ei] == 1:
                        if new_edge_count0[ei] == 4:
                            # Last vertex must be 1
                            last_v = [v for v in edge_set[ei] if new_assignment[v] == -1][0]
                            if new_assignment[last_v] != 1:
                                if new_assignment[last_v] == 0:
                                    valid = False
                                    break
                                new_assignment[last_v] = 1
                                new_edge_count1[ei] += 1
                                new_edge_unassigned[ei] -= 1
                                changed = True
                        elif new_edge_count1[ei] == 4:
                            # Last vertex must be 0
                            last_v = [v for v in edge_set[ei] if new_assignment[v] == -1][0]
                            if new_assignment[last_v] != 0:
                                if new_assignment[last_v] == 1:
                                    valid = False
                                    break
                                new_assignment[last_v] = 0
                                new_edge_count0[ei] += 1
                                new_edge_unassigned[ei] -= 1
                                changed = True
            
            if valid:
                if backtrack(new_assignment, new_edge_count0, new_edge_count1, new_edge_unassigned):
                    return True
        return False
    
    initial_assignment = [-1] * n
    initial_edge_count0 = [0] * m
    initial_edge_count1 = [0] * m
    initial_edge_unassigned = [5] * m
    
    # If there exists a proper 2-coloring, return False (it IS 2-colorable)
    # If no proper 2-coloring exists, return True (it is NOT 2-colorable)
    has_proper_coloring = backtrack(initial_assignment, initial_edge_count0, initial_edge_count1, initial_edge_unassigned)
    return not has_proper_coloring

def _try_remove_edges(n, edges, max_time=10.0):
    """Try to remove edges while maintaining non-2-colorability."""
    edges = list(edges)
    start_time = time.time()
    
    for i in range(len(edges)):
        if time.time() - start_time > max_time:
            break
        candidate = edges[:i] + edges[i+1:]
        if _is_not_2_colorable(n, candidate):
            edges = candidate
            i = -1  # Restart since we removed an edge
    return edges

def construct():
    """Find a small non-2-colorable 5-uniform hypergraph."""
    start_time = time.time()
    deadline = 140  # Safety margin
    
    best_result = None
    best_edges = 999
    
    # Start with a known construction: the 51-edge hypergraph
    # We'll use a random search to find small non-2-colorable hypergraphs
    
    # Try various n values
    for n in range(10, 20):
        if time.time() > deadline:
            break
        
        # Try to find a non-2-colorable hypergraph with few edges
        # Use random edge selection with pruning
        
        max_edges_to_try = 60
        for target_edges in range(30, max_edges_to_try + 1):
            if time.time() > deadline:
                break
            
            # Try multiple random attempts
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                # Randomly select edges
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                
                edges = []
                for e in all_edges:
                    if len(edges) >= target_edges:
                        break
                    candidate = edges + [e]
                    # Quick check: only add if it might help
                    edges = candidate
                
                if len(edges) == target_edges:
                    if _is_not_2_colorable(n, edges):
                        if target_edges < best_edges:
                            best_edges = target_edges
                            best_result = {'n': n, 'edges': [list(e) for e in edges]}
                            # Try to reduce further
                            reduced = _try_remove_edges(n, edges, max_time=5.0)
                            if len(reduced) < best_edges:
                                best_edges = len(reduced)
                                best_result = {'n': n, 'edges': [list(e) for e in reduced]}
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
