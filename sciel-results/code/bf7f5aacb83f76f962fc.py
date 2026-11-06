"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    edge_set = [tuple(sorted(e)) for e in edges]
    
    # Build incidence: vertex -> list of edge indices
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vert_edges[v].append(i)
    
    # State: assignment[v] = -1 (unassigned), 0 or 1
    assignment = [-1] * n
    
    def unit_propagate():
        """Apply unit propagation. Returns False if contradiction, True otherwise."""
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_set):
                # Count colors in this edge
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if assignment[v] == 0:
                        c0 += 1
                    elif assignment[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # If 4 are color 0, the rest must be color 1
                if c0 == 4 and len(unassigned) == 1:
                    assignment[unassigned[0]] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    assignment[unassigned[0]] = 0
                    changed = True
                # If all 5 are same color, contradiction
                elif c0 == 5 or c1 == 5:
                    return False
                # If 4 of one color and more than 1 unassigned, impossible to avoid mono
                # Actually no, we just need at least one of each. If 4 are 0, at least 1 must be 1.
                # If there are 2 unassigned and 4 are 0, that's fine (one can be 1, one can be 0)
                # But if 4 are 0 and 0 unassigned, that's c0==5 handled above
                # If 4 are 0 and 1 unassigned, that unassigned MUST be 1 (handled above)
        return True
    
    def backtrack(idx):
        """Try to assign vertex idx. Returns True if a valid 2-coloring exists."""
        if idx == n:
            # Check all edges are not monochromatic
            for i, e in enumerate(edge_set):
                colors = set(assignment[v] for v in e)
                if len(colors) == 1:
                    return False
            return True
        
        for color in [0, 1]:
            assignment[idx] = color
            if unit_propagate():
                # Find next unassigned vertex
                next_idx = -1
                for j in range(idx + 1, n):
                    if assignment[j] == -1:
                        next_idx = j
                        break
                if next_idx == -1:
                    # All assigned, check
                    if backtrack(n):
                        return True
                else:
                    if backtrack(next_idx):
                        return True
            assignment[idx] = -1
            # Reset any vertices that were assigned by unit propagation
            # This is tricky - we need to track what was changed
            # Simplification: just reset all
            for j in range(n):
                if assignment[j] != -1 and j != idx:
                    # We can't easily undo unit propagation
                    # Better approach: save state before propagation
                    pass
        return False
    
    # Better backtracking with proper state saving
    def backtrack2(assignment_state):
        """Backtracking with explicit state."""
        # Find first unassigned vertex
        idx = -1
        for i in range(n):
            if assignment_state[i] == -1:
                idx = i
                break
        if idx == -1:
            # All assigned, check
            for e in edge_set:
                colors = set(assignment_state[v] for v in e)
                if len(colors) == 1:
                    return False
            return True
        
        for color in [0, 1]:
            new_state = assignment_state[:]
            new_state[idx] = color
            # Unit propagation
            changed = True
            valid = True
            while changed and valid:
                changed = False
                for e in edge_set:
                    c0 = sum(1 for v in e if new_state[v] == 0)
                    c1 = sum(1 for v in e if new_state[v] == 1)
                    unassigned = [v for v in e if new_state[v] == -1]
                    
                    if c0 == 5 or c1 == 5:
                        valid = False
                        break
                    if c0 == 4 and len(unassigned) == 1:
                        new_state[unassigned[0]] = 1
                        changed = True
                    elif c1 == 4 and len(unassigned) == 1:
                        new_state[unassigned[0]] = 0
                        changed = True
            
            if valid:
                if backtrack2(new_state):
                    return True
        return False
    
    return not backtrack2([-1] * n)

def _search_for_small(n, num_edges, time_limit=10):
    """Search for a non-2-colorable hypergraph with given n and num_edges."""
    start_time = time.time()
    all_edges = list(combinations(range(n), 5))
    
    while time.time() - start_time < time_limit:
        edges = random.sample(all_edges, num_edges)
        if _is_not_2_colorable(n, edges):
            return edges
    return None

def construct():
    """Find a small non-2-colorable 5-uniform hypergraph."""
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    # Try different sizes and edge counts
    best_result = None
    
    # Known: complete hypergraph on 9 vertices works with 126 edges
    # Try to find smaller ones
    
    # Strategy: try n from 9 to 15, and edge counts from 30 upward
    for n in range(9, 16):
        if time.time() > deadline:
            break
        for num_edges in range(30, 127, 5):
            if time.time() > deadline:
                break
            result = _search_for_small(n, num_edges, time_limit=5)
            if result is not None:
                if best_result is None or num_edges < len(best_result['edges']):
                    best_result = {'n': n, 'edges': [list(e) for e in result]}
                    # Try to reduce further
                    break
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
