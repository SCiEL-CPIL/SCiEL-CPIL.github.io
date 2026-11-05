"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
import time
from itertools import combinations

# Deadline guard: evaluator kills after 150s, keep safety margin
_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_time():
    return _elapsed() > _DEADLINE

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph has a proper 2-coloring
    (no edge is monochromatic) using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    colors = [-1] * n
    
    def propagate():
        """Apply unit propagation. Returns False on contradiction, True otherwise."""
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # Contradiction: all 5 same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit propagation: 4 of one color forces the 5th
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 1
                        changed = True
                    elif colors[v] == 0:
                        return False
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 0
                        changed = True
                    elif colors[v] == 1:
                        return False
        return True
    
    def backtrack(idx):
        """Try to assign vertex idx and beyond. Returns True if satisfiable."""
        if idx >= n:
            # Check all edges are not monochromatic
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Try both colors
        for c in (0, 1):
            colors[idx] = c
            saved = colors[idx:]
            if propagate():
                if backtrack(idx + 1):
                    return True
            colors[idx:] = saved
            if _check_time():
                return True  # Assume satisfiable to avoid timeout issues
        
        return False
    
    return backtrack(0)

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Returns {'n': n, 'edges': [[v1,...,v5], ...]}
    """
    # Start with complete 5-uniform hypergraph on 9 vertices (126 edges)
    n = 9
    current_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Greedily remove edges while maintaining non-2-colorability
    # Try removing each edge in order; if the result is still non-2-colorable, keep the removal
    improved = True
    removals = 0
    while improved and not _check_time() and removals < 100:
        improved = False
        for i in range(len(current_edges)):
            if _check_time():
                break
            candidate = current_edges[:i] + current_edges[i+1:]
            if not _is_2_colorable(n, candidate):
                current_edges = candidate
                improved = True
                removals += 1
                break
    
    return {'n': n, 'edges': current_edges}
