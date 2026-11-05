"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
import time
import random
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
    if not edges:
        return True
        
    # Precompute edge masks for fast monochromaticity check if needed,
    # but the backtracking below uses vertex colors directly.
    
    # Represent colors: -1 = unassigned, 0 = color A, 1 = color B
    colors = [-1] * n
    
    # Build index: for each vertex, which edges contain it
    # Not strictly necessary for the propagation logic below if we iterate all edges,
    # but useful for future optimizations. For now, iterate all edges.
    
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
        if _check_time():
            # If we run out of time, assume satisfiable to avoid hanging.
            # This is a heuristic safeguard.
            return True
            
        if idx >= n:
            # Check all edges are not monochromatic
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                if c0 == 5:
                    return False
            return True
        
        # Try both colors
        for c in (0, 1):
            colors[idx] = c
            # Save state for undo
            saved = colors[idx:]
            if propagate():
                if backtrack(idx + 1):
                    return True
            # Restore
            colors[idx:] = saved
            
        return False
    
    return backtrack(0)

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Returns {'n': n, 'edges': [[v1,...,v5], ...]}
    """
    best_edges = None
    best_count = float('inf')
    
    # Start with the complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    best_edges = list(all_edges)
    best_count = len(all_edges)
    
    start_time = time.time()
    limit = 145.0  # 5s margin from the 150s hard limit
    
    # Strategy: Start with the complete hypergraph and greedily remove edges
    # as long as the hypergraph remains non-2-colorable.
    # Use random restarts with different orderings to escape local optima.
    
    for restart in range(1000):
        if time.time() - start_time > limit:
            break
            
        # Shuffle edges to try different removal orders
        edges_to_try = list(all_edges)
        random.shuffle(edges_to_try)
        
        kept = set(all_edges)
        
        for e in edges_to_try:
            if e not in kept:
                continue
            
            # Check time
            if time.time() - start_time > limit:
                break
                
            # Try removing edge e
            kept.discard(e)
            
            # Check if the remaining hypergraph is still non-2-colorable
            # i.e., _is_2_colorable returns False
            if not _is_2_colorable(n, list(kept)):
                # Still non-2-colorable, keep it removed
                pass
            else:
                # Became 2-colorable, restore the edge
                kept.add(e)
        
        # Update best if we found a smaller set
        if len(kept) < best_count:
            best_count = len(kept)
            best_edges = list(kept)
            
            # If we reach the known lower bound of 51, we can stop or continue
            # to see if we can go lower (unlikely but possible if the bound is not tight).
            if best_count <= 51:
                # Continue for a bit to see if we can improve further, but likely 51 is optimal.
                # We'll let the time limit handle stopping.
                pass

    return {'n': n, 'edges': [list(e) for e in best_edges]}
