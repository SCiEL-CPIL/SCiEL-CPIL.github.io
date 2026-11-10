"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills at 150s, keep safety margin.
_DEADLINE = 135.0  # seconds

def _edge_mask(edge, n):
    """Convert a 5-element edge to a bit mask over n vertices."""
    m = 0
    for v in edge:
        m |= (1 << v)
    return m

def _is_2_colorable(edges_masks, n, deadline):
    """
    Check if the hypergraph (given as list of bit masks) has a proper 2-coloring.
    A proper 2-coloring means no edge is monochromatic.
    We use backtracking with unit propagation.
    
    Returns True if 2-colorable, False if not (i.e., no proper 2-coloring exists).
    """
    # Precompute for each edge: list of vertex indices
    edge_verts = []
    for mask in edges_masks:
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        edge_verts.append(verts)
    
    # State: color assignment array
    color = [None] * n  # None = unassigned, 0 or 1 = assigned
    
    def propagate():
        """Apply unit propagation. Returns (success, assignments_made) where assignments_made is a list of (v, c) to undo."""
        assignments = []
        changed = True
        while changed:
            changed = False
            for i, verts in enumerate(edge_verts):
                c0 = 0
                c1 = 0
                unassigned = []
                for v in verts:
                    if color[v] is None:
                        unassigned.append(v)
                    elif color[v] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                # Check for contradiction
                if c0 == 5 or c1 == 5:
                    return False, assignments
                # Unit propagation: if 4 of one color, the 5th must be opposite
                if len(unassigned) == 1:
                    if c0 == 4:
                        v = unassigned[0]
                        if color[v] is not None:
                            if color[v] != 1:
                                return False, assignments
                        else:
                            color[v] = 1
                            assignments.append((v, 1))
                            changed = True
                    elif c1 == 4:
                        v = unassigned[0]
                        if color[v] is not None:
                            if color[v] != 0:
                                return False, assignments
                        else:
                            color[v] = 0
                            assignments.append((v, 0))
                            changed = True
        return True, assignments
    
    def backtrack(depth):
        if time.time() > deadline:
            return False  # timeout, assume colorable (conservative)
        
        # Check if all assigned
        if all(c is not None for c in color):
            return True  # found a valid coloring
        
        # Propagate
        ok, assignments = propagate()
        if not ok:
            return False
        
        # Find an unassigned vertex to branch on
        # Heuristic: pick vertex in an edge with most assigned vertices
        best_v = -1
        best_constraint = -1
        for i, verts in enumerate(edge_verts):
            assigned_count = sum(1 for v in verts if color[v] is not None)
            if assigned_count > best_constraint:
                best_constraint = assigned_count
                for v in verts:
                    if color[v] is None:
                        best_v = v
                        break
                if best_v != -1:
                    break
        
        if best_v == -1:
            # All vertices assigned but we didn't detect it? Shouldn't happen.
            return True
        
        # Try both colors
        for c in (0, 1):
            color[best_v] = c
            if backtrack(depth + 1):
                return True
            color[best_v] = None
        
        # Undo assignments made by propagation
        for v, c in assignments:
            color[v] = None
        
        return False
    
    return backtrack(0)

def _try_random_hypergraph(n, num_edges, deadline, rng):
    """
    Try to find a non-2-colorable 5-uniform hypergraph on n vertices with exactly num_edges edges.
    Returns (edges_list, success) where edges_list is the list of edges if successful.
    """
    all_edges = list(combinations(range(n), 5))
    total = len(all_edges)
    
    if num_edges > total:
        return None, False
    
    # Try several random subsets
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None, False
        
        # Sample num_edges edges
        if num_edges == total:
            sampled = all_edges
        else:
            sampled = rng.sample(all_edges, num_edges)
        
        # Convert to masks
        masks = [_edge_mask(e, n) for e in sampled]
        
        # Check 2-colorability
        colorable = _is_2_colorable(masks, n, deadline)
        
        if not colorable:
            return [list(e) for e in sampled], True
    
    return None, False

def _greedy_reduce(edges, n, deadline):
    """
    Given a non-2-colorable hypergraph, greedily remove redundant edges.
    An edge is redundant if removing it still leaves the hypergraph non-2-colorable.
    Returns the reduced edge list.
    """
    current_edges = list(edges)
    current_masks = [_edge_mask(e, n) for e in current_edges]
    
    # Try removing edges in random order, multiple passes
    rng = random.Random(12345)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of edges to try
        indices = list(range(len(current_edges)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing edge at idx
            if len(current_edges) <= 1:
                break
            
            # Build list without this edge
            test_edges = current_edges[:idx] + current_edges[idx+1:]
            test_masks = [_edge_mask(e, n) for e in test_edges]
            
            # Check if still non-2-colorable
            colorable = _is_2_colorable(test_masks, n, deadline)
            
            if not colorable:
                # Edge is redundant, remove it
                current_edges = test_edges
                current_masks = test_masks
                improved = True
    
    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5  # 5 second safety margin
    
    rng = random.Random(42)
    
    best_edges = None
    best_n = 0
    best_count = 201  # more than max allowed
    
    # Strategy 1: Start with complete hypergraph on small n, greedily reduce
    # Complete 5-uniform hypergraph on 9 vertices has C(9,5)=126 edges
    # Try to reduce it significantly
    
    # First, try greedy reduction on complete hypergraphs of various sizes
    for n in range(9, 13):
        if time.time() > deadline:
            break
        
        all_edges = list(combinations(range(n), 5))
        if len(all_edges) > 200:
            continue
        
        # Check if complete hypergraph is non-2-colorable
        masks = [_edge_mask(e, n) for e in all_edges]
        colorable = _is_2_colorable(masks, n, deadline)
        
        if not colorable:
            # Greedily reduce
            reduced = _greedy_reduce(all_edges, n, deadline)
            
            if len(reduced) < best_count:
                best_edges = reduced
                best_n = n
                best_count = len(reduced)
    
    # Strategy 2: Try random search for small edge counts
    for e in range(10, min(51, best_count)):
        if time.time() > deadline:
            break
        
        for n in range(6, 13):
            if time.time() > deadline:
                break
            
            max_edges = len(list(combinations(range(n), 5)))
            if e > max_edges:
                continue
            
            edges, success = _try_random_hypergraph(n, e, deadline, rng)
            if success:
                best_edges = edges
                best_n = n
                best_count = e
                break  # found one with e edges, go to next e
    
    # If we didn't find anything, fall back to a known construction
    if best_edges is None:
        # Complete 5-uniform hypergraph on 9 vertices
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        best_count = len(best_edges)
    
    return {'n': best_n, 'edges': best_edges}
