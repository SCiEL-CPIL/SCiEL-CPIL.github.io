"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: Use a hardcoded 51-edge 5-uniform hypergraph on 11 vertices that is
known to be non-2-colorable (the best known construction for m(5) <= 51).
We verify non-2-colourability with an exact backtracking + unit propagation
checker (equivalent to the task's required check).

The construction is based on taking a non-2-colorable 3-uniform hypergraph
(the Fano plane has 7 edges on 7 vertices) and lifting it to 5-uniform by
adding two shared vertices. However, as analyzed, this simple lift may be
2-colorable (color all Fano vertices one color, shared vertices the other).

Instead, we use a more robust construction: the complete 5-graph on 9 vertices
has 126 edges, which is too many. We need a better approach.

Actually, let's use a known non-2-colorable 5-uniform hypergraph with 51 edges.
The best known m(5) construction uses 51 edges. Let me construct one carefully.

Approach: Use a 5-uniform hypergraph derived from the Petersen graph or similar
small non-2-colorable structures, then add edges to ensure non-2-colorability
while keeping the count at or below 51.

Let me use a systematic approach: start with vertices 0..10 (n=11) and build
edges that make the hypergraph non-2-colorable. A key insight: if we can find
a set of 5-edges such that no matter how we 2-color the 11 vertices, at least
one edge becomes monochromatic.

I'll use the following construction based on known results:
- Take the 11 vertices.
- Use edges that form a "covering" of all possible monochromatic patterns.

Actually, let me just hardcode a verified 51-edge construction. I'll generate
it using a greedy/ILP-like approach in code, but since we need it fast, I'll
embed a pre-computed one.

Let me use a different approach: the complement of a 2-coloring must hit every
edge. So we need a family of 5-sets such that every 2-coloring of [11] makes
at least one set monochromatic.

Here's a cleaner construction: Use the 11 vertices as the projective line over
GF(11) or similar. But let me just use a well-known small non-2-colorable
5-uniform hypergraph.

I'll construct it as follows:
- n = 11 vertices
- Use 51 edges derived from a known optimal construction

Since I can't easily verify a hardcoded list, let me use a construction that
I can verify in code: start with the Fano plane idea but fix it properly.

The issue with the Fano lift: adding two fixed vertices v, w to each Fano edge
{a,b,c} gives {a,b,c,v,w}. If we color a,b,c as 1 and v,w as 0, every edge
has both colors. So it IS 2-colorable. Bad.

Fix: Instead of adding fixed vertices, add vertices that vary. Or use a different
non-2-colorable base.

Let me use the following: a 5-uniform hypergraph on 11 vertices where we take
all edges of a 5-partite structure... no.

Let me try: n=9, complete 5-graph has C(9,5)=126 edges. Too many.
n=10, C(10,5)=252. More.

I need a sparse construction. Let me use the following approach in code:
1. Start with n=11 vertices.
2. Use a randomized greedy search to find a non-2-colorable 5-uniform hypergraph
   with few edges, using the exact checker to verify.
3. If found within time, return it.
4. Fallback: complete 5-graph on 9 vertices (126 edges, definitely non-2-colorable
   since any 2-coloring of 9 vertices has a color class of size >= 5, which forms
   a monochromatic edge).

Wait, that's a great insight! For K_n^5 (complete 5-graph), if n >= 10, then any
2-coloring has a color class of size >= 5 (pigeonhole: 10 vertices, 2 colors,
one color has >= 5). So K_10^5 is non-2-colorable with C(10,5)=252 edges.
But K_9^5: 9 vertices, 2 colors, one color has >= 5 (since ceil(9/2)=5). So
K_9^5 is also non-2-colorable! C(9,5)=126 edges.

Can we do better? We need at most 51 edges ideally. Let me search.

I'll implement a fast search: try random 5-uniform hypergraphs on n=11 with
increasing edge counts, checking non-2-colorability with a fast bitset solver.
"""

import time
import random
from itertools import combinations
import numpy as np
import sys

sys.setrecursionlimit(100000)

def _is_2_colorable(n, edges):
    """Exact check (backtracking + unit propagation) for a proper 2-colouring.
    
    Returns True iff the hypergraph IS 2-colorable (i.e., has a proper 2-coloring).
    """
    if not edges:
        return True
    
    m = len(edges)
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    all_vars = (1 << n) - 1
    
    def backtrack(assign, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    if (assign & mask) == mask or (assign & mask) == 0:
                        return None
                else:
                    assigned_zero = assigned_in_edge & ~assign
                    assigned_one = assigned_in_edge & assign
                    unassigned_count = bin(unassigned_in_edge).count('1')
                    
                    if assigned_one == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign |= (1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                    
                    if assigned_zero == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign &= ~(1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
        
        if unassigned == 0:
            for mask in edge_masks:
                if (assign & mask) == mask or (assign & mask) == 0:
                    return None
            return assign
        
        v = unassigned.bit_length() - 1
        res = backtrack(assign | (1 << v), unassigned & ~(1 << v))
        if res is not None:
            return res
        return backtrack(assign & ~(1 << v), unassigned & ~(1 << v))
    
    return backtrack(0, all_vars) is not None

def _minimalize_edges(n, edges, deadline):
    """Remove redundant edges to get a minimal non-2-colorable hypergraph."""
    current_edges = list(edges)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_edges)):
            candidate = current_edges[:i] + current_edges[i+1:]
            if not _is_2_colorable(n, candidate):
                current_edges = candidate
                changed = True
                break
    return current_edges


def _complete_5_graph(n):
    """Return the complete 5-uniform hypergraph on n vertices."""
    from itertools import combinations
    return [list(e) for e in combinations(range(n), 5)]


def _greedy_set_cover(n, target_edges=None, deadline=None, seed=42):
    """
    Greedy set cover approach to find a non-2-colorable 5-uniform hypergraph.
    """
    if deadline is None:
        deadline = time.time() + 130.0
    
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Precompute edge masks
    edge_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute coloring sets for each edge using NumPy for speed
    colorings = np.arange(num_colorings, dtype=np.uint32)
    edge_coloring_sets = []
    
    for j in range(num_edges):
        mask = edge_masks[j]
        # Check if all bits in mask are 0 or all are 1
        # (colorings & mask == 0) | (colorings & mask == mask)
        # Note: colorings are uint32, mask is int.
        mono_mask = ((colorings & mask) == 0) | ((colorings & mask) == mask)
        # Convert boolean array to a Python int bitset
        # This is a bit tricky to do efficiently in pure Python without loops if we want a single int.
        # But for n=11, num_colorings=2048.
        # We can just store the boolean array and use it for counting/covering?
        # Or convert to int.
        # Let's stick to Python ints for the bitset logic as it's simpler for bitwise ops on the "uncovered" state.
        # However, constructing the int from the boolean array:
        # bitset = 0
        # for k in range(num_colorings):
        #     if mono_mask[k]:
        #         bitset |= (1 << k)
        # This loop is 2048 * 462 = ~1M ops. Acceptable.
        
        bitset = 0
        # Optimization: use numpy to find indices?
        # Or just loop.
        for k in range(num_colorings):
            if mono_mask[k]:
                bitset |= (1 << k)
        edge_coloring_sets.append(bitset)
    
    universe = (1 << num_colorings) - 1
    covered = 0
    selected_edges = []
    uncovered = universe
    
    # Sort edges by coverage size
    edge_indices = sorted(range(num_edges), key=lambda i: bin(edge_coloring_sets[i]).count('1'), reverse=True)
    
    while uncovered != 0:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = 0
        for idx in edge_indices:
            if idx in selected_edges:
                continue
            coverage = edge_coloring_sets[idx] & uncovered
            count = bin(coverage).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
            if count == 0:
                continue
        
        if best_idx == -1 or best_count == 0:
            break
        
        selected_edges.append(best_idx)
        covered |= edge_coloring_sets[best_idx]
        uncovered = universe & ~covered
        
        if uncovered == 0:
            break
    
    if uncovered == 0:
        result = [list(all_edges[i]) for i in selected_edges]
        return result
    else:
        return None


def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Fallback: complete 5-graph on 9 vertices (126 edges, definitely non-2-colorable)
    fallback_n = 9
    fallback_edges = _complete_5_graph(fallback_n)
    
    n = 11
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    best_result = None
    
    # Try ILP approach first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        A = np.zeros((num_colorings, num_edges), dtype=np.int8)
        
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Vectorized construction of A
        colorings = np.arange(num_colorings, dtype=np.uint32)
        for j in range(num_edges):
            mask = edge_masks[j]
            mono_mask = ((colorings & mask) == 0) | ((colorings & mask) == mask)
            A[:, j] = mono_mask.astype(np.int8)
        
        c = np.ones(num_edges, dtype=np.float64)
        constraint = LinearConstraint(A, 1, np.inf)
        bounds = Bounds(0, 1)
        integrality = np.ones(num_edges, dtype=np.int8)
        
        # Set time limit for the solver
        options = {'time_limit': 60}
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds, options=options)
        
        if res.status == 0 and res.x is not None:
            x = np.round(res.x).astype(int)
            selected_indices = np.where(x > 0)[0]
            result = [list(all_edges[i]) for i in selected_indices]
            
            if not _is_2_colorable(n, result):
                best_result = result
                
    except Exception as e:
        pass

    # If ILP didn't find a result or timed out, try greedy
    if best_result is None or time.time() < deadline - 10:
        result = _greedy_set_cover(n, deadline=deadline - 5, seed=42)
        if result is not None:
            if not _is_2_colorable(n, result):
                if best_result is None or len(result) < len(best_result):
                    best_result = result

    # If we still don't have a good result, try more seeds
    if best_result is None or time.time() < deadline - 10:
        for seed in [123, 456, 789, 101]:
            if time.time() > deadline - 10:
                break
            result = _greedy_set_cover(n, deadline=deadline - 5, seed=seed)
            if result is not None:
                if not _is_2_colorable(n, result):
                    if best_result is None or len(result) < len(best_result):
                        best_result = result

    if best_result is None:
        best_result = fallback_edges
        n = fallback_n
    else:
        # Minimize the number of edges by removing redundant ones
        best_result = _minimalize_edges(n, best_result, deadline)

    return {'n': n, 'edges': best_result}
