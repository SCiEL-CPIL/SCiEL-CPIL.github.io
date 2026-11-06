"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""

import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0 - 5.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _edge_to_mask(edge):
    m = 0
    for v in edge:
        m |= (1 << v)
    return m

def _is_2_colorable(n, masks):
    """Exact check via backtracking + unit propagation.
    Returns True if a 2-coloring exists with no monochromatic edge.
    We search for an assignment where every edge has both colors.
    Equivalent: no edge is monochromatic.
    """
    # We'll use a recursive DPLL-style search.
    # State: color[v] in {-1, 0, 1}. 
    # An edge is 'bad' if all its vertices are colored and all same color.
    # An edge is 'forced' if all but one vertex are colored the same color -> last must be opposite.
    # If all vertices of an edge are colored and not all same -> edge satisfied.
    
    color = [-1] * n
    # Precompute for each edge its vertex list (as tuple of ints)
    edges = []
    for m in masks:
        vs = []
        for v in range(n):
            if (m >> v) & 1:
                vs.append(v)
        edges.append(vs)
    
    # Order edges for propagation: we'll just iterate all each time (n small)
    
    def unit_propagate():
        """Return (ok, color). Apply unit propagation: if an edge has all but one
        colored and those are same color, force the last to opposite.
        If an edge becomes monochromatic -> conflict (ok=False).
        Iterate until fixpoint."""
        changed = True
        while changed:
            changed = False
            for vs in edges:
                # count colors
                c0 = c1 = un = 0
                last_un = -1
                for v in vs:
                    c = color[v]
                    if c == 0:
                        c0 += 1
                    elif c == 1:
                        c1 += 1
                    else:
                        un += 1
                        last_un = v
                if un == 0:
                    # all colored: must be mixed
                    if c0 == 0 or c1 == 0:
                        return False, color
                elif un == 1:
                    # all but one colored
                    if c0 == 0:
                        # all colored are 1 -> last must be 0
                        if color[last_un] == -1:
                            color[last_un] = 0
                            changed = True
                        elif color[last_un] == 1:
                            return False, color
                    elif c1 == 0:
                        if color[last_un] == -1:
                            color[last_un] = 1
                            changed = True
                        elif color[last_un] == 0:
                            return False, color
                    # else both colors present -> no constraint
        return True, color
    
    def search():
        # Find an uncolored vertex; if none, check all edges satisfied
        # Use heuristic: pick vertex in an edge closest to monochromatic
        # Simple: pick first uncolored
        # Better: pick vertex with max 'pressure'
        best_v = -1
        best_score = -1
        for v in range(n):
            if color[v] == -1:
                # score = number of edges where v is the unique uncolored and rest same color
                score = 0
                for vs in edges:
                    if v in vs:
                        c0 = c1 = 0
                        for u in vs:
                            if u == v: continue
                            cu = color[u]
                            if cu == 0: c0 += 1
                            elif cu == 1: c1 += 1
                        if c0 == 0 or c1 == 0:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_v = v
        if best_v == -1:
            # all colored: verify no monochromatic edge
            for vs in edges:
                c0 = c1 = 0
                for v in vs:
                    if color[v] == 0: c0 += 1
                    else: c1 += 1
                if c0 == 0 or c1 == 0:
                    return False
            return True
        # Try both colors
        for c in (0, 1):
            color[best_v] = c
            ok, color = unit_propagate()
            if ok and search():
                return True
        color[best_v] = -1
        return False
    
    return search()

def _generate_complete(n=9, r=5):
    edges = []
    for combo in combinations(range(n), r):
        edges.append(list(combo))
    return n, edges

def _try_reduce(n, edges, seed):
    """Try to reduce edge count via randomized greedy deletion."""
    rng = random.Random(seed)
    masks = [_edge_to_mask(e) for e in edges]
    # Verify starting point is non-2-colorable
    if _is_2_colorable(n, masks):
        return None
    best = len(masks)
    best_masks = list(masks)
    # Greedy: try to remove each edge (in random order) if still non-2-colorable
    # Repeat passes
    for _pass in range(50):
        if _time_left() < 2.0:
            break
        order = list(range(len(masks)))
        rng.shuffle(order)
        for i in order:
            if _time_left() < 1.0:
                break
            if i >= len(masks):
                continue
            # Try removing edge i
            test = masks[:i] + masks[i+1:]
            if not _is_2_colorable(n, test):
                # Still non-2-colorable without it
                masks = test
                best = len(masks)
                best_masks = list(masks)
                # restart order since indices shifted
                order = list(range(len(masks)))
                rng.shuffle(order)
    # Also try local search: swap one edge for another? Keep simple.
    if best < len(edges):
        # Reconstruct edges from masks
        new_edges = []
        for m in best_masks:
            vs = []
            for v in range(n):
                if (m >> v) & 1:
                    vs.append(v)
            new_edges.append(vs)
        return new_edges
    return None

def construct():
    # Start from complete 5-graph on 9 vertices (known non-2-colorable)
    n, edges = _generate_complete(9, 5)
    best_n, best_edges = n, edges
    # Try reduction with several seeds
    t0 = time.time()
    seed = 0
    while _time_left() > 5.0:
        if seed % 5 == 0:
            # periodic check
            pass
        res = _try_reduce(n, edges, seed)
        if res is not None and len(res) < len(best_edges):
            best_edges = res
            best_n = n
        seed += 1
        if seed > 200:
            break
    # Also try starting from complete on 10 vertices and reducing
    if _time_left() > 20.0:
        n2, edges2 = _generate_complete(10, 5)
        seed = 1000
        while _time_left() > 10.0 and seed < 1050:
            res = _try_reduce(n2, edges2, seed)
            if res is not None and len(res) < len(best_edges):
                best_edges = res
                best_n = n2
            seed += 1
    return {'n': best_n, 'edges': best_edges}
