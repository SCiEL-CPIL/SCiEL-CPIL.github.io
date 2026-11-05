"""Few edges in 5-uniform hypergraphs without Property B.

Strategy:
1. Start with the known 51-edge construction (even 5-subsets of 10 vertices).
2. Use a fast 2-SAT-style backtracking verifier (forbid monochromatic edges).
3. Local search: greedily delete edges while maintaining non-2-colorability.
4. Try to find smaller constructions via randomized search on 10-14 vertices.
"""
import time
import random
from itertools import combinations

_DEADLINE = 148.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _not_2_colorable(n, edges):
    """
    Check if the hypergraph is NOT 2-colorable using backtracking with unit propagation.
    Returns True if no proper 2-coloring exists.
    
    For each edge, we forbid all 5 vertices being the same color.
    This is equivalent to: for each edge e={v1..v5}, at least one vertex has color 0 AND at least one has color 1.
    Which means: NOT(all 0) AND NOT(all 1).
    
    We model this as a constraint satisfaction problem and use backtracking with unit propagation.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Build incidence: for each vertex, list of edges containing it
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # Vertex ordering by degree (higher degree first)
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    # State:
    # color[v] = -1 (unassigned), 0, or 1
    # For each edge, track how many vertices are 0, how many are 1, how many unassigned
    color = [-1] * n
    pos_count = [0] * m   # count of vertices with color 1 in each edge
    neg_count = [0] * m   # count of vertices with color 0 in each edge
    uncount = [5] * m     # count of unassigned vertices in each edge
    
    def _propagate():
        """Unit propagation: if an edge has 4 of one color and 1 unassigned, force the last."""
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                pc = pos_count[ei]
                nc = neg_count[ei]
                uc = uncount[ei]
                if pc == 5 or nc == 5:
                    return False  # monochromatic edge found - contradiction
                if uc == 0:
                    continue
                if pc == 4 and uc == 1:
                    # Must have at least one 0, so the unassigned must be 0
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 0
                            neg_count[ei] += 1
                            uncount[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    neg_count[e2] += 1
                                    uncount[e2] -= 1
                            changed = True
                            break
                elif nc == 4 and uc == 1:
                    # Must have at least one 1, so the unassigned must be 1
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 1
                            pos_count[ei] += 1
                            uncount[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    pos_count[e2] += 1
                                    uncount[e2] -= 1
                            changed = True
                            break
        return True
    
    def _search(idx):
        # Skip already assigned vertices
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
        if idx == n:
            return True  # All assigned, no contradiction
        
        v = v_order[idx]
        
        # Save state for backtracking
        color_snap = color[:]
        pos_snap = pos_count[:]
        neg_snap = neg_count[:]
        unc_snap = uncount[:]
        
        # Try color 0
        color[v] = 0
        valid = True
        for ei in v_edges[v]:
            neg_count[ei] += 1
            uncount[ei] -= 1
            if neg_count[ei] == 5:
                valid = False
                break
        if valid and _propagate():
            if _search(idx + 1):
                return True
        # Restore
        color[:] = color_snap
        pos_count[:] = pos_snap
        neg_count[:] = neg_snap
        uncount[:] = unc_snap
        
        # Try color 1
        color[v] = 1
        valid = True
        for ei in v_edges[v]:
            pos_count[ei] += 1
            uncount[ei] -= 1
            if pos_count[ei] == 5:
                valid = False
                break
        if valid and _propagate():
            if _search(idx + 1):
                return True
        # Restore
        color[:] = color_snap
        pos_count[:] = pos_snap
        neg_count[:] = neg_snap
        uncount[:] = unc_snap
        
        return False
    
    return not _search(0)


def _known_51_edges():
    """The known 51-edge construction: even 5-subsets of 10 vertices."""
    evenset = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in evenset) % 2 == 0:
            edges.append(list(c))
    return edges


_DEFAULT_EDGES = _known_51_edges()


def _try_remove_edges(n, edges, rng, time_limit=5.0):
    """
    Try to remove edges while maintaining non-2-colorability.
    Returns the reduced edge list if successful.
    """
    current = list(edges)
    improved = True
    t_start = time.monotonic()
    
    while improved and (time.monotonic() - t_start) < time_limit:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if (time.monotonic() - t_start) >= time_limit:
                break
            # Try removing edge at idx
            candidate = current[:idx] + current[idx+1:]
            if _not_2_colorable(n, candidate):
                current = candidate
                improved = True
                break
    
    return current


def _random_search(n, target_edges, rng, time_limit=5.0):
    """
    Randomly sample hypergraphs and check if they are not 2-colorable.
    """
    t_start = time.monotonic()
    all_5sets = list(combinations(range(n), 5))
    
    for _ in range(1000):
        if (time.monotonic() - t_start) >= time_limit:
            break
        # Sample random edges
        edges = [list(e) for e in rng.sample(all_5sets, target_edges)]
        if _not_2_colorable(n, edges):
            return edges
    return None


def construct():
    if _elapsed() > _DEADLINE - 15:
        return {'n': 10, 'edges': _DEFAULT_EDGES}
    
    rng = random.Random(42)
    best_edges = _DEFAULT_EDGES
    best_n = 10
    
    # First, try to reduce the 51-edge construction
    t0 = time.monotonic()
    reduced = _try_remove_edges(10, _DEFAULT_EDGES, rng, time_limit=min(30.0, _DEADLINE - 30 - _elapsed()))
    if len(reduced) < len(best_edges):
        best_edges = reduced
        best_n = 10
    
    # Try random search on different n values
    for n in [10, 11, 12, 13, 14]:
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Try to find a smaller hypergraph
        for target in range(29, len(best_edges)):
            if _elapsed() > _DEADLINE - 20:
                break
            
            t0 = time.monotonic()
            result = _random_search(n, target, rng, time_limit=min(10.0, _DEADLINE - 25 - _elapsed()))
            if result is not None:
                best_edges = result
                best_n = n
                break
    
    return {'n': best_n, 'edges': best_edges}
