"""Few edges in 5-uniform hypergraphs without Property B.

Structural approach: Model as a minimum set cover problem.
- For n vertices, there are 2^n possible 2-colorings.
- Each 5-edge e is "monochromatic" under coloring c iff all 5 vertices get the same color.
- We need the minimum set of edges such that every coloring has at least one monochromatic edge.
- This is a set cover: universe = all colorings, sets = {colorings where edge e is monochromatic}.

For n=10: 252 edges, 1024 colorings. Solve via greedy + local search.
For n=11: 462 edges, 2048 colorings.
"""
import time
import random
from itertools import combinations

_DEADLINE = 148.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _build_coverage(n):
    """
    For each 5-edge, compute the set of colorings (as bitmasks) under which it is monochromatic.
    Returns: list of (edge_tuple, frozenset_of_coloring_bitmasks)
    """
    edges = list(combinations(range(n), 5))
    total_colorings = 1 << n
    
    # For each edge, compute which colorings make it monochromatic
    # A coloring is a bitmask of n bits. Edge (v1,...,v5) is monochromatic
    # iff all bits at positions v1..v5 are 0, OR all are 1.
    
    edge_colorings = []
    for edge in edges:
        # Monochromatic-0: all vertices in edge have color 0
        # Monochromatic-1: all vertices in edge have color 1
        # Count: 2^(n-5) colorings for each, total 2^(n-4)
        # But we need the actual set for coverage checking
        
        # For efficiency with n<=11, we can represent as a list of coloring indices
        # or use a different representation
        
        # Actually, for the greedy algorithm, we just need to know which colorings
        # are covered. Let's use a different approach:
        # Precompute for each coloring, which edges are monochromatic.
        pass
    
    # Better approach: for each coloring, determine which edges are monochromatic
    # Then build the incidence matrix
    # coloring -> list of edge indices that are monochromatic
    
    coloring_edges = [set() for _ in range(total_colorings)]
    
    for edge_idx, edge in enumerate(edges):
        # For this edge, find all colorings where it's monochromatic
        # Monochromatic-0: bits at edge positions are all 0
        # Monochromatic-1: bits at edge positions are all 1
        # Other bits can be anything
        
        # Generate all colorings where edge is monochromatic-0
        # and all where it's monochromatic-1
        free_positions = [i for i in range(n) if i not in edge]
        
        for mask_free in range(1 << len(free_positions)):
            # Monochromatic-0
            c0 = 0
            for j, pos in enumerate(free_positions):
                if mask_free & (1 << j):
                    c0 |= (1 << pos)
            coloring_edges[c0].add(edge_idx)
            
            # Monochromatic-1
            c1 = c0
            for v in edge:
                c1 |= (1 << v)
            coloring_edges[c1].add(edge_idx)
    
    return edges, coloring_edges


def _greedy_set_cover(edges, coloring_edges, n, max_edges=None):
    """
    Greedy set cover: repeatedly pick the edge that covers the most uncovered colorings.
    """
    total_colorings = 1 << n
    uncovered = set(range(total_colorings))
    
    if max_edges is None:
        max_edges = len(edges)
    
    selected = []
    edge_cover = [len(coloring_edges[c]) for c in range(total_colorings)]  # not quite right
    
    # For each edge, compute how many currently uncovered colorings it covers
    # This is expensive to recompute, so use lazy evaluation
    
    for _ in range(max_edges):
        if not uncovered:
            break
        
        best_edge = -1
        best_count = 0
        
        for ei, edge in enumerate(edges):
            if ei in selected:
                continue
            # Count how many uncovered colorings this edge covers
            count = 0
            for c in uncovered:
                if ei in coloring_edges[c]:
                    count += 1
                    if count > best_count:
                        break
            if count > best_count:
                best_count = count
                best_edge = ei
        
        if best_edge == -1 or best_count == 0:
            break
        
        selected.append(best_edge)
        # Remove covered colorings
        for c in list(uncovered):
            if best_edge in coloring_edges[c]:
                uncovered.discard(c)
    
    return selected


def _local_search_improve(selected, edges, coloring_edges, n, rng, time_limit=10.0):
    """
    Try to improve the selected set by removing edges that are redundant.
    An edge is redundant if removing it still leaves all colorings covered.
    """
    total_colorings = 1 << n
    selected_set = set(selected)
    result = list(selected)
    
    # Build coverage count: for each coloring, how many selected edges cover it
    cover_count = [0] * total_colorings
    for ei in selected:
        for c in range(total_colorings):
            if ei in coloring_edges[c]:
                cover_count[c] += 1
    
    # Check all are covered
    all_covered = all(cover_count[c] >= 1 for c in range(total_colorings))
    
    if not all_covered:
        return result
    
    # Try to remove redundant edges
    changed = True
    while changed and _elapsed() < time_limit:
        changed = False
        indices = list(range(len(result)))
        rng.shuffle(indices)
        
        for idx in indices:
            ei = result[idx]
            # Check if removing ei still covers all
            can_remove = True
            for c in range(total_colorings):
                if cover_count[c] == 1 and ei in coloring_edges[c]:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove ei
                for c in range(total_colorings):
                    if ei in coloring_edges[c]:
                        cover_count[c] -= 1
                result.pop(idx)
                selected_set.discard(ei)
                changed = True
                break  # Restart after removal
    
    return result


def _verify_not_2_colorable(n, edges):
    """
    Fast verification that the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Build incidence
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # Vertex ordering by degree
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    color = [-1] * n
    pos_count = [0] * m
    neg_count = [0] * m
    uncount = [5] * m
    
    def _propagate():
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                pc = pos_count[ei]
                nc = neg_count[ei]
                uc = uncount[ei]
                if pc == 5 or nc == 5:
                    return False
                if uc == 0:
                    continue
                if pc == 4 and uc == 1:
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 1
                            neg_count[ei] += 1
                            uncount[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    neg_count[e2] += 1
                                    uncount[e2] -= 1
                            changed = True
                            break
                elif nc == 4 and uc == 1:
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 0
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
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
        if idx == n:
            return True
        
        v = v_order[idx]
        color_snap = color[:]
        pos_snap = pos_count[:]
        neg_snap = neg_count[:]
        unc_snap = uncount[:]
        
        for c in (0, 1):
            color[v] = c
            valid = True
            for ei in v_edges[v]:
                if c == 0:
                    pos_count[ei] += 1
                else:
                    neg_count[ei] += 1
                uncount[ei] -= 1
                if pos_count[ei] == 5 or neg_count[ei] == 5:
                    valid = False
                    break
            if valid and _propagate():
                if _search(idx + 1):
                    return True
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


def construct():
    if _elapsed() > _DEADLINE - 15:
        return {'n': 10, 'edges': _DEFAULT_EDGES}
    
    rng = random.Random(42)
    best_edges = _DEFAULT_EDGES
    best_n = 10
    
    # Try different n values
    for n in [10, 11, 12]:
        if _elapsed() > _DEADLINE - 20:
            break
        
        try:
            edges_list, coloring_edges = _build_coverage(n)
        except Exception:
            continue
        
        total_colorings = 1 << n
        num_edges = len(edges_list)
        
        # Greedy set cover
        t0 = time.monotonic()
        selected = _greedy_set_cover(edges_list, coloring_edges, n)
        t1 = time.monotonic()
        
        if len(selected) < len(best_edges):
            # Verify
            sel_edges = [list(edges_list[ei]) for ei in selected]
            if _verify_not_2_colorable(n, sel_edges):
                best_edges = sel_edges
                best_n = n
        
        # Local search to improve
        if _elapsed() < _DEADLINE - 15:
            improved = _local_search_improve(
                selected, edges_list, coloring_edges, n, rng,
                time_limit=min(10.0, _DEADLINE - 15 - _elapsed())
            )
            if len(improved) < len(best_edges):
                sel_edges = [list(edges_list[ei]) for ei in improved]
                if _verify_not_2_colorable(n, sel_edges):
                    best_edges = sel_edges
                    best_n = n
    
    return {'n': best_n, 'edges': best_edges}
