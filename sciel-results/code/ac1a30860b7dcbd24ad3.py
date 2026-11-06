import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin under 120s

def _is_2_colorable(n, edges):
    """
    Exact check: returns True if the 5-uniform hypergraph on vertices 0..n-1
    with the given edges has a proper 2-colouring (no monochromatic edge).
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True

    # Pre-compute for each vertex, the list of edges it appears in
    vert_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(ei)

    # Assignment: -1 = unassigned, 0 or 1 = colour
    assignment = [-1] * n

    def propagate(assignment, edges, n):
        """
        Unit propagation:
        For each edge, if 4 vertices have the same colour, the 5th must be different.
        If an edge has all 5 vertices assigned and all same colour -> conflict.
        Returns (success, new_assignment).
        """
        changed = True
        while changed:
            changed = False
            for ei, e in enumerate(edges):
                vals = [assignment[v] for v in e]
                # Check for conflict: all assigned and all same
                if all(v != -1 for v in vals):
                    if len(set(vals)) == 1:
                        return False, assignment
                    continue
                # Unit clause: if 4 are same colour c, the 5th must be 1-c
                count0 = sum(1 for v in vals if v == 0)
                count1 = sum(1 for v in vals if v == 1)
                unassigned = [v for v, a in zip(e, vals) if a == -1]
                if count0 == 4 and len(unassigned) == 1:
                    new_val = 1
                    if assignment[unassigned[0]] != -1 and assignment[unassigned[0]] != new_val:
                        return False, assignment
                    assignment[unassigned[0]] = new_val
                    changed = True
                elif count1 == 4 and len(unassigned) == 1:
                    new_val = 0
                    if assignment[unassigned[0]] != -1 and assignment[unassigned[0]] != new_val:
                        return False, assignment
                    assignment[unassigned[0]] = new_val
                    changed = True
        return True, assignment

    def backtrack(assignment, edges, n):
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT:
            return True  # Assume colourable to be safe

        # Propagate
        ok, assignment = propagate(assignment, edges, n)
        if not ok:
            return False  # Conflict: this branch fails

        # Find unassigned vertex with most constraints
        unassigned = [i for i in range(n) if assignment[i] == -1]
        if not unassigned:
            # All assigned, check all edges satisfied
            for e in edges:
                vals = set(assignment[v] for v in e)
                if len(vals) == 1:
                    return False  # Conflict
            return True  # All satisfied -> colouring exists

        # Choose vertex with most edges
        best_v = max(unassigned, key=lambda v: len(vert_edges[v]))

        # Try both colours
        for c in [0, 1]:
            new_assign = assignment[:]
            new_assign[best_v] = c
            if backtrack(new_assign, edges, n):
                return True
        return False

    return backtrack(assignment, edges, n)


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    # Base: complete 5-uniform hypergraph on 9 vertices (126 edges)
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    edges = list(all_edges)  # mutable list

    # Greedy removal: try to remove each edge if the hypergraph remains non-2-colourable
    # Process edges in a deterministic order
    remaining = set(range(len(edges)))
    removed = set()

    for idx in range(len(edges)):
        if idx in removed:
            continue
        if time.time() > deadline - 5:
            break

        # Try removing this edge
        candidate_edges = [edges[i] for i in remaining if i != idx]

        # Quick check: if candidate_edges is empty, removing leaves 0 edges -> 2-colourable
        if not candidate_edges:
            continue

        # Check if still non-2-colourable
        if not _is_2_colorable(n, candidate_edges):
            removed.add(idx)

    final_edges = [edges[i] for i in range(len(edges)) if i not in removed]

    # Verify the result is valid
    assert 5 <= n <= 26
    assert len(final_edges) <= 200
    assert len(set(tuple(sorted(e)) for e in final_edges)) == len(final_edges)
    for e in final_edges:
        assert len(e) == 5
        assert len(set(e)) == 5
        assert all(0 <= v < n for v in e)

    # Final verification that it's NOT 2-colourable
    if _is_2_colorable(n, final_edges):
        # Fallback: return the complete graph
        return {'n': n, 'edges': all_edges}

    return {'n': n, 'edges': final_edges}
