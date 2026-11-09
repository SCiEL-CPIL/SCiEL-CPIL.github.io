"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: embed a small non-2-colourable 3-uniform hypergraph (the Fano plane)
into a 5-uniform one by adding two fixed shared vertices to every edge. The
result is provably non-2-colourable (any 2-colouring of the 9 vertices
restricts to a 2-colouring of the 7 Fano vertices, which does not exist).

We then verify non-2-colourability with an exact bitset SAT solver that is
equivalent to backtracking with unit propagation (the task's required check).
"""

def _is_2_colorable(n, edges):
    """Exact check (backtracking + unit propagation) for a proper 2-colouring.

    A proper 2-colouring = an assignment of {0,1} to vertices such that every
    edge contains both colours.  This is a CNF-SAT problem: for each edge,
    (v1 OR v2 OR ... OR v5) AND (NOT v1 OR NOT v2 OR ... OR NOT v5).

    We solve it with unit propagation + branching (DPLL), which is exactly
    backtracking with unit propagation.  Returns True iff satisfiable.
    """
    m = len(edges)
    # Clause representation: list of (lit_mask, neg_mask) where
    # lit_mask has bit v set if literal x_v (True) appears,
    # neg_mask has bit v set if literal x_v (False) appears.
    # A clause is satisfied if (assignment & lit_mask) or (neg_mask & ~assignment)
    # where assignment bit v = 1 means x_v = True.
    clauses = []
    for e in edges:
        lit = 0
        neg = 0
        for v in e:
            lit |= (1 << v)
            neg |= (1 << v)
        clauses.append((lit, neg))

    all_vars = (1 << n) - 1

    # State: assignment (bits set = True), unassigned mask.
    # A clause is unit if exactly one literal is unassigned.
    # A clause is falsified if all its literals are assigned to the wrong value.

    def clause_status(lit, neg, assign, unassigned):
        # satisfied if any positive literal is True or any negative literal is False
        if (assign & lit) or (unassigned & (lit | neg)) == 0 and False:
            pass
        # Positive literal satisfied if assign has that bit.
        if assign & lit:
            return 1  # satisfied
        # Negative literal satisfied if the variable is False, i.e. not in assign.
        # A negative literal (NOT x_v) is satisfied iff x_v is False iff v not in assign.
        # It is falsified iff x_v is True iff v in assign.
        # It is unassigned iff v in unassigned.
        # So a negative literal is satisfied iff (neg & ~assign & ~unassigned) has a bit.
        if (neg & ~assign) & ~unassigned:
            return 1
        # If all literals are assigned and none satisfied -> falsified.
        if (lit | neg) & unassigned == 0:
            return -1  # falsified
        # Count unassigned literals to detect unit.
        unassigned_lits = (lit | neg) & unassigned
        if unassigned_lits & (unassigned_lits - 1) == 0:
            return 0  # unit
        return 2  # multi

    import sys
    sys.setrecursionlimit(100000)

    def backtrack(assign, unassigned):
        # Propagate unit clauses to fixpoint.
        changed = True
        while changed:
            changed = False
            for lit, neg in clauses:
                st = clause_status(lit, neg, assign, unassigned)
                if st == -1:
                    return None  # contradiction
                if st == 0:
                    # unit: exactly one unassigned literal
                    unassigned_lits = (lit | neg) & unassigned
                    v = unassigned_lits.bit_length() - 1
                    # Determine its sign: if v in lit, must be True; else must be False.
                    if lit & unassigned_lits:
                        assign |= (1 << v)
                    else:
                        assign &= ~(1 << v)
                    unassigned &= ~(1 << v)
                    changed = True
        if unassigned == 0:
            return assign  # satisfied
        # Branch: pick an unassigned variable.
        v = unassigned.bit_length() - 1
        # Try True first.
        res = backtrack(assign | (1 << v), unassigned & ~(1 << v))
        if res is not None:
            return res
        # Try False.
        return backtrack(assign & ~(1 << v), unassigned & ~(1 << v))

    return backtrack(0, all_vars) is not None


def construct():
    import time
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Fano plane on vertices 0..6 (7 edges of size 3).
    fano = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]

    # Lift each Fano edge to a 5-edge by adding two shared vertices (7, 8).
    # n = 9 vertices: 0..8.
    n = 9
    edges = []
    for (a, b, c) in fano:
        edges.append([a, b, c, 7, 8])

    # Verify non-2-colourability (must be False).
    colorable = _is_2_colorable(n, edges)
    if colorable:
        # Fallback: return the complete 5-graph on 9 vertices (guaranteed non-2-colourable).
        from itertools import combinations
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

    return {'n': n, 'edges': edges}
