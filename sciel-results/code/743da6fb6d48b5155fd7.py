import time
import itertools
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0

def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0  # 5s safety margin

def _is_2_colorable(n, edges):
    """Exact check via backtracking with unit propagation.
    Returns True if a proper 2-colouring exists (every edge has both colours)."""
    # Precompute edge membership as bitmasks for speed
    edge_masks = [0] * len(edges)
    for i, e in enumerate(edges):
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks[i] = m
    k = len(edges)

    # colour[v] in {-1, 0, 1}; 0 = unassigned
    colour = [0] * n
    # active[i] = edge not yet satisfied (not yet seen both colours)
    # We track for each edge: has_red, has_blue
    has_red = [False] * k
    has_blue = [False] * k
    satisfied = [False] * k  # edge already has both colours -> no longer constraining

    # For unit propagation: an edge is "pending" if not satisfied.
    # If an edge has all-but-one of its vertices assigned the same colour and that colour
    # is present, the last vertex is forced to the opposite colour (else monochromatic).

    def propagate():
        # Returns False on contradiction, True otherwise; modifies colour/has/satisfied
        changed = True
        while changed:
            changed = False
            for i in range(k):
                if satisfied[i]:
                    continue
                e = edges[i]
                # count unassigned and colours
                unassigned = []
                red = 0
                blue = 0
                for v in e:
                    cv = colour[v]
                    if cv == 0:
                        unassigned.append(v)
                    elif cv == 1:
                        red += 1
                    else:
                        blue += 1
                # If already both colours, mark satisfied
                if red > 0 and blue > 0:
                    satisfied[i] = True
                    changed = True
                    continue
                # If only one colour present so far and one vertex unassigned -> force opposite
                if len(unassigned) == 1 and red > 0 and blue == 0:
                    v = unassigned[0]
                    if colour[v] == 0:
                        colour[v] = -1
                        changed = True
                        # propagate to other edges containing v
                    else:
                        return False
                elif len(unassigned) == 1 and blue > 0 and red == 0:
                    v = unassigned[0]
                    if colour[v] == 0:
                        colour[v] = 1
                        changed = True
                    else:
                        return False
                # If no colour present yet and all assigned impossible (handled by unassigned count)
        return True

    def backtrack(idx):
        if idx >= n:
            return True  # all assigned, no contradiction -> proper colouring exists
        # choose vertex idx
        for val in (1, -1):
            # save
            old = [colour[idx]]
            old_sat = [satisfied[i] for i in range(k)]
            old_hr = [has_red[i] for i in range(k)]
            old_hb = [has_blue[i] for i in range(k)]
            colour[idx] = val
            if propagate():
                if backtrack(idx + 1):
                    return True
            # restore
            colour[idx] = old[0]
            satisfied[:] = old_sat
            has_red[:] = old_hr
            has_blue[:] = old_hb
        return False

    return backtrack(0)


def _min_edges_for_n(n, max_edges_to_consider, time_budget):
    """Find minimum edge hitting set for n vertices via ILP.
    Returns (num_edges, edges) or None."""
    all_edges = list(combinations(range(n), 5))
    m = len(all_edges)
    # Enumerate colourings: 2^n. For each, compute which edges are monochromatic.
    # To keep memory/time bounded, cap n.
    num_col = 1 << n
    if num_col > 4096:
        return None
    # For each colouring, list of edge indices that are monochromatic.
    # Represent colouring as bitmask: bit v = colour (1=red,0=blue)
    # An edge is monochromatic if all 5 bits equal.
    edge_bitmasks = [0] * m
    for i, e in enumerate(all_edges):
        bm = 0
        for v in e:
            bm |= (1 << v)
        edge_bitmasks[i] = bm

    # Build constraints: for each colouring, the set of monochromatic edges.
    # We need a hitting set: pick edges so every colouring is hit.
    # Build a matrix: rows = colourings, cols = edges; entry=1 if edge mono under colouring.
    # Then min sum x_e s.t. A x >= 1, x in {0,1}.
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    # Build A row by row (sparse-ish). Use dense for small sizes.
    A_rows = []
    for c in range(num_col):
        row = np.zeros(m, dtype=np.int8)
        for i in range(m):
            bm = edge_bitmasks[i]
            # extract 5 bits
            vals = [(c >> v) & 1 for v in range(n) if (bm >> v) & 1]
            if len(set(vals)) <= 1:
                row[i] = 1
        A_rows.append(row)
    A = np.array(A_rows, dtype=float)

    # Some colourings may have zero monochromatic edges? For 5-uniform on n>=5,
    # every colouring has some monochromatic 5-set? No: e.g. n=9, colouring 5 red 4 blue ->
    # red 5-set is mono. Actually by pigeonhole for n>=... not guaranteed for all.
    # If a row is all zeros, that colouring is already "proper" (no mono edge possible),
    # meaning NO edge set can hit it -> hypergraph always 2-colourable. Return None.
    for r in range(num_col):
        if A[r].sum() == 0:
            return None

    c_obj = np.ones(m)
    cons = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    res = milp(c=c_obj, constraints=cons, integrality=np.ones(m), bounds=bounds,
               options={'time_limit': time_budget, 'mip_gap': 0.0})
    if res.status != 0:
        return None
    x = res.x
    sel = [i for i in range(m) if x[i] > 0.5]
    edges = [list(all_edges[i]) for i in sel]
    return (len(edges), edges)


def _verify(n, edges):
    """Return True if hypergraph has NO proper 2-colouring."""
    return not _is_2_colorable(n, edges)


def construct():
    best = None  # (num_edges, n, edges)
    # Try increasing n; the 51-edge optimum is expected around n=10..12.
    # Cap edges considered / colourings for tractability.
    for n in range(9, 14):
        if _remaining() < 10.0:
            break
        budget = max(1.0, min(30.0, _remaining() - 10.0))
        try:
            res = _min_edges_for_n(n, 200, budget)
        except Exception:
            res = None
        if res is not None:
            cnt, edges = res
            if _remaining() < 3.0:
                break
            # Verify with exact backtracking
            if _verify(n, edges):
                if best is None or cnt < best[0]:
                    best = (cnt, n, edges)
                if cnt <= 51:
                    # reached or beat known best; can stop trying larger n for smaller
                    break
    if best is None:
        # Fallback: complete hypergraph on 9 vertices (always valid, 126 edges)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    cnt, n, edges = best
    return {'n': n, 'edges': edges}
