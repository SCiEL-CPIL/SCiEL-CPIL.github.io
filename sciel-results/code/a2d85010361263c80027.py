"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}


def _matvec2(mat, v):
    """Compute mat @ v for a list-of-int (0/1) matrix and int bitmask v.
    Returns an int bitmask of rows having any of the set columns."""
    out = 0
    if v == 0:
        return 0
    for r, row in enumerate(mat):
        if row & v:
            out |= (1 << r)
    return out


def _diameter_leq2_ok(mat):
    """True iff graph is connected and every pair is at distance <= 2."""
    n = len(mat)
    allmask = (1 << n) - 1
    # connected (BFS once)
    comp = 1
    frontier = 1
    while frontier:
        nxt = _matvec2(mat, frontier) & ~comp
        comp |= nxt
        frontier = nxt
    if comp != allmask:
        return False
    # distance > 2 means exists pair (i,j) non-adjacent with no common neighbor
    for i in range(n):
        A = mat[i]
        N2 = _matvec2(mat, A)
        # i cannot reach any vertex that is neither adjacent nor a distance-2 neighbor
        if (A | N2) != allmask:
            return False
    return True


def _edge_critical(mat, u, v):
    """True iff removing edge (u,v) makes the graph disconnected or diameter > 2.
    
    After removing (u,v), a new distance-3+ pair (a,b) must satisfy:
    - a was adjacent to u (and not to v, else a~v still gives dist(a,b)<=2 via v)
    - b was adjacent to v (and not to u, else b~u still gives dist(a,b)<=2 via u)
    - a and b are not adjacent
    - a and b have no common neighbor in the graph WITHOUT edge (u,v)
    
    In the original graph, a and b may share u or v as common neighbors.
    After removing (u,v): u is still a neighbor of a but not of b (since b~u would mean
    b is adjacent to u, which we excluded). Wait, let me reconsider.
    
    Actually: a~u, b~v. After removing (u,v):
    - u is still adjacent to a, but u is no longer adjacent to b (edge removed)
    - v is still adjacent to b, but v is no longer adjacent to u (edge removed)
    
    For a and b to be at distance >= 3 after removal:
    - a and b not adjacent
    - No common neighbor w in G \\ {(u,v)}
    
    In G, common neighbors of a and b include potentially u (if b~u) and v (if a~v).
    We excluded b~u and a~v, so u and v are NOT common neighbors of a and b in G either.
    Wait, that's not right. Let me re-examine.
    
    a~u means u is a neighbor of a. For u to be a common neighbor of a and b, we need b~u too.
    We set cand_b = Av & ~(1<<u), so b is NOT adjacent to u. So u is not a common neighbor.
    Similarly, v~b, and for v to be common neighbor we need a~v. cand_a = Au & ~(1<<v),
    so a is NOT adjacent to v. So v is not a common neighbor.
    
    Therefore, common neighbors of a and b in G are exactly the common neighbors in G\\{(u,v)}.
    So we just need: a,b not adjacent AND (mat[a] & mat[b]) == 0.
    """
    n = len(mat)
    Au = mat[u]
    Av = mat[v]
    # a neighbors of u (excluding v), b neighbors of v (excluding u)
    cand_a = Au & ~(1 << v)
    cand_b = Av & ~(1 << u)
    # For each candidate a, find b in cand_b that is not adjacent to a and shares no common neighbor with a
    for a in range(n):
        if not (cand_a >> a) & 1:
            continue
        # b must not be adjacent to a
        not_adj_b = cand_b & ~mat[a]
        if not not_adj_b:
            continue
        # b must have no common neighbor with a: mat[a] & mat[b] == 0
        # For each b in not_adj_b, check if mat[a] & mat[b] == 0
        # Build mask of b that have NO common neighbor with a
        # b has a common neighbor with a iff b is in N2(a) = _matvec2(mat, mat[a])
        N2a = _matvec2(mat, mat[a])
        # b in not_adj_b that are NOT in N2a have no common neighbor with a
        good_b = not_adj_b & ~N2a
        if good_b:
            return True
    return False


def _valid_and_critical(mat):
    """Check diameter <= 2 (and exactly 2 by connectivity for n>=2) and all edges critical."""
    n = len(mat)
    if not _diameter_leq2_ok(mat):
        return False
    # ensure diameter exactly 2 (not 1): need some non-edge
    allmask = (1 << n) - 1
    is_complete = True
    for i in range(n):
        if (mat[i] & allmask) != (allmask ^ (1 << i)):
            is_complete = False
            break
    if is_complete:
        return False  # diameter 1
    # check every edge critical
    for u in range(n):
        row = mat[u]
        for v in range(u + 1, n):
            if (row >> v) & 1:
                if not _edge_critical(mat, u, v):
                    return False
    return True


def _mat_to_edges(mat):
    n = len(mat)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (mat[u] >> v) & 1:
                edges.append([int(u), int(v)])
    return edges


def construct():
    start = time.time()
    deadline = start + 78  # safety margin

    n = P['n']
    if n < 2:
        return []

    half = n // 2
    mat = [0] * n
    # Build K_{half, n-half}
    for u in range(half):
        for v in range(half, n):
            mat[u] |= (1 << v)
            mat[v] |= (1 << u)

    if not _valid_and_critical(mat):
        # fallback: still return K_{half,n-half} edges (it is known valid)
        return _mat_to_edges(mat)

    best_mat = list(mat)
    best_edges = _mat_to_edges(mat)
    best_count = len(best_edges)

    def try_add(u, v):
        nonlocal best_mat, best_edges, best_count
        if (mat[u] >> v) & 1:
            return False
        mat[u] |= (1 << v)
        mat[v] |= (1 << u)
        if _valid_and_critical(mat):
            best_mat = list(mat)
            best_edges = _mat_to_edges(mat)
            best_count = len(best_edges)
            return True
        mat[u] &= ~(1 << v)
        mat[v] &= ~(1 << u)
        return False

    # Pass 1: single-edge additions, iterate until no improvement
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2:
                    break
                if not ((mat[u] >> v) & 1):
                    if try_add(u, v):
                        improved = True
        else:
            continue
        break

    # Pass 2: targeted edge swaps - only try swaps that might help
    # Strategy: for each missing edge (a,b) that can't be added directly,
    # try removing an edge that's "blocking" it
    def edges_list(mat):
        out = []
        for u in range(n):
            for v in range(u + 1, n):
                if (mat[u] >> v) & 1:
                    out.append((u, v))
        return out

    def missing_list(mat):
        out = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((mat[u] >> v) & 1):
                    out.append((u, v))
        return out

    improved = True
    passes = 0
    while improved and passes < 3 and time.time() < deadline - 2:
        improved = False
        passes += 1
        present = edges_list(mat)
        missing = missing_list(mat)
        for (au, av) in missing:
            if time.time() > deadline - 2:
                break
            # Try removing each present edge and adding (au,av)
            # Only try if removing the edge doesn't disconnect the graph
            for (ru, rv) in present:
                if time.time() > deadline - 2:
                    break
                # Quick check: if ru or rv is au or av, this is just a replacement
                # Remove (ru,rv), add (au,av)
                mat[ru] &= ~(1 << rv)
                mat[rv] &= ~(1 << ru)
                mat[au] |= (1 << av)
                mat[av] |= (1 << au)
                if _valid_and_critical(mat):
                    best_mat = list(mat)
                    best_edges = _mat_to_edges(mat)
                    best_count = len(best_edges)
                    improved = True
                else:
                    mat[au] &= ~(1 << av)
                    mat[av] &= ~(1 << au)
                    mat[ru] |= (1 << rv)
                    mat[rv] |= (1 << ru)

    # Final rebuild from best_mat to guarantee consistency
    final = _mat_to_edges(best_mat)
    # type safety: ensure Python ints and list of lists
    return [[int(u), int(v)] for (u, v) in final]
# EVOLVE-BLOCK-END
