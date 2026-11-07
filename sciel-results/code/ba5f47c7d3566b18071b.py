# EVOLVE-BLOCK-START
import time
import random
import sys

N = 128
# Safety margin: evaluator kills at 90s; we guard at 80s.
DEADLINE = 80.0

def _all_bad_triples():
    """Yield every (a,b,c) with 1<=a<b<c<=128 such that a+b, a+c, b+c are all in 1..128."""
    # Precompute pair-sum existence as a 129x129 bool for speed.
    in_range = [False] * (2 * N + 1)
    for i in range(1, N + 1):
        in_range[i] = True
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                break  # since c>b, a+c > a+b > N, no triples for larger b
            if not in_range[s_ab]:
                continue
            for c in range(b + 1, N + 1):
                s_ac = a + c
                if s_ac > N:
                    break
                if not in_range[s_ac]:
                    continue
                s_bc = b + c
                if s_bc > N:
                    break
                if in_range[s_bc]:
                    triples.append((a, b, c))
    return triples

def _build_conflict_graph(triples):
    """Build adjacency as list of Python int bitmasks (128 bits each)."""
    adj = [0] * (N + 1)  # 1-indexed
    for (a, b, c) in triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    return adj

def _greedy_color_bound(adj, candidates_mask):
    """
    Upper bound on the max independent set size within candidates_mask,
    using a greedy sequential coloring (like Tomita's BB).
    Returns an integer >= true MIS size.
    """
    # Extract list of candidate vertices
    verts = []
    m = candidates_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length()  # 1-indexed vertex id
        verts.append(v)
        m ^= lsb
    n = len(verts)
    if n == 0:
        return 0
    # Greedy coloring: assign colors to vertices in order, each vertex gets
    # the smallest color not used by its already-colored neighbors in verts.
    color_of = [0] * (n + 1)
    max_color = 0
    for i in range(n):
        v = verts[i]
        # Determine forbidden colors from neighbors already colored
        forbidden = [False] * (max_color + 2)
        adj_v = adj[v]
        for j in range(i):
            u = verts[j]
            if (adj_v >> u) & 1:
                c = color_of[j + 1]
                if c <= max_color:
                    forbidden[c] = True
        # Find smallest available color
        c = 1
        while c <= max_color and forbidden[c]:
            c += 1
        if c > max_color:
            max_color = c
        color_of[i + 1] = c
    return max_color

def _branch_and_bound(adj, candidates_mask, best_size, deadline, t0):
    """
    Exact branch-and-bound for maximum independent set on the conflict graph,
    restricted to candidates_mask. Returns (best_size, best_mask).
    Uses Tomita-style coloring bound.
    """
    best = [best_size]
    best_mask = [0]

    def bb(cand_mask, current_mask, current_size):
        nonlocal best, best_mask
        if time.time() - t0 > deadline:
            return
        if cand_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
                best_mask[0] = current_mask
            return
        # Upper bound via greedy coloring
        ub = _greedy_color_bound(adj, cand_mask)
        if current_size + ub <= best[0]:
            return
        # Pick a vertex to branch on (first available)
        m = cand_mask
        lsb = m & (-m)
        v = lsb.bit_length()
        # Branch 1: include v
        new_cand = cand_mask & ~(adj[v] | (1 << v))
        bb(new_cand, current_mask | (1 << v), current_size + 1)
        # Branch 2: exclude v
        bb(cand_mask ^ (1 << v), current_mask, current_size)

    bb(candidates_mask, 0, 0)
    return best[0], best_mask[0]

def _find_bad_triple_in_set(S):
    """Return one bad triple (a,b,c) in set S, or None."""
    Sset = S
    L = sorted(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(i + 1, n):
            b = L[j]
            if (a + b) not in Sset:
                continue
            for k in range(j + 1, n):
                c = L[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return (a, b, c)
    return None

def _error_repair(S, triples_by_vertex, deadline, t0):
    """
    Repair: while a bad triple exists in S, remove the vertex in that triple
    that participates in the most bad triples (most 'guilty').
    This is the structural 'error_repair' mechanism.
    """
    S = set(S)
    # Precompute for each vertex the count of bad triples it appears in
    # (global, not restricted to S) as a heuristic for "guilt".
    while time.time() - t0 < deadline:
        t = _find_bad_triple_in_set(S)
        if t is None:
            break
        a, b, c = t
        # Pick the one with highest global triple count
        counts = [(triples_by_vertex[a], a), (triples_by_vertex[b], b), (triples_by_vertex[c], c)]
        counts.sort(reverse=True)
        S.discard(counts[0][1])
    return S

def _greedy_mis_seed(adj, deadline, t0):
    """
    Greedy maximal independent set seed: repeatedly pick the vertex with
    minimum degree in the remaining candidate set (a classic MIS heuristic).
    Returns a mask.
    """
    candidates = (1 << (N + 1)) - 2  # vertices 1..128
    selected = 0
    while candidates and (time.time() - t0) < deadline:
        # Find vertex with minimum degree within candidates
        best_v = -1
        best_deg = N + 1
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length()
            deg = bin(adj[v] & candidates).count("1")
            if deg < best_deg:
                best_deg = deg
                best_v = v
            if deg == 0:
                break
            m ^= lsb
        if best_v == -1:
            break
        selected |= (1 << best_v)
        candidates &= ~(adj[best_v] | (1 << best_v))
    return selected

def _local_search_improve(adj, mask, deadline, t0):
    """
    Try to improve an independent-set mask by swapping one selected vertex
    for an unselected one if it increases size (via 1-swap with B&B on small subproblem).
    Simple version: try adding each unselected vertex that doesn't conflict,
    and if it creates a conflict, try removing the least-conflicting selected vertex.
    """
    mask = mask
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        selected = mask
        unselected = ((1 << (N + 1)) - 2) & ~selected
        # Try each unselected vertex
        m = unselected
        while m and not improved:
            lsb = m & (-m)
            v = lsb.bit_length()
            conflicts = adj[v] & selected
            if conflicts == 0:
                # Can add v directly
                mask |= (1 << v)
                improved = True
                break
            # Otherwise, try to remove one conflicting vertex to add v
            # Only worthwhile if we can remove 1 and add 1 (size same) -- not useful.
            # For size increase, we'd need to remove 0 and add 1, already handled.
            m ^= lsb
    return mask

def construct():
    t0 = time.time()
    deadline = DEADLINE

    # 1. Enumerate all bad triples (this is the structural "conflict" model)
    triples = _all_bad_triples()
    triples_by_vertex = [0] * (N + 1)
    for (a, b, c) in triples:
        triples_by_vertex[a] += 1
        triples_by_vertex[b] += 1
        triples_by_vertex[c] += 1

    # 2. Build conflict graph
    adj = _build_conflict_graph(triples)

    # 3. Get a good initial seed via greedy MIS
    seed_mask = _greedy_mis_seed(adj, deadline, t0)
    best_size = bin(seed_mask).count("1")
    best_mask = seed_mask

    # 4. Branch-and-bound exact search (with time limit)
    remaining = ((1 << (N + 1)) - 2) & ~seed_mask
    # We'll run B&B on the full candidate set but start with the seed as a lower bound.
    # To keep it fast, we restrict B&B to a subset: vertices that could potentially
    # improve over the seed. We'll just run B&B on the full set with the seed as lower bound.
    t1 = time.time()
    try:
        bb_size, bb_mask = _branch_and_bound(adj, ((1 << (N + 1)) - 2), best_size, deadline - 5.0, t0)
        if bb_size > best_size:
            best_size = bb_size
            best_mask = bb_mask
    except Exception:
        pass

    # 5. Convert mask to set
    S = set()
    m = best_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length()
        S.add(v)
        m ^= lsb

    # 6. ERROR REPAIR: fix any residual bad triples by removing the most-guilty vertex
    S = _error_repair(S, triples_by_vertex, deadline - 2.0, t0)

    # 7. Final local search improvement (try to add back vertices)
    # Rebuild mask from repaired S
    mask = 0
    for v in S:
        mask |= (1 << v)
    mask = _local_search_improve(adj, mask, deadline - 1.0, t0)
    S = set()
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length()
        S.add(v)
        m ^= lsb

    # 8. Final validation and repair if needed
    S = _error_repair(S, triples_by_vertex, deadline - 0.5, t0)

    # 9. Ensure all are Python ints in 1..128
    result = [int(x) for x in sorted(S) if 1 <= x <= 128]

    # Fallback if something went wrong
    if len(result) < 64:
        result = list(range(65, 129))

    return result

# EVOLVE-BLOCK-END
