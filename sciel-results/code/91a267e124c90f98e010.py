"""Construct a small 3-regular graph of girth >= g.

DIFFERENT FAMILY from the seed: instead of enumerating circulant / doubly-circulant
offsets and certifying girth with a per-start Python BFS over non-backtracking
walk states, we:
  (a) build explicit Cayley graphs of small finite groups (SL(2,q), GL(2,q),
      affine AGL(1,q)) with a fixed 3-generator involution set, and
  (b) certify girth >= g with a vectorized NumPy non-backtracking matrix power:
        M = (A - I) restricted to directed edges,
        cycle of length L < g  <=>  (M^L) has a nonzero diagonal entry.
      This is O(g * n) vectorized multiplications instead of O(n * 2^g) Python
      loop iterations, freeing the wall-clock budget to *test many more
      candidates* (the actual bottleneck of the seed).

The LUW construction is retained only as a guaranteed-valid fallback.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# LUW construction (guaranteed-valid fallback, unchanged from seed)
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Vectorized girth certification via non-backtracking matrix powers
# ---------------------------------------------------------------------------
def _girth_ok_vectorized(edges, n, g):
    """Return True iff the simple graph (n, edges) has girth >= g.

    Builds the directed non-backtracking matrix M of size 2n x 2n where
    state (v, u) means 'at v, came from u'.  A cycle of length L < g exists
    iff (M^L) has a nonzero diagonal entry for some 3 <= L < g.
    """
    if n < 3:
        return True
    # Build directed-edge index map
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Assign an index to each directed edge (u -> v)
    de_idx = {}
    de_u = []
    de_v = []
    for u in range(n):
        for v in adj[u]:
            de_idx[(u, v)] = len(de_u)
            de_u.append(u)
            de_v.append(v)
    D = len(de_u)
    if D == 0:
        return True
    # Build M: M[(u,v), (v,w)] = 1 for w != u
    # M has D rows (from-state (u,v)) and D cols (to-state (v,w))
    # Use dense int8; D <= 3n
    M = np.zeros((D, D), dtype=np.int8)
    # For each directed edge (u,v), the out-neighbors are (v,w) for w in adj[v], w != u
    # Precompute: for each directed-edge index e=(u,v), list of to-indices
    for e, (u, v) in enumerate(de_idx):
        for w in adj[v]:
            if w == u:
                continue
            M[e, de_idx[(v, w)]] = 1
    # Now M^L diagonal: (M^L)[e, e'] where e and e' are the same directed edge
    # A closed non-backtracking walk of length L starting at directed edge e
    # and ending at directed edge e' corresponds to M^L[e, e'].
    # A cycle of length L exists iff there is a closed walk of length L, i.e.
    # some (M^L)[e, e] != 0?  Not exactly -- a cycle of length L starting at
    # vertex s corresponds to a closed walk s->...->s of length L, which in
    # directed-edge terms is a sequence of L directed edges where edge i's head
    # = edge i+1's tail, and edge L's head = edge 1's tail.  This is a closed
    # walk in the line graph of the directed edges, i.e. (M^L)[e, f] where f
    # is a directed edge with tail = e's tail and head = e's head?  No.
    #
    # Simpler: a cycle of length L in the undirected graph corresponds to a
    # closed non-backtracking walk of length L in the directed-edge graph where
    # we return to the SAME directed edge (or its reverse).  Actually, the
    # standard result: girth >= g iff M^L has zero diagonal for all 3 <= L < g
    # where M is the non-backtracking matrix on directed edges and we check
    # (M^L)[e, e] for all directed edges e.  Because a cycle of length L gives
    # exactly 2L directed-edge closed walks of length L (L starting vertices x
    # 2 directions), each corresponding to (M^L)[e, e] != 0 for some e.
    #
    # So: girth >= g iff for all L in [3, g-1], diag(M^L) is all zero.

    # Power M iteratively, checking diagonal at each length
    P = M.copy()  # P = M^1
    # length 1 and 2 can't form cycles in a simple graph (min cycle = 3)
    for L in range(3, g):
        P = P @ M  # P = M^L
        # Check diagonal
        if np.any(np.diag(P) != 0):
            return False
    return True


# ---------------------------------------------------------------------------
# Group / Cayley graph machinery
# ---------------------------------------------------------------------------
def _sl2q_elements(q):
    """Return list of (a,b,c,d) in SL(2,q), i.e. ad-bc = 1 mod q."""
    elems = []
    one = 1 % q
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # ad - bc = 1 => d = (1 + bc) * a^{-1} if a != 0
                if a == 0:
                    # -bc = 1 => bc = -1 mod q; d is free? No: 0*d - b*c = 1 => -bc = 1
                    if (b * c) % q == (q - 1) % q:
                        for d in range(q):
                            elems.append((a, b, c, d))
                    continue
                # a*d = 1 + b*c => d = (1+b*c) * a^{-1}
                # Need a^{-1} mod q; only exists if gcd(a,q)=1.  For prime q, always.
                # For non-prime q, skip if not invertible.
                # We'll only use prime q in practice.
                aa = a
                # compute inverse via pow (works for prime q)
                try:
                    ainv = pow(a, -1, q)
                except Exception:
                    continue
                d = (1 + b * c) % q
                d = (d * ainv) % q
                elems.append((a, b, c, d))
    return elems


def _sl2q_matmul(e1, e2, q):
    a1, b1, c1, d1 = e1
    a2, b2, c2, d2 = e2
    a = (a1 * a2 + b1 * c2) % q
    b = (a1 * b2 + b1 * d2) % q
    c = (c1 * a2 + d1 * c2) % q
    d = (c1 * b2 + d1 * d2) % q
    return (a, b, c, d)


def _sl2q_cayley(q, gens):
    """Build Cayley graph of SL(2,q) with generator set gens (list of elements).

    Returns (n, edges) if the result is a simple 3-regular graph, else None.
    gens must be a set of 3 involutions (g^2 = I), none equal to I, mutually
    distinct, and no generator is the inverse of another (they're all
    involutions so inverse = self, which is fine for undirected).
    """
    elems = _sl2q_elements(q)
    n = len(elems)
    if n < 3:
        return None
    idx = {e: i for i, e in enumerate(elems)}
    identity = (1 % q, 0, 0, 1 % q)
    if identity not in idx:
        return None
    # Verify gens are valid
    for g in gens:
        if g == identity:
            return None
        if g not in idx:
            return None
        # Check involution
        if _sl2q_matmul(g, g, q) != identity:
            return None
    if len(set(gens)) != 3:
        return None
    # Build edges
    edges = []
    for i, e in enumerate(elems):
        for g in gens:
            f = _sl2q_matmul(e, g, q)
            j = idx.get(f)
            if j is None:
                return None
            if i < j:
                edges.append((i, j))
    # Verify 3-regular and simple
    if len(edges) != n * 3 // 2:
        return None
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
    if len(set(tuple(sorted(e)) for e in edges)) != len(edges):
        return None
    return n, edges


def _find_sl2q_involution_gens(q):
    """Find a set of 3 involutions in SL(2,q) that generate a connected Cayley graph.

    In SL(2,q), involutions (non-identity elements of order 2) are matrices with
    trace 0 and determinant 1, i.e. a + d = 0, ad - bc = 1 => -a^2 - bc = 1.
    For q odd, these exist.  We search for a small set.
    """
    elems = _sl2q_elements(q)
    identity = (1 % q, 0, 0, 1 % q)
    involutions = []
    for e in elems:
        if e == identity:
            continue
        if _sl2q_matmul(e, e, q) == identity:
            involutions.append(e)
    if len(involutions) < 3:
        return None
    # Try the first few triples; check connectivity via BFS
    # For efficiency, just try a small number of triples
    import itertools
    for combo in itertools.combinations(involutions[:20], 3):
        gens = list(combo)
        result = _sl2q_cayley(q, gens)
        if result is not None:
            # Quick connectivity check
            n, edges = result
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            seen = [False] * n
            seen[0] = True
            stack = [0]
            cnt = 1
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        cnt += 1
                        stack.append(v)
            if cnt == n:
                return result
    return None


def _affine_cayley(q, a, b, c):
    """Cayley graph of AGL(1,q) = {x -> a*x + b : a in F_q*, b in F_q} with
    generators corresponding to affine maps.  Group size q*(q-1).

    Generators: we pick 3 affine maps that are involutions.
    An affine map f(x) = a*x + b is an involution iff f(f(x)) = x.
    f(f(x)) = a*(a*x+b) + b = a^2*x + a*b + b.  For this to equal x:
      a^2 = 1 and a*b + b = 0 => b(a+1) = 0.
    So either a = 1 and b = 0 (identity, excluded), or a = -1 (b free),
    or a = 1 (excluded), or b = 0 and a = -1.
    For q odd: a = -1, b arbitrary gives involutions: f(x) = -x + b, f(f(x)) = x.
    Also a = 1, b = 0 is identity.
    So involutions in AGL(1,q) for q odd: {x -> -x + b : b in F_q}.
    That's q involutions.  Pick 3 of them.

    The Cayley graph with generators {x->-x+b1, x->-x+b2, x->-x+b3} on group of
    size q*(q-1).  This is a valid 3-regular graph.
    """
    if q < 3:
        return None
    if q % 2 == 0:
        return None  # for even q, a=-1=1, different structure
    n = q * (q - 1)
    # Group elements: (a, b) representing x -> a*x + b, a in 1..q-1, b in 0..q-1
    # Identity: (1, 0)
    # Product: (a1,b1)*(a2,b2) = (a1*a2 mod q, a1*b2 + b1 mod q)
    def prod(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1 * a2) % q, (a1 * b2 + b1) % q)
    # Involutions: (q-1, b) i.e. a = -1 = q-1
    # Generators: (q-1, b1), (q-1, b2), (q-1, b3)
    # Check they are involutions: (q-1,b)*(q-1,b) = ((q-1)^2 mod q, (q-1)*b + b) = (1, -b+b) = (1,0). Yes.
    gens = [(q - 1, b) for b in (a, b, c)]
    # Verify distinct and not identity
    if len(set(gens)) != 3:
        return None
    for g in gens:
        if g == (1, 0):
            return None
    # Build Cayley graph
    all_elems = [(A, B) for A in range(1, q) for B in range(q)]
    idx = {e: i for i, e in enumerate(all_elems)}
    edges = []
    for i, e in enumerate(all_elems):
        for g in gens:
            f = prod(e, g)
            j = idx.get(f)
            if j is None:
                return None
            if i < j:
                edges.append((i, j))
    if len(edges) != n * 3 // 2:
        return None
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
    if len(set(tuple(sorted(e)) for e in edges)) != len(edges):
        return None
    return n, edges


# ---------------------------------------------------------------------------
# Search over the Cayley-graph family
# ---------------------------------------------------------------------------
def _search_cayley(k, g, deadline):
    """Search SL(2,q) and AGL(1,q) Cayley graphs for 3-regular girth >= g.

    Returns (n, edges) for the smallest valid graph found, or None.
    """
    best = None
    best_n = float('inf')

    # Moore bound
    moore = 2 * sum((k - 1)**i for i in range(g // 2))

    # --- AGL(1,q) ---
    # n = q*(q-1).  We want n >= moore and n as small as possible.
    # q*(q-1) >= moore => q >= ~sqrt(moore)
    import math
    q_lo = max(3, int(math.ceil(math.sqrt(moore))))
    # Make q_lo odd
    if q_lo % 2 == 0:
        q_lo += 1
    q_hi = q_lo + 40
    for q in range(q_lo, q_hi + 1, 2):  # odd q only
        if time.time() > deadline:
            break
        n_est = q * (q - 1)
        if n_est >= best_n:
            break
        # Try a few generator triples (b1, b2, b3)
        # By symmetry, we can fix b1 = 0 (translate the group), so b1 = 0.
        # Then try (b2, b3) pairs.
        # The girth depends on the differences b2-b1, b3-b1, b3-b2.
        # Try small differences.
        tried = set()
        for b2 in range(1, min(q, 12)):
            for b3 in range(b2 + 1, min(q, 12 + b2)):
                if time.time() > deadline:
                    break
                key = (b2, b3)
                if key in tried:
                    continue
                tried.add(key)
                result = _affine_cayley(q, 0, b2, b3)
                if result is None:
                    continue
                n, edges = result
                if n >= best_n:
                    continue
                if _girth_ok_vectorized(edges, n, g):
                    best = (n, edges)
                    best_n = n
                    break
            if best_n <= n_est:
                break

    # --- SL(2,q) ---
    # n = q^3 - q.  Larger than AGL for same q, but richer structure.
    # Only try for small q where n is competitive.
    q_sl_lo = max(3, int(math.cbrt(moore)) if hasattr(math, 'cbrt') else int(round(moore ** (1.0 / 3.0))))
    if q_sl_lo % 2 == 0:
        q_sl_lo += 1
    q_sl_hi = q_sl_lo + 20
    for q in range(q_sl_lo, q_sl_hi + 1, 2):
        if time.time() > deadline:
            break
        n_est = q * q * q - q
        if n_est >= best_n:
            break
        result = _find_sl2q_involution_gens(q)
        if result is None:
            continue
        n, edges = result
        if n >= best_n:
            continue
        if _girth_ok_vectorized(edges, n, g):
            best = (n, edges)
            best_n = n

    return best


def _deadline_reached(t0, limit):
    return (time.time() - t0) > limit


def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 130.0  # safety margin under 150s wall-clock

    # Try Cayley graph constructions over finite groups
    result = _search_cayley(k, g, HARD_LIMIT)
    if result is not None:
        return result

    # Fallback: guaranteed-valid LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        raise

# EVOLVE-BLOCK-END
