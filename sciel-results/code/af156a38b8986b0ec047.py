"""Search-based construction for R(C4,K11) > n.

Strategy:
  * Polarity graphs of projective planes PG(2,q) are provably C4-free
    (two points share at most one neighbour), so they are strong structural
    seeds.  For some q their independence number is < 11, certifying
    R(C4,K11) > n.
  * Verification uses fast integer bitmasks (no set intersections in hot
    loops) so the 150 s budget is spent on search, not checking.
"""
import time
import random

# EVOLVE-BLOCK-START

# ---------- finite geometry helpers ----------

def _inv_mod(a, q):
    """Multiplicative inverse of a mod q (q prime).  a in 1..q-1."""
    return pow(a, q - 2, q)


def _projective_points(q):
    """All points of PG(2,q) as triples (x,y,z) with first non-zero coord = 1."""
    pts = []
    invs = [0] + [_inv_mod(a, q) for a in range(1, q)]
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    ix = invs[x]
                    pts.append((1, (y * ix) % q, (z * ix) % q))
                elif y != 0:
                    iy = invs[y]
                    pts.append((0, 1, (z * iy) % q))
                else:
                    pts.append((0, 0, 1))
    return pts


def _polarity_graph(q):
    """C4-free polarity graph of PG(2,q): p~q iff dot(p,q) == 0 (mod q)."""
    pts = _projective_points(q)
    n = len(pts)
    adj = [0] * n  # bitmask of neighbours
    for i in range(n):
        a1, b1, c1 = pts[i]
        for j in range(i + 1, n):
            a2, b2, c2 = pts[j]
            if (a1 * a2 + b1 * b2 + c1 * c2) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return n, adj


# ---------- fast verification (bitmask based) ----------

def _is_c4_free(n, adj):
    """No two vertices may share two or more neighbours."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] & adj[j]) & ((adj[i] & adj[j]) - 1):  # >= 2 bits set
                return False
    return True


def _greedy_alpha(n, adj):
    """Greedy (min-degree) lower bound on the independence number."""
    rem = (1 << n) - 1
    count = 0
    while rem:
        best_v = -1
        best_deg = n
        r = rem
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = (adj[v] & rem).bit_count()
            if d < best_deg:
                best_deg = d
                best_v = v
                if d == 0:
                    break
            r ^= lb
        count += 1
        rem &= ~(adj[best_v] | (1 << best_v))
    return count


def _has_independent_11(n, adj, limit=11):
    """Exact: is there an independent set of size >= limit?  (bounded DFS)"""
    if n < limit:
        return False

    def dfs(rem, k):
        if k >= limit:
            return True
        if rem.bit_count() < limit - k:
            return False
        # pick a vertex with max degree inside rem (strongest pruning)
        best_v = -1
        best_deg = -1
        r = rem
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = (adj[v] & rem).bit_count()
            if d > best_deg:
                best_deg = d
                best_v = v
            r ^= lb
        # branch 1: include best_v
        if dfs(rem & ~(adj[best_v] | (1 << best_v)), k + 1):
            return True
        # branch 2: exclude best_v
        if dfs(rem & ~(1 << best_v), k):
            return True
        return False

    return dfs((1 << n) - 1, 0)


# ---------- greedy extension (try to grow a valid graph) ----------

def _can_add(adj, n, cand_mask):
    """Can a new vertex n with neighbour-set cand_mask keep the graph C4-free?"""
    # (1) every existing vertex u must have <= 1 neighbour in cand
    for u in range(n):
        c = adj[u] & cand_mask
        if c & (c - 1):  # >= 2
            return False
    # (2) no two vertices in cand may share a neighbour
    r = cand_mask
    while r:
        lb = r & (-r)
        v = lb.bit_length() - 1
        if adj[v] & (cand_mask ^ lb):
            return False
        r ^= lb
    return True


def _grow_from_seed(n0, adj0, deadline, seed):
    """Greedily add vertices to a C4-free seed while keeping alpha < 11."""
    rng = random.Random(seed)
    n = n0
    adj = adj0[:]
    while n < 43 and time.time() < deadline:
        added = False
        for _ in range(300):
            if n == 0:
                cand = 0
            else:
                target = max(1, n // 4 + rng.randint(0, max(1, n // 6)))
                target = min(target, n)
                order = list(range(n))
                rng.shuffle(order)
                cand = 0
                for v in order[:target]:
                    cand |= (1 << v)
            if _can_add(adj, n, cand):
                # quick Caro-Wei upper bound on alpha must stay <= 10
                cw = 0.0
                for u in range(n):
                    nd = adj[u].bit_count() + (1 if (cand >> u) & 1 else 0)
                    cw += 1.0 / (nd + 1.0)
                cw += 1.0 / (cand.bit_count() + 1.0)
                if cw <= 10.3:
                    adj.append(cand)
                    for u in range(n):
                        if (cand >> u) & 1:
                            adj[u] |= (1 << n)
                    n += 1
                    added = True
                    break
        if not added:
            break
    return n, adj


# ---------- main entry point ----------

def construct():
    deadline = time.time() + 140.0  # safety margin before the 150 s kill

    best_n = 0
    best_adj = None

    def consider(n, adj):
        nonlocal best_n, best_adj
        if n <= best_n or n < 1:
            return
        if not _is_c4_free(n, adj):
            return
        if _has_independent_11(n, adj):
            return
        best_n = n
        best_adj = adj[:]

    # Strategy 1: polarity graphs (provably C4-free)
    for q in [5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline - 8:
            break
        try:
            n, adj = _polarity_graph(q)
            if not _is_c4_free(n, adj):
                continue
            if _has_independent_11(n, adj):
                continue
            consider(n, adj)
            # try to extend the valid polarity graph
            if time.time() < deadline - 6:
                ng, adjg = _grow_from_seed(n, adj, deadline - 2, seed=q * 1000 + 1)
                consider(ng, adjg)
        except Exception:
            continue

    # Strategy 2: greedy from scratch (multiple seeds)
    for seed in range(20):
        if time.time() > deadline - 6:
            break
        try:
            ng, adjg = _grow_from_seed(0, [], deadline - 2, seed=10000 + seed)
            consider(ng, adjg)
        except Exception:
            continue

    # Fallback: ten disjoint triangles (alpha = 10, C4-free, n = 30)
    if best_n < 30:
        n = 30
        adj = [0] * n
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        best_n = n
        best_adj = adj

    edges = []
    for i in range(best_n):
        m = best_adj[i]
        while m:
            lb = m & (-m)
            j = lb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lb

    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
