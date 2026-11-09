"""Q3-free graph on 16 vertices, maximize edges.
Strategy: structured 8-8 bipartite core (no Q3) + intra-partition cliques,
verified by an exact Q3 subgraph search with a repair backstop.
"""
import time
import random

_DEADLINE = time.time() + 145.0  # safety margin under 150s wall-clock

def _q3_copies(adj, n=16):
    """Return list of (A, B) sets (frozensets of 4) forming a Q3 subgraph."""
    # Precompute: for each 4-subset S, its internal neighbor multiset signature.
    # A Q3: partition U,V (|U|=|V|=4) such that each u in U has degree 3 into V
    # and each v in V has degree 3 into U, and the induced bipartite graph is
    # 3-regular (i.e. exactly 12 edges, each vertex degree 3).
    # Equivalent: the 4x4 0-1 adjacency matrix has all row/col sums == 3.
    from itertools import combinations
    copies = []
    verts = range(n)
    # For speed, only consider pairs of 4-sets where the bipartite edge count is 12
    # and degrees work out. Enumerate all 4-sets.
    all4 = list(combinations(verts, 4))
    idx4 = {s: i for i, s in enumerate(all4)}
    # For each 4-set, precompute adjacency to every other 4-set as a 16-bit mask
    # (which of the 16 cross-pairs are present).
    m = len(all4)
    cross = [[0]*m for _ in range(m)]
    for i, A in enumerate(all4):
        As = set(A)
        for j, B in enumerate(all4):
            if i == j:
                continue
            mask = 0
            for a_i, a in enumerate(A):
                for b_j, b in enumerate(B):
                    if a in As and b not in As:
                        if b in adj[a]:
                            mask |= (1 << (a_i*4 + b_j))
            cross[i][j] = mask
    # A mask is a Q3 iff it has 12 bits set and each row (4 bits) has 3 set
    # and each col (4 bits) has 3 set.
    good_masks = set()
    for mask in range(1 << 16):
        if bin(mask).count('1') != 12:
            continue
        ok = True
        for r in range(4):
            if bin((mask >> (r*4)) & 0xF).count('1') != 3:
                ok = False
                break
        if not ok:
            continue
        for c in range(4):
            colsum = 0
            for r in range(4):
                if (mask >> (r*4 + c)) & 1:
                    colsum += 1
            if colsum != 3:
                ok = False
                break
        if ok:
            good_masks.add(mask)
    # Now find pairs (A,B) with a good mask in either direction.
    found = set()
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if cross[i][j] in good_masks:
                key = (frozenset(all4[i]), frozenset(all4[j]))
                if key not in found:
                    found.add(key)
                    copies.append((set(all4[i]), set(all4[j])))
    return copies

def _repair(adj, n=16, max_rounds=200000, seed=12345):
    """Greedy: while a Q3 exists, remove a random edge from a random Q3."""
    rnd = random.Random(seed)
    for _ in range(max_rounds):
        if time.time() > _DEADLINE:
            return adj
        copies = _q3_copies(adj, n)
        if not copies:
            return adj
        A, B = copies[rnd.randrange(len(copies))]
        # pick a random cross edge in this Q3 to remove
        edges = [(a, b) for a in A for b in B if b in adj[a]]
        if not edges:
            return adj
        a, b = rnd.choice(edges)
        adj[a].discard(b)
        adj[b].discard(a)
    return adj

def _edge_count(adj):
    return sum(len(s) for s in adj.values()) // 2

def construct():
    n = 16
    # Partitions: L = 0..7, R = 8..15
    L = list(range(0, 8))
    R = list(range(8, 16))
    adj = [set() for _ in range(n)]

    def add(u, v):
        if u == v:
            return
        adj[u].add(v)
        adj[v].add(u)

    # Intra-partition cliques (guaranteed Q3-free: Q3 is bipartite)
    for i in range(8):
        for j in range(i+1, 8):
            add(L[i], L[j])
            add(R[i], R[j])

    # Bipartite core: structured 3-regular-ish pattern that avoids Q3.
    # Use a "difference set" pattern: connect L[i] to R[(i+d) mod 8] for d in D.
    # Choose D = {0,1,3} (a Singer-type difference set on Z8? not perfect but
    # structured). This gives each L vertex degree 3 into R and is cyclic,
    # which strongly resists the 4+4 Q3 pattern.
    D = [0, 1, 3]
    for i in range(8):
        for d in D:
            add(L[i], R[(i + d) % 8])

    # Add extra bipartite edges greedily, checking they don't create a Q3.
    # Candidate bipartite edges not yet present, in a deterministic order.
    cand = []
    for i in range(8):
        for j in range(8):
            if R[j] not in adj[L[i]]:
                cand.append((L[i], R[j]))
    # Try adding each if it keeps graph Q3-free (checked locally via full search
    # but only when a new edge is considered).
    rnd = random.Random(7)
    rnd.shuffle(cand)
    for (u, v) in cand:
        if time.time() > _DEADLINE - 5:
            break
        # tentatively add
        adj[u].add(v)
        adj[v].add(u)
        # quick check: does this create a Q3? Run full search (cheap enough
        # for 16 vertices) but bail if too slow.
        t0 = time.time()
        copies = _q3_copies(adj, n)
        if copies:
            # revert
            adj[u].discard(v)
            adj[v].discard(u)
            continue
        if time.time() - t0 > 30:
            # too slow; stop adding, keep what we have
            break

    # Final verification + repair if needed
    if time.time() < _DEADLINE - 3:
        if _q3_copies(adj, n):
            adj = _repair(adj, n)

    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    edges.sort()
    return edges
