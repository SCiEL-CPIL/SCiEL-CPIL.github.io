"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 15
DEADLINE = time.time() + 80.0  # safety margin vs 90s kill


def _check_valid(nbrs):
    """Full validation: no simple 3-edge path has equal-degree endpoints."""
    deg = [len(n) for n in nbrs]
    for u in range(N):
        for v in nbrs[u]:
            if v <= u:
                continue
            seen = set()
            for s in nbrs[u]:
                if s == v:
                    continue
                seen.add(deg[s])
            for t in nbrs[v]:
                if t == u:
                    continue
                if deg[t] in seen:
                    return False
    return True


def construct():
    start = time.time()
    best = 0
    best_nbrs = None
    rng = random.Random(12345)

    def try_add(nbrs, u, v):
        """Return True and mutate nbrs if adding edge (u,v) keeps validity."""
        if v in nbrs[u]:
            return True
        # degrees before adding
        deg = [len(n) for n in nbrs]
        # candidate forbidden degrees: neighbors of u (excl v) and neighbors of v (excl u)
        bad = set()
        for s in nbrs[u]:
            if s != v:
                bad.add(deg[s])
        for t in nbrs[v]:
            if t != u:
                if deg[t] in bad:
                    return False
        # also ensure no new degree collision: after adding, deg[u] and deg[v] increase.
        # The check above already covers all s in nbrs[u] and t in nbrs[v].
        # But we must also consider that s could equal a neighbor whose degree equals deg[t]?
        # That's covered by the set intersection.
        nbrs[u].add(v)
        nbrs[v].add(u)
        return True

    def greedy_fill(nbrs, order):
        """Try to add edges in a given order of candidate pairs."""
        for (u, v) in order:
            if v in nbrs[u]:
                continue
            try_add(nbrs, u, v)

    # Precompute all candidate pairs
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    for restart in range(2000):
        if time.time() > DEADLINE:
            break
        # Random permutation of pairs for diversity
        perm = all_pairs[:]
        rng.shuffle(perm)
        nbrs = [set() for _ in range(N)]
        greedy_fill(nbrs, perm)
        # Local search: try to add more edges, then swap
        improved = True
        while improved:
            improved = False
            for (u, v) in all_pairs:
                if v not in nbrs[u]:
                    if try_add(nbrs, u, v):
                        improved = True
        ecount = sum(len(n) for n in nbrs) // 2
        if ecount > best:
            best = ecount
            best_nbrs = [set(n) for n in nbrs]
            if best >= 90:
                break

    if best_nbrs is None:
        # Fallback: a valid sparse graph (star is valid)
        nbrs = [set() for _ in range(N)]
        for v in range(1, N):
            nbrs[0].add(v)
            nbrs[v].add(0)
        best = N - 1
        best_nbrs = nbrs

    # Final validation
    if not _check_valid(best_nbrs):
        # Should not happen, but fallback to star
        nbrs = [set() for _ in range(N)]
        for v in range(1, N):
            nbrs[0].add(v)
            nbrs[v].add(0)
        best_nbrs = nbrs

    edges = []
    for u in range(N):
        for v in best_nbrs[u]:
            if u < v:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
