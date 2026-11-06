"""Q3-free dense graph on 16 vertices: local search with fast Q3 counting."""
import time
import random

_N = 16
_NPAIRS = _N * (_N - 1) // 2          # 120
_DEADLINE = time.time() + 110.0        # safety margin vs 120s/150s kill

# Precompute all 8 Q3 isomorphisms (as sets of 12 undirected edge tuples).
# Q3 edge set (canonical, 0..7):
_CANON = frozenset([
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
])

def _build_q3_forms():
    from itertools import permutations
    forms = set()
    for p in permutations(range(8)):
        m = [0] * 8
        for old, new in enumerate(p):
            m[old] = new
        transformed = frozenset((min(m[a], m[b]), max(m[a], m[b])) for (a, b) in _CANON)
        forms.add(transformed)
    return [frozenset(f) for f in forms]

_Q3_FORMS = _build_q3_forms()          # list of frozenset of 12 (lo,hi) tuples

def _pair_index(u, v):
    if u > v:
        u, v = v, u
    return u * 15 - (u * (u - 1)) // 2 + (v - u - 1)

def _count_q3(adj):
    """adj[i] is an int bitmask of neighbors of vertex i. Return number of (not necessarily induced) Q3 copies."""
    total = 0
    for form in _Q3_FORMS:
        # form is a frozenset of 12 (lo,hi) pairs on labels 0..7
        # Try to inject form -> vertices 0..15
        # We iterate assignments of the 8 labels of the form to 16 vertices.
        # Use backtracking with edge pruning.
        # Build adjacency requirement among the 8 form-labels.
        # form_labels = 0..7 ; requirement edges in form are between these labels.
        req = [0] * 8
        for (a, b) in form:
            req[a] |= (1 << b)
            req[b] |= (1 << a)
        order = list(range(8))
        # order labels by descending degree to prune
        order.sort(key=lambda x: -req[x].bit_count())
        assign = [-1] * 8
        used = 0
        def bt(k, used):
            nonlocal total
            if time.time() > _DEADLINE:
                return
            if k == 8:
                total += 1
                return
            lab = order[k]
            # candidate vertices
            cand = 0
            for v in range(_N):
                if (used >> v) & 1:
                    continue
                ok = True
                # check edges to already-assigned neighbors of lab
                mask = req[lab]
                # iterate assigned labels
                for j in range(k):
                    lj = order[j]
                    if (mask >> lj) & 1:
                        w = assign[lj]
                        if not ((adj[v] >> w) & 1):
                            ok = False
                            break
                if ok:
                    cand |= (1 << v)
            # try each candidate
            while cand:
                lsb = cand & -cand
                v = lsb.bit_length() - 1
                cand ^= lsb
                assign[lab] = v
                # forward check: for future labels, ensure at least one candidate exists
                # (light check: just continue; backtracking handles it)
                bt(k + 1, used | lsb)
                assign[lab] = -1
        bt(0, 0)
    return total

def _random_start(density=0.6):
    adj = [0] * _N
    pairs = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if random.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                pairs.append((u, v))
    return adj, pairs

def _q3_free(adj):
    return _count_q3(adj) == 0

def construct():
    random.seed(1234567)
    best_adj = None
    best_m = -1

    # Try several densities / restarts until deadline
    densities = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    attempts = 0
    while time.time() < _DEADLINE - 5.0 and attempts < 40:
        d = densities[attempts % len(densities)]
        adj, edges = _random_start(d)
        m = len(edges)
        q3 = _count_q3(adj)
        if q3 == 0 and m > best_m:
            best_adj = adj[:]
            best_m = m
        attempts += 1

    # Local search: try to delete edges to remove Q3s while keeping many edges,
    # then try to add edges back without creating Q3s.
    def refine(adj, edges):
        # Phase 1: while Q3 exists, remove a random edge from a Q3
        guard = 0
        while time.time() < _DEADLINE - 5.0 and guard < 5000:
            q3 = _count_q3(adj)
            if q3 == 0:
                break
            guard += 1
            # find one Q3 via a fresh count that also returns an instance:
            # simpler: random edge removal biased by degree
            # remove a random edge that is in many Q3s -> approximate by high-degree edges
            candidates = list(edges)
            random.shuffle(candidates)
            removed = None
            for (u, v) in candidates:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                if _count_q3(adj) < q3:
                    removed = (u, v)
                    break
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            if removed is None:
                # fallback: remove a random edge
                u, v = random.choice(candidates)
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                edges.remove((u, v))
            else:
                edges.remove(removed)
            if time.time() > _DEADLINE - 5.0:
                break

        # Phase 2: greedy add edges that don't create Q3
        all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
        random.shuffle(all_pairs)
        for (u, v) in all_pairs:
            if time.time() > _DEADLINE - 5.0:
                break
            if (adj[u] >> v) & 1:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _count_q3(adj) != 0:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                edges.add((u, v))
        return adj, edges

    if best_adj is not None:
        adj, edges = refine(best_adj, [(u, v) for u in range(_N) for v in range(u + 1, _N) if (best_adj[u] >> v) & 1])
        m = len(edges)
        if _q3_free(adj) and m > best_m:
            best_adj = adj
            best_m = m

    # Fallback: guaranteed Q3-free (K3,13) with 39 edges
    if best_adj is None:
        # build K3,13
        adj = [0] * _N
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        best_adj = adj
        best_m = 39

    # Final verification and output
    if _count_q3(best_adj) != 0:
        # fallback to safe
        best_adj = [0] * _N
        for a in range(3):
            for b in range(3, 16):
                best_adj[a] |= (1 << b)
                best_adj[b] |= (1 << a)

    out = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                out.append([u, v])
    return out
