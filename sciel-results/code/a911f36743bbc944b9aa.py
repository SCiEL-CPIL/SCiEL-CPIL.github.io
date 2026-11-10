# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 20
FULL = (1 << N) - 1

def construct():
    """
    Build a K2,3-free graph on vertices 0..19 maximizing edges.
    K2,3-free <=> no two vertices have >=3 common neighbours.
    Strategy: Paley-style init -> greedy add (regularity-biased) -> repair ->
    local search (edge swaps / delete-then-readd) -> multi-restart with deadline.
    """
    start = time.time()
    DEADLINE = start + 110.0   # safety margin before 120s limit

    def now():
        return time.time()

    def common(u, v, adj):
        return (adj[u] & adj[v]).bit_count()

    def is_k23_free(adj):
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True

    def num_edges(adj):
        return sum(adj[i].bit_count() for i in range(N)) // 2

    def can_add(u, v, adj):
        if adj[u] >> v & 1:
            return True  # already present
        # adding u-v: new common-neighbor sets involving u or v
        c = adj[u] & adj[v]
        # For any w in N(u) (w!=v), common(u,w) gains v if v in N(w) (true) => +1
        # For any w in N(v) (w!=u), common(v,w) gains u => +1
        # For w in N(u)&N(v): common(u,w) and common(v,w) each +1, and common(u,v) becomes |c|
        # Easiest: temporarily test
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        ok = True
        # check pairs involving u
        for w in range(N):
            if w == u:
                continue
            if (adj[u] & adj[w]).bit_count() >= 3:
                ok = False
                break
        if ok:
            for w in range(N):
                if w == v:
                    continue
                if (adj[v] & adj[w]).bit_count() >= 3:
                    ok = False
                    break
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return ok

    def remove_edge(u, v, adj):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def add_edge(u, v, adj):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def repair(adj):
        """Remove edges until K2,3-free. Returns number removed."""
        removed = 0
        # Repeatedly find a violating pair, remove an edge among the shared neighborhood.
        for _ in range(2000):
            bad = None
            for u in range(N):
                for v in range(u + 1, N):
                    c = adj[u] & adj[v]
                    if c.bit_count() >= 3:
                        bad = (u, v, c)
                        break
                if bad:
                    break
            if bad is None:
                break
            u, v, c = bad
            # remove edges (u,w) and (v,w) for one shared w to drop count by 1
            w = (c & -c).bit_length() - 1
            remove_edge(u, w, adj)
            remove_edge(v, w, adj)
            removed += 2
        return removed

    def greedy_add(adj):
        """Add as many edges as possible, biased toward low-degree vertices."""
        improved = True
        while improved:
            improved = False
            # candidate edges ordered by (deg[u]+deg[v], u, v)
            cands = []
            for u in range(N):
                du = adj[u].bit_count()
                for v in range(u + 1, N):
                    if adj[u] >> v & 1:
                        continue
                    cands.append((du + adj[v].bit_count(), u, v))
            cands.sort()
            for _, u, v in cands:
                if adj[u] >> v & 1:
                    continue
                if can_add(u, v, adj):
                    add_edge(u, v, adj)
                    improved = True

    def local_search(adj, time_limit):
        """Improve via random edge swaps and delete-then-readd."""
        rng = random.Random()
        best = num_edges(adj)
        # Try swaps: remove edge (a,b), add edge (c,d) not present, keep valid.
        for _ in range(200000):
            if now() > time_limit:
                break
            # candidate removal edge
            a = rng.randrange(N)
            nb = adj[a]
            if nb == 0:
                continue
            b = (nb & -nb).bit_length() - 1
            if b == a:
                continue
            # candidate add edge
            c = rng.randrange(N)
            d = rng.randrange(N)
            if c == d or c > d:
                if c > d:
                    c, d = d, c
                elif c == d:
                    continue
            if adj[c] >> d & 1:
                continue
            if (a, b) == (c, d) or (a, b) == (d, c):
                continue
            # test: remove (a,b) then add (c,d)
            remove_edge(a, b, adj)
            ok = can_add(c, d, adj)
            if ok:
                add_edge(c, d, adj)
                # net same edge count; only keep if it enables future? keep for diversity
            else:
                add_edge(a, b, adj)
        # delete-then-readd: remove a random edge, then greedy fill
        for _ in range(2000):
            if now() > time_limit:
                break
            # pick random existing edge
            for _ in range(50):
                a = rng.randrange(N)
                nb = adj[a]
                if nb == 0:
                    continue
                b = (nb & -nb).bit_length() - 1
                if b != a:
                    break
            else:
                continue
            remove_edge(a, b, adj)
            greedy_add(adj)
        return adj

    def paley_init(seed):
        """Quadratic-residue style symmetric graph on N vertices (mod prime-ish)."""
        rng = random.Random(seed)
        # Use a random symmetric 0/1 matrix with moderate density, self-adjacent.
        adj = [0] * N
        p = 0.45
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def structured_init(seed):
        """Cycle / circulant based init: connect each vertex to +/-k for a set of steps."""
        rng = random.Random(seed)
        steps = rng.sample(range(1, N // 2 + 1), k=rng.randint(3, 6))
        adj = [0] * N
        for s in steps:
            for u in range(N):
                v = (u + s) % N
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_adj = [0] * N
    best_count = 0

    for restart in range(400):
        if now() > DEADLINE:
            break
        seed = 1000 + restart * 7
        if restart % 2 == 0:
            adj = paley_init(seed)
        else:
            adj = structured_init(seed)
        repair(adj)
        greedy_add(adj)
        # local search with a per-restart budget
        per_budget = min(DEADLINE - 1.0, now() + 0.25)
        adj = local_search(adj, per_budget)
        repair(adj)
        greedy_add(adj)
        c = num_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
        if now() > DEADLINE:
            break

    # final polish on best
    if now() < DEADLINE - 1.0:
        best_adj = local_search(best_adj, DEADLINE - 1.0)
        repair(best_adj)
        greedy_add(best_adj)
        best_adj = local_search(best_adj, DEADLINE - 0.5)
        repair(best_adj)

    # ensure valid
    if not is_k23_free(best_adj):
        repair(best_adj)
    if not is_k23_free(best_adj):
        # fallback to seed
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] >> v & 1:
                edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
