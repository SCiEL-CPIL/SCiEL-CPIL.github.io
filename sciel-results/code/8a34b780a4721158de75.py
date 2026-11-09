import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 143.0

    def time_left():
        return DEADLINE - time.time()

    def bit_count(x):
        return bin(x).count('1')

    # ---------- deterministic C4-free fallback ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- C4-free construction (correct constraint) ----------
    def build_c4free(n, seed):
        rng = random.Random(seed)
        best_adj = None
        best_density = -1

        for attempt in range(6):
            adj = [0] * n
            order = list(range(n))
            rng.shuffle(order)
            for idx, i in enumerate(order):
                for j in order[idx + 1:]:
                    if not _can_add(adj, n, i, j):
                        continue
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            density = sum(bit_count(b) for b in adj)
            if density > best_density:
                best_density = density
                best_adj = adj
        return best_adj

    def _can_add(adj, n, i, j):
        # Full C4 check: adding edge (i,j) must not give any pair >1 common neighbour.
        # Pair (i,j): common neighbours unchanged, so just need |N(i)∩N(j)| <= 1.
        if bit_count(adj[i] & adj[j]) > 1:
            return False
        # For k in N(i): pair (j,k) gains common neighbour i.
        #   need |N(j)∩N(k)| == 0 (i not yet in N(j)).
        kmask = adj[i]
        while kmask:
            lsb = kmask & -kmask
            k = lsb.bit_length() - 1
            if bit_count(adj[j] & adj[k]) > 0:
                return False
            kmask ^= lsb
        # For k in N(j): pair (i,k) gains common neighbour j.
        kmask = adj[j]
        while kmask:
            lsb = kmask & -kmask
            k = lsb.bit_length() - 1
            if bit_count(adj[i] & adj[k]) > 0:
                return False
            kmask ^= lsb
        return True

    def densify(adj, n, seed):
        rng = random.Random(seed)
        changed = True
        while changed and time.time() < DEADLINE - 4.0:
            changed = False
            pairs = list(combinations(range(n), 2))
            rng.shuffle(pairs)
            for i, j in pairs:
                if (adj[i] >> j) & 1:
                    continue
                if _can_add(adj, n, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    changed = True
        return adj

    # ---------- Fast exact alpha via bitset branch and bound ----------
    def exact_alpha(adj, n, timeout=8.0):
        if n == 0:
            return 0
        best = [0]
        start = time.time()

        def solve(cand, cur):
            if time.time() - start > timeout:
                return
            if cand == 0:
                if cur > best[0]:
                    best[0] = cur
                return
            if cur + bit_count(cand) <= best[0]:
                return
            # pick max-degree vertex in induced subgraph
            v = -1
            maxdeg = -1
            tm = cand
            while tm:
                lsb = tm & -tm
                idx = lsb.bit_length() - 1
                d = bit_count(adj[idx] & cand)
                if d > maxdeg:
                    maxdeg = d
                    v = idx
                tm ^= lsb
            # include v
            nbr = adj[v] & cand
            solve(cand & ~(nbr | (1 << v)), cur + 1)
            # exclude v
            solve(cand & ~(1 << v), cur)

        solve((1 << n) - 1, 0)
        return best[0]

    # ---------- Candidate validation ----------
    def fast_ok(adj, n):
        # C4 check
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    if bit_count(adj[i] & adj[j]) > 1:
                        return False
        # Greedy MIS filter
        rng = random.Random(999)
        for _ in range(12):
            order = list(range(n))
            rng.shuffle(order)
            indep = []
            in_set = 0
            for v in order:
                if (in_set >> v) & 1:
                    continue
                if all(((adj[v] >> u) & 1) == 0 for u in indep):
                    indep.append(v)
                    in_set |= (1 << v)
            if len(indep) >= 11:
                return False
        return True

    # ---------- Main search ----------
    best = None
    target_ns = [43, 42, 41, 40, 39, 38, 37]

    for n in target_ns:
        if time.time() >= DEADLINE - 4.0:
            break
        max_trials = 60 if n <= 40 else 30
        for trial in range(max_trials):
            if time.time() >= DEADLINE - 4.0:
                break
            adj = build_c4free(n, trial)
            adj = densify(adj, n, trial)
            if not fast_ok(adj, n):
                continue
            alpha = exact_alpha(adj, n, timeout=6.0)
            if alpha < 11:
                best = (n, adj)
                break
        if best is not None:
            break

    if best is None:
        return fallback()

    n, adj = best
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
