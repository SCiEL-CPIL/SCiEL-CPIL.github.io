"""Find large subset of [1,256] where a+b is never a perfect square (including a=b)."""
import math
import time
import random

# Precompute perfect squares up to 512 (max sum is 256+256=512)
_SQ_SET = set()
_i = 1
while _i * _i <= 512:
    _SQ_SET.add(_i * _i)
    _i += 1
del _i


def _is_sq(n):
    return n in _SQ_SET


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin vs 90s kill

    N = 256
    nodes = list(range(1, N + 1))
    n = N

    # Build conflict graph: edge (u,v) if u+v is a perfect square (u<=v)
    # Includes self-loops (u==v) when 2u is a square
    adj = [set() for _ in range(n)]
    for i in range(n):
        u = nodes[i]
        # self-loop
        if _is_sq(2 * u):
            adj[i].add(i)
        for j in range(i + 1, n):
            v = nodes[j]
            if _is_sq(u + v):
                adj[i].add(j)
                adj[j].add(i)

    # degrees (count self-loop once)
    deg = [len(adj[i]) for i in range(n)]

    # ---------- Greedy construction ----------
    def greedy_order(order_seed=None):
        """Greedy: pick nodes in some order, include if no conflict with chosen."""
        rng = random.Random(order_seed)
        # Sort candidates by (degree, value) with random tiebreak
        idxs = list(range(n))
        rng.shuffle(idxs)
        # Primary key: ascending degree (pick low-degree first), then random
        # We'll do a priority-based greedy
        chosen = set()
        # Use a simple pass: iterate by degree ascending
        idxs_sorted = sorted(range(n), key=lambda i: (deg[i], nodes[i]))
        for i in idxs_sorted:
            if i in chosen:
                continue
            # check conflict with all chosen
            ok = True
            for j in chosen:
                if j in adj[i]:
                    ok = False
                    break
            if ok:
                chosen.add(i)
        return chosen

    best = greedy_order(0)
    best_size = len(best)

    # Try multiple greedy seeds
    for seed in range(1, 20):
        if time.time() > deadline:
            break
        c = greedy_order(seed)
        if len(c) > best_size:
            best = c
            best_size = len(c)

    # ---------- Local search: add, remove, swap ----------
    def valid_set(s):
        """Return set if valid (no conflicts), else None."""
        lst = sorted(s)
        for a in lst:
            if _is_sq(2 * a):
                return None
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if _is_sq(lst[i] + lst[j]):
                    return None
        return set(lst)

    # Verify best
    v = valid_set(best)
    if v is None:
        # fallback: rebuild safely
        best = set()
        best_size = 0
        for i in range(n):
            if _is_sq(2 * nodes[i]):
                continue
            ok = True
            for j in best:
                if _is_sq(nodes[i] + nodes[j]):
                    ok = False
                    break
            if ok:
                best.add(i)
                best_size = len(best)

    def try_add(cur):
        """Try to add a node; return (new_set, added) or (cur, None)."""
        for i in range(n):
            if i in cur:
                continue
            if _is_sq(2 * nodes[i]):
                continue
            ok = True
            for j in cur:
                if j in adj[i]:
                    ok = False
                    break
            if ok:
                return cur | {i}, i
        return cur, None

    def try_swap(cur):
        """Try 1-out 1-in swap that keeps validity and increases size is impossible (same size).
        Instead try 1-out 2-in to grow by 1."""
        # For each removed r, see if we can add two new nodes a,b that don't conflict with each other
        # and with cur\{r}
        lst = sorted(cur)
        for r in lst:
            rest = cur - {r}
            # candidates: nodes not in cur, not conflicting with rest, not self-square
            cands = []
            for i in range(n):
                if i in rest:
                    continue
                if _is_sq(2 * nodes[i]):
                    continue
                if i == r:
                    continue
                ok = True
                for j in rest:
                    if j in adj[i]:
                        ok = False
                        break
                if ok:
                    cands.append(i)
            # find pair a,b in cands with no conflict between them
            m = len(cands)
            for a_idx in range(m):
                a = cands[a_idx]
                for b_idx in range(a_idx + 1, m):
                    b = cands[b_idx]
                    if b in adj[a]:
                        continue
                    return (rest | {a, b}), (r, a, b)
        return cur, None

    # Iterative local search
    rng = random.Random(12345)
    cur = set(best)
    cur_size = best_size
    stagnation = 0
    max_stagnation = 300

    while time.time() < deadline:
        improved = False

        # Phase 1: add
        new_cur, _ = try_add(cur)
        if len(new_cur) > len(cur):
            cur = new_cur
            improved = True
            stagnation = 0
            if len(cur) > best_size:
                best = set(cur)
                best_size = len(cur)

        # Phase 2: 1-out 2-in
        if not improved:
            new_cur, _ = try_swap(cur)
            if len(new_cur) > len(cur):
                cur = new_cur
                improved = True
                stagnation = 0
                if len(cur) > best_size:
                    best = set(cur)
                    best_size = len(cur)

        # Phase 3: random perturbation if stuck
        if not improved:
            stagnation += 1
            if stagnation >= max_stagnation:
                # perturb: remove a random node, then greedily re-add
                if cur:
                    r = rng.choice(sorted(cur))
                    cur = cur - {r}
                    new_cur, _ = try_add(cur)
                    cur = new_cur
                stagnation = 0
                if len(cur) > best_size:
                    best = set(cur)
                    best_size = len(cur)

        if not improved and len(cur) == best_size:
            # small random restart
            if rng.random() < 0.05:
                # drop a random few and re-add
                drop = min(3, len(cur))
                for _ in range(drop):
                    if cur:
                        cur = cur - {rng.choice(sorted(cur))}
                for _ in range(drop * 2):
                    new_cur, _ = try_add(cur)
                    cur = new_cur
                if len(cur) > best_size:
                    best = set(cur)
                    best_size = len(cur)

    # Final verification
    v = valid_set(best)
    if v is None:
        # rebuild
        best = set()
        for i in range(n):
            if _is_sq(2 * nodes[i]):
                continue
            ok = True
            for j in best:
                if _is_sq(nodes[i] + nodes[j]):
                    ok = False
                    break
            if ok:
                best.add(i)

    return [nodes[i] for i in sorted(best)]
