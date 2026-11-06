import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin (evaluator kills at 150s)
    n = 40

    def now_dead():
        return time.time() >= deadline

    def greedy(seed, hard_deadline):
        ra = [0] * n
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.seed(seed)
        random.shuffle(pairs)

        def creates2(u, v):
            mask_uv = (1 << u) | (1 << v)
            nu = ra[u] & ~(1 << v)
            nv = ra[v] & ~(1 << u)
            a = nu
            while a:
                lb = a & (-a)
                ai = lb.bit_length() - 1
                a ^= lb
                d = nv & ~lb
                while d:
                    ld = d & (-d)
                    di = ld.bit_length() - 1
                    d ^= ld
                    bmask = ra[ai] & ~lb & ~ld & ~mask_uv
                    b = bmask
                    while b:
                        lb2 = b & (-b)
                        bi = lb2.bit_length() - 1
                        b ^= lb2
                        if ra[bi] & ld & ~lb & ~mask_uv & ~lb2:
                            return True
            return False

        for (u, v) in pairs:
            if time.time() >= hard_deadline:
                break
            if not ((ra[u] >> v) & 1) and not creates2(u, v):
                ra[u] |= 1 << v
                ra[v] |= 1 << u
        return ra, sum(ra[u].bit_count() for u in range(n)) // 2

    def verify_c6_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    continue
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

                def creates(u, v):
                    mask_uv = (1 << u) | (1 << v)
                    nu = adj[u] & ~(1 << v)
                    nv = adj[v] & ~(1 << u)
                    a = nu
                    while a:
                        lb = a & (-a)
                        ai = lb.bit_length() - 1
                        a ^= lb
                        d = nv & ~lb
                        while d:
                            ld = d & (-d)
                            di = ld.bit_length() - 1
                            d ^= ld
                            bmask = adj[ai] & ~lb & ~ld & ~mask_uv
                            b = bmask
                            while b:
                                lb2 = b & (-b)
                                bi = lb2.bit_length() - 1
                                b ^= lb2
                                if adj[bi] & ld & ~lb & ~mask_uv & ~lb2:
                                    return True
                    return False

                bad = creates(u, v)
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                if bad:
                    return False
        return True

    # Fallback guaranteed-valid: 8 x K5
    def fallback():
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([5 * k + a, 5 * k + b])
        return edges

    best_adj = None
    best_cnt = 0
    attempt = 0
    while time.time() < deadline - 3:
        attempt += 1
        ra, cnt = greedy(1000 + attempt, deadline - 3)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = ra[:]

    if best_adj is None or not verify_c6_free(best_adj[:]):
        return fallback()

    edges = []
    for u in range(n):
        m = best_adj[u]
        for v in range(u + 1, n):
            if (m >> v) & 1:
                edges.append([u, v])
    return edges
