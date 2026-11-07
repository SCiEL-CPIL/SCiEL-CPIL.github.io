"""Rewritten program: SA search over graphs on 15 vertices avoiding equal-degree 3-path endpoints."""
import random
import time

N = 15
M = N * (N - 1) // 2  # 105
TOTAL_EDGES = M

# Precompute adjacency index lists for each vertex (for fast degree/path checks)
NEI = []
for u in range(N):
    lst = [v for v in range(N) if v != u]
    NEI.append(lst)

def _build_adj(adj):
    """adj is a list of 105 bits -> list of neighbor lists."""
    return [[v for v in NEI[u] if adj[u * (N - 1 - u) + (v if v > u else u)]] for u in range(N)]

def _deg(adj):
    return [len(_build_adj(adj)[u]) for u in range(N)]

def _violations(adj):
    """Count of 3-edge paths s-u-v-t (s!=t) with deg(s)==deg(t)."""
    adjl = _build_adj(adj)
    deg = [len(x) for x in adjl]
    v = 0
    for u in range(N):
        nu = adjl[u]
        for v_ in nu:
            if v_ <= u:
                continue
            nv = adjl[v_]
            for s in nu:
                if s == v_:
                    continue
                for t in nv:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        v += 1
    return v

# ---------- Simulated Annealing ----------
def construct():
    start = time.time()
    deadline = start + 78.0  # safety margin under 90s wall-clock
    rng = random.Random(12345)

    # Start from a moderate random graph
    adj = [1 if rng.random() < 0.45 else 0 for _ in range(M)]

    def idx(u, v):
        if u > v:
            u, v = v, u
        return u * (N - 1 - u) + v

    def edge_count(a):
        return sum(a)

    # Current state
    cur_adj = adj[:]
    cur_v = _violations(cur_adj)
    cur_ec = edge_count(cur_adj)

    best_adj = cur_adj[:]
    best_ec = cur_ec
    best_v = cur_v

    T = 30.0
    T_min = 0.001
    cooling = 0.9985

    def propose_move():
        r = rng.random()
        if r < 0.35:
            # 1-add
            i = rng.randrange(M)
            if cur_adj[i]:
                return None
            return ('add', i)
        elif r < 0.55:
            # 1-remove
            ones = [i for i in range(M) if cur_adj[i]]
            if not ones:
                return None
            i = rng.choice(ones)
            return ('rem', i)
        else:
            # 2-swap: add one, remove another
            zeros = [i for i in range(M) if not cur_adj[i]]
            ones = [i for i in range(M) if cur_adj[i]]
            if not zeros or not ones:
                return None
            a = rng.choice(zeros)
            b = rng.choice(ones)
            if a == b:
                return None
            return ('swap', a, b)

    it = 0
    while time.time() < deadline:
        mv = propose_move()
        if mv is None:
            # try to fill zeros
            mv = ('add', rng.randrange(M))
            if cur_adj[mv[1]]:
                continue

        # Build candidate adjacency
        cand = cur_adj[:]
        if mv[0] == 'add':
            cand[mv[1]] = 1
            cand_ec = cur_ec + 1
        elif mv[0] == 'rem':
            cand[mv[1]] = 0
            cand_ec = cur_ec - 1
        else:  # swap
            cand[mv[1]] = 1
            cand[mv[2]] = 0
            cand_ec = cur_ec

        cand_v = _violations(cand)

        # Acceptance
        if cand_v == 0:
            # prefer more edges
            if cand_ec >= cur_ec:
                accept = True
            else:
                d = cand_ec - cur_ec
                accept = rng.random() < min(1.0, (T / (abs(d) + 1e-9)))
        else:
            # invalid: accept with probability depending on violation penalty
            # score = ec - K*v ; accept if better or with Boltzmann
            K = 2.0
            cur_score = cur_ec - K * cur_v
            cand_score = cand_ec - K * cand_v
            d = cand_score - cur_score
            if d >= 0:
                accept = True
            else:
                accept = rng.random() < min(1.0, (T / (abs(d) + 1e-9)))

        if accept:
            cur_adj = cand
            cur_ec = cand_ec
            cur_v = cand_v
            if cur_v == 0 and cur_ec > best_ec:
                best_ec = cur_ec
                best_adj = cur_adj[:]
                best_v = 0

        T = max(T_min, T * cooling)
        it += 1

    # Final polish: try greedy additions on the best valid graph
    if best_v == 0:
        pol = best_adj[:]
        pol_ec = best_ec
        # repeatedly add any edge that keeps validity
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(M):
                if pol[i]:
                    continue
                cand = pol[:]
                cand[i] = 1
                if _violations(cand) == 0:
                    pol = cand
                    pol_ec += 1
                    changed = True
        if pol_ec > best_ec:
            best_ec = pol_ec
            best_adj = pol[:]

    # Build edge list
    edges = []
    for u in range(N):
        for v in NEI[u]:
            if v <= u:
                continue
            if best_adj[idx(u, v)]:
                edges.append([int(u), int(v)])

    # Return as list of [int, int]
    return edges
