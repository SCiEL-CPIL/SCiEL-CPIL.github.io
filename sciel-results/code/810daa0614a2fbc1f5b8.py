"""Search-based construction for the 16-vertex / 65-edge triangle-degree-sum task."""
import time
import random
import numpy as np

N = 16
M = 65
ALL = [(i, j) for i in range(N) for j in range(i + 1, N)]
ALL_KEYS = np.array(ALL, dtype=np.int64)
NPAIRS = len(ALL)
# map (i,j) with i<j -> index
_KEY2IDX = { (i, j): k for k, (i, j) in enumerate(ALL) }


def _norm(e):
    a, b = int(e[0]), int(e[1])
    return (a, b) if a < b else (b, a)


def _score(adj, deg):
    """max over triangles of deg sum; returns (score, num_triangles)."""
    best = 0
    cnt = 0
    for i in range(N):
        ni = np.nonzero(adj[i, :])[0]
        for ii in range(len(ni)):
            u = ni[ii]
            for jj in range(ii + 1, len(ni)):
                v = ni[jj]
                if adj[u, v]:
                    s = deg[i] + deg[u] + deg[v]
                    if s > best:
                        best = s
                    cnt += 1
    return best, cnt


def _init_adj(seed):
    """K_{8,8} plus a random internal edge, shuffled."""
    rng = random.Random(seed)
    adj = np.zeros((N, N), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = adj[v, u] = 1
    # one internal edge (random, to give the search some room)
    a, b = rng.sample(range(8), 2)
    adj[a, b] = adj[b, a] = 1
    return adj


def _edge_indices(adj):
    idx = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                idx.append(_KEY2IDX[(i, j)])
    return idx


def _random_walk_optimize(adj, deg, iters, t0, t1, deadline, rng):
    """Simulated annealing over single-edge move neighborhood."""
    best_adj = adj.copy()
    best_deg = deg.copy()
    best_score, best_cnt = _score(best_adj, best_deg)
    cur_adj = adj.copy()
    cur_deg = deg.copy()
    cur_score, _ = _score(cur_adj, cur_deg)
    cur_edges = _edge_indices(cur_adj)

    # candidate edges not currently in the graph
    cur_set = set(cur_edges)
    non_edges = [k for k in range(NPAIRS) if k not in cur_set]

    for it in range(iters):
        if time.time() > deadline:
            break
        t = t0 + (t1 - t0) * (it / max(1, iters))
        # choose a random edge to remove
        ei = cur_edges[rng.randrange(len(cur_edges))]
        i, j = ALL[ei]
        # choose a random non-edge to add
        ek = non_edges[rng.randrange(len(non_edges))]
        a, b = ALL[ek]

        # apply
        cur_adj[i, j] = 0; cur_adj[j, i] = 0
        cur_adj[a, b] = 1; cur_adj[b, a] = 1
        cur_deg[i] -= 1; cur_deg[j] -= 1
        cur_deg[a] += 1; cur_deg[b] += 1

        ns, _ = _score(cur_adj, cur_deg)
        delta = ns - cur_score
        if delta <= 0 or rng.random() < np.exp(-delta / max(t, 1e-9)):
            cur_score = ns
            cur_edges[ei] = ek
            cur_set.remove(ei); cur_set.add(ek)
            non_edges.remove(ek); non_edges.append(ei)
            if ns < best_score:
                best_score = ns
                best_adj = cur_adj.copy()
                best_deg = cur_deg.copy()
                best_cnt = _
        else:
            # revert
            cur_adj[i, j] = 1; cur_adj[j, i] = 1
            cur_adj[a, b] = 0; cur_adj[b, a] = 0
            cur_deg[i] += 1; cur_deg[j] += 1
            cur_deg[a] -= 1; cur_deg[b] -= 1
            cur_edges[ei] = ei  # unchanged (was ei)
            # fix lists: we appended ei and removed ek earlier only on accept; nothing changed here
    return best_adj, best_deg, best_score


def construct():
    deadline = time.time() + 115.0  # safety margin before 120s
    rng = random.Random(12345)

    global_best_adj = None
    global_best_score = 10**9
    global_best_deg = None

    # --- Phase 1: many annealing restarts ---
    n_restarts = 40
    iters_per = 2500
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        adj = _init_adj(rng.randrange(10**9))
        deg = adj.sum(axis=1).astype(np.int64)
        b_adj, b_deg, b_score = _random_walk_optimize(
            adj, deg, iters_per, 5.0, 0.05, deadline, rng)
        if b_score < global_best_score:
            global_best_score = b_score
            global_best_adj = b_adj
            global_best_deg = b_deg

    if global_best_adj is None:
        adj = _init_adj(0)
        global_best_adj = adj
        global_best_deg = adj.sum(axis=1).astype(np.int64)
        global_best_score = _score(adj, global_best_deg)[0]

    # --- Phase 2: local search polish (greedy improving single-edge moves) ---
    adj = global_best_adj.copy()
    deg = global_best_deg.copy()
    cur_score, _ = _score(adj, deg)
    cur_edges = _edge_indices(adj)
    cur_set = set(cur_edges)
    non_edges = [k for k in range(NPAIRS) if k not in cur_set]
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for ei in list(cur_edges):
            if time.time() > deadline - 5.0:
                break
            i, j = ALL[ei]
            for ek in non_edges:
                a, b = ALL[ek]
                # quick degree-based prune: estimate change
                cur_adj = adj
                cur_adj[i, j] = 0; cur_adj[j, i] = 0
                cur_adj[a, b] = 1; cur_adj[b, a] = 1
                deg[i] -= 1; deg[j] -= 1
                deg[a] += 1; deg[b] += 1
                ns, _ = _score(cur_adj, deg)
                # revert for evaluation
                cur_adj[i, j] = 1; cur_adj[j, i] = 1
                cur_adj[a, b] = 0; cur_adj[b, a] = 0
                deg[i] += 1; deg[j] += 1
                deg[a] -= 1; deg[b] -= 1
                if ns < cur_score:
                    # accept
                    adj[i, j] = 0; adj[j, i] = 0
                    adj[a, b] = 1; adj[b, a] = 1
                    deg[i] -= 1; deg[j] -= 1
                    deg[a] += 1; deg[b] += 1
                    cur_score = ns
                    cur_edges[ei] = ek
                    cur_set.remove(ei); cur_set.add(ek)
                    non_edges.remove(ek); non_edges.append(ei)
                    improved = True
                    if cur_score <= 23:
                        improved = False
                        break
            if cur_score <= 23:
                break
        if cur_score <= 23:
            break

    # final
    if cur_score < global_best_score:
        global_best_score = cur_score
        global_best_adj = adj
        global_best_deg = deg

    edges = [[int(i), int(j)] for i in range(N) for j in range(i + 1, N) if global_best_adj[i, j]]
    assert len(edges) == M, len(edges)
    return {'n': N, 'edges': edges}
