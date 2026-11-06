"""Minimize 4-cycles in a 31-vertex, 91-edge graph."""
import time
import random
import numpy as np

N = 31
M = 91
TIME_LIMIT = 110.0

def count_c4(adj):
    total = 0
    for i in range(N):
        ai = adj[i]
        for j in range(i + 1, N):
            aj = adj[j]
            c = 0
            if len(ai.nonzero()[0]) <= len(aj.nonzero()[0]):
                for k in ai.nonzero()[0]:
                    if aj[k]:
                        c += 1
            else:
                for k in aj.nonzero()[0]:
                    if ai[k]:
                        c += 1
            if c >= 2:
                total += c * (c - 1) // 2
    return total

def build_polarity_graph():
    q = 5
    pts = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                v = [a, b, c]
                if v == [0, 0, 0]:
                    continue
                for i in range(3):
                    if v[i] != 0:
                        inv = pow(v[i], q - 2, q)
                        v = [(x * inv) % q for x in v]
                        break
                pts.append(tuple(v))
    pts = list(dict.fromkeys(pts))
    adj = [np.zeros(N, dtype=bool) for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a = pts[i]; b = pts[j]
            if (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) % q == 0:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def edges_list(adj):
    return [(i, j) for i in range(N) for j in range(i+1, N) if adj[i][j]]

def add_edge_to_reach_m(adj):
    while len(edges_list(adj)) < M:
        best = None; best_delta = None
        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    continue
                c = 0
                for k in range(N):
                    if adj[i][k] and adj[j][k]:
                        c += 1
                if best_delta is None or c < best_delta:
                    best_delta = c; best = (i, j)
        adj[best[0]][best[1]] = True
        adj[best[1]][best[0]] = True
    return adj

def local_search(adj, seed, deadline):
    rng = random.Random(seed)
    adj = [row.copy() for row in adj]
    cur = count_c4(adj)
    best = cur
    best_adj = [row.copy() for row in adj]
    while time.time() < deadline:
        for _ in range(150):
            if time.time() > deadline:
                break
            e = edges_list(adj)
            rem = rng.choice(e)
            while True:
                a = rng.randrange(N); b = rng.randrange(N)
                if a == b:
                    continue
                if a > b: a, b = b, a
                if not adj[a][b]:
                    break
            add = (a, b)
            if rem == add:
                continue
            adj[rem[0]][rem[1]] = False; adj[rem[1]][rem[0]] = False
            adj[add[0]][add[1]] = True; adj[add[1]][add[0]] = True
            new = count_c4(adj)
            if new < cur:
                cur = new
                if cur < best:
                    best = cur
                    best_adj = [row.copy() for row in adj]
            else:
                adj[add[0]][add[1]] = False; adj[add[1]][add[0]] = False
                adj[rem[0]][rem[1]] = True; adj[rem[1]][rem[0]] = True
    return best_adj, best

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT
    adj = build_polarity_graph()
    if len(edges_list(adj)) < M:
        adj = add_edge_to_reach_m(adj)
    elif len(edges_list(adj)) > M:
        e = edges_list(adj)
        rng = random.Random(0)
        while len(e) > M:
            rem = rng.choice(e)
            e.remove(rem)
            adj[rem[0]][rem[1]] = False
            adj[rem[1]][rem[0]] = False
    best_adj = [row.copy() for row in adj]
    best_c = count_c4(adj)
    for seed in range(60):
        if time.time() > deadline - 5:
            break
        cand_adj, cand_c = local_search(adj, seed, deadline - 2)
        if cand_c < best_c:
            best_c = cand_c
            best_adj = cand_adj
    return [[i, j] for i, j in edges_list(best_adj)]
