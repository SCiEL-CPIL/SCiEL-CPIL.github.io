"""Optimize triangle degree-sum for 16-vertex, 65-edge graphs."""
import random
import time

def _score(adj):
    n = 16
    deg = [0] * n
    for u in range(n):
        du = 0
        for v in range(u + 1, n):
            if adj[u][v]:
                du += 1
        deg[u] = du
    best = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            for w in range(v + 1, n):
                if adj[u][w] and adj[v][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > best:
                        best = s
    return best

def _edges_from_adj(adj):
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if adj[u][v]:
                edges.append([u, v])
    return edges

def _init_adj():
    adj = [[False] * 16 for _ in range(16)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = adj[v][u] = True
    for u in range(8):
        for v in range(u + 1, 8):
            if random.random() < 0.15:
                adj[u][v] = adj[v][u] = True
    for u in range(8, 16):
        for v in range(u + 1, 16):
            if random.random() < 0.15:
                adj[u][v] = adj[v][u] = True
    return adj

def _make_feasible(adj):
    # Add random missing edges until 65
    missing = [(u, v) for u in range(16) for v in range(u + 1, 16) if not adj[u][v]]
    random.shuffle(missing)
    count = sum(1 for u in range(16) for v in range(u + 1, 16) if adj[u][v])
    for u, v in missing:
        if count >= 65:
            break
        adj[u][v] = adj[v][u] = True
        count += 1
    return adj

def _make_feasible_v2(adj):
    # If over 65, remove random present edges
    present = [(u, v) for u in range(16) for v in range(u + 1, 16) if adj[u][v]]
    count = len(present)
    if count > 65:
        random.shuffle(present)
        for u, v in present:
            if count <= 65:
                break
            adj[u][v] = adj[v][u] = False
            count -= 1
    return adj

def _swap_adj(adj):
    # Pick a present edge and a missing edge, swap
    present = [(u, v) for u in range(16) for v in range(u + 1, 16) if adj[u][v]]
    missing = [(u, v) for u in range(16) for v in range(u + 1, 16) if not adj[u][v]]
    if not present or not missing:
        return adj, False
    pu, pv = random.choice(present)
    mu, mv = random.choice(missing)
    if (pu, pv) == (mu, mv) or (pu, pv) == (mv, mu):
        return adj, False
    adj[pu][pv] = adj[pv][pu] = False
    adj[mu][mv] = adj[mv][mu] = True
    return adj, True

def construct():
    deadline = time.time() + 120
    best_score = 10**9
    best_adj = None

    # Try several random restarts
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        adj = _init_adj()
        adj = _make_feasible(adj)
        adj = _make_feasible_v2(adj)
        s = _score(adj)
        if s < best_score:
            best_score = s
            best_adj = [row[:] for row in adj]
        if best_score <= 23:
            break

    # Local search / simulated annealing on best
    if best_adj is not None:
        adj = [row[:] for row in best_adj]
        cur_score = best_score
        T = 1.0
        steps = 0
        while time.time() < deadline - 20 and steps < 200000:
            steps += 1
            new_adj, ok = _swap_adj(adj)
            if not ok:
                continue
            new_score = _score(new_adj)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < (0.01 * (-delta) + 0.001):
                adj = new_adj
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj = [row[:] for row in adj]
                    if best_score <= 23:
                        break
            T *= 0.999
        # Deterministic polish: try every single swap
        if time.time() < deadline - 5 and best_score > 23:
            adj = [row[:] for row in best_adj]
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                base_score = _score(adj)
                for u in range(16):
                    if time.time() > deadline - 5:
                        break
                    for v in range(u + 1, 16):
                        if not adj[u][v]:
                            continue
                        # try removing (u,v) and adding some missing edge
                        adj[u][v] = adj[v][u] = False
                        for mu in range(16):
                            if time.time() > deadline - 5:
                                break
                            for mv in range(mu + 1, 16):
                                if adj[mu][mv]:
                                    continue
                                if (mu, mv) == (u, v):
                                    continue
                                adj[mu][mv] = adj[mv][mu] = True
                                s = _score(adj)
                                adj[mu][mv] = adj[mv][mu] = False
                                if s < base_score:
                                    adj[u][v] = adj[v][u] = False
                                    adj[mu][mv] = adj[mv][mu] = True
                                    best_adj = [row[:] for row in adj]
                                    best_score = s
                                    improved = True
                                    break
                            if improved:
                                break
                        if not improved:
                            adj[u][v] = adj[v][u] = True
                        if improved:
                            break
                    if improved:
                        break
                if best_score <= 23:
                    break

    # Fallback: known good construction
    if best_adj is None:
        adj = [[False] * 16 for _ in range(16)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = adj[v][u] = True
        # Add 4 within-A edges forming a path, 4 within-B edges forming a path
        for i in range(4):
            adj[i][i + 1] = adj[i + 1][i] = True
        for i in range(4):
            adj[8 + i][8 + i + 1] = adj[8 + i + 1][8 + i] = True
        best_adj = adj
        best_score = _score(adj)

    edges = _edges_from_adj(best_adj)
    assert len(edges) == 65, f"expected 65 edges, got {len(edges)}"
    return {'n': 16, 'edges': edges}
