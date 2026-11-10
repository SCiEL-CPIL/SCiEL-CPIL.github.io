"""Improved program: greedy edge addition from star baseline."""
# EVOLVE-BLOCK-START
import time

_N = 15
_DEADLINE = 80.0  # safety margin before 90s kill

def _degrees(adj):
    return [sum(adj[i][j] for j in range(_N)) for i in range(_N)]

def _is_valid(adj, deg):
    """Check no 3-edge path s-u-v-t (s!=t) has deg(s)==deg(t)."""
    for u in range(_N):
        for v in range(u + 1, _N):
            if not adj[u][v]:
                continue
            # neighbors of u excluding v
            nu = [x for x in range(_N) if x != v and adj[u][x]]
            nv = [x for x in range(_N) if x != u and adj[v][x]]
            for s in nu:
                for t in nv:
                    if s != t and deg[s] == deg[t]:
                        return False
    return True

def construct():
    start = time.time()
    # Start from star centered at 0
    adj = [[0] * _N for _ in range(_N)]
    for v in range(1, _N):
        adj[0][v] = 1
        adj[v][0] = 1
    deg = _degrees(adj)

    # Candidate edges sorted by heuristic: prefer leaf-leaf (both non-center)
    # then edges that keep degrees more "spread out"
    def edge_key(e):
        u, v = e
        # Prefer edges between vertices with lower current degree (leaves)
        # to create more degree diversity
        return (min(deg[u], deg[v]), max(deg[u], deg[v]))

    candidates = [(u, v) for u in range(_N) for v in range(u + 1, _N) if not adj[u][v]]
    candidates.sort(key=edge_key)

    best_edges = [[0, v] for v in range(1, _N)]
    best_count = _N - 1

    for u, v in candidates:
        if time.time() - start > _DEADLINE:
            break
        # Try adding edge (u, v)
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
        if _is_valid(adj, deg):
            best_edges.append([u, v])
            best_count += 1
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
            deg[u] -= 1
            deg[v] -= 1

    # Ensure proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    return result

# EVOLVE-BLOCK-END
