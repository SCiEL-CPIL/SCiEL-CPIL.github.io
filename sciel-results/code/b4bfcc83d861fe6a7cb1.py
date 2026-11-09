import numpy as np
import time
from collections import deque


def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # explicit guard, well under the 60 s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    eu = np.array([e[0] for e in edges], dtype=np.int64)
    ev = np.array([e[1] for e in edges], dtype=np.int64)

    # Adjacency with edge indices
    adj = [[] for _ in range(n)]
    for idx in range(E):
        u, v = edges[idx]
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    # Spanning tree (BFS)
    in_tree = [False] * n
    tree_edge = [False] * E
    in_tree[0] = True
    q = deque([0])
    while q:
        u = q.popleft()
        for v, idx in adj[u]:
            if not in_tree[v]:
                in_tree[v] = True
                tree_edge[idx] = True
                q.append(v)

    non_tree = [i for i in range(E) if not tree_edge[i]]
    k = len(non_tree)  # cycle rank

    best_signs = None
    best_radius = float('inf')

    # Precompute base adjacency with all +1 signs
    base_signs_f = np.ones(E, dtype=np.float64)
    A_base = np.zeros((n, n), dtype=np.float64)
    np.add.at(A_base, (eu, ev), base_signs_f)
    np.add.at(A_base, (ev, eu), base_signs_f)

    # Precompute: for each non-tree edge, the (u,v) pair
    non_tree_uv = [(eu[non_tree[j]], ev[non_tree[j]]) for j in range(k)]
    
    # Exhaustive search over all cohomology classes (2^k possibilities)
    # mask == 0 is the trivial (disconnected) class
    total = 1 << k
    for mask in range(1, total):
        if time.time() > deadline:
            break
        
        # Start with A_base and apply the -1 signs for non-tree edges set in mask
        A = A_base.copy()
        for j in range(k):
            if (mask >> j) & 1:
                u, v = non_tree_uv[j]
                A[u, v] -= 2.0  # was +1, now -1
                A[v, u] -= 2.0
        
        r = float(np.max(np.abs(np.linalg.eigvalsh(A))))
        if r < best_radius:
            best_radius = r
            # Reconstruct the signs array
            signs = np.ones(E, dtype=np.int8)
            for j in range(k):
                if (mask >> j) & 1:
                    signs[non_tree[j]] = -1
            best_signs = signs.copy()

    # If exhaustive search didn't find anything (shouldn't happen for m <= 9),
    # fall back to random search
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(200000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=E)
            # Check connectivity
            if not _lift_connected(n, edges, signs):
                continue
            A = np.zeros((n, n), dtype=np.float64)
            np.add.at(A, (eu, ev), signs.astype(np.float64))
            np.add.at(A, (ev, eu), signs.astype(np.float64))
            r = float(np.max(np.abs(np.linalg.eigvalsh(A))))
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    return best_signs.astype(np.int64)


def _lift_connected(n, edges, signs):
    """True iff the 2-lift is connected (sign assignment is not a coboundary)."""
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        s = int(signs[idx])
        adj[u].append((v, s))
        adj[v].append((u, s))
    c = [-1] * n
    c[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        cu = c[u]
        for v, s in adj[u]:
            exp = (cu + (1 if s == -1 else 0)) % 2
            if c[v] == -1:
                c[v] = exp
                q.append(v)
            elif c[v] != exp:
                return True
    return False
