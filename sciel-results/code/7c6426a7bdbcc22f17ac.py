import numpy as np
import time
from collections import deque
from scipy.linalg import eigvalsh as sp_eigvalsh


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
    k = len(non_tree)  # cycle rank = m + 1

    # Pre-allocate base matrix with tree edges set to +1
    A_base = np.zeros((n, n), dtype=np.float64)
    for idx in range(E):
        if tree_edge[idx]:
            u, v = edges[idx]
            A_base[u, v] = 1.0
            A_base[v, u] = 1.0

    # Pre-extract non-tree edge endpoints for fast update
    non_tree_u = np.array([edges[i][0] for i in non_tree], dtype=np.int64)
    non_tree_v = np.array([edges[i][1] for i in non_tree], dtype=np.int64)

    best_signs = None
    best_radius = float('inf')

    # Exact search over cohomology classes.
    total = 1 << k
    for mask in range(1, total):  # mask == 0 is the trivial (disconnected) class
        if time.time() > deadline:
            break
        
        # Create signs for non-tree edges
        # We can avoid creating a full array if we just update the matrix
        A = A_base.copy()
        
        # Update non-tree edges
        # Using a loop is fine for small k
        for j in range(k):
            if (mask >> j) & 1:
                u = non_tree_u[j]
                v = non_tree_v[j]
                A[u, v] = -1.0
                A[v, u] = -1.0
            else:
                u = non_tree_u[j]
                v = non_tree_v[j]
                A[u, v] = 1.0
                A[v, u] = 1.0
        
        # Compute spectral radius
        evals = sp_eigvalsh(A, check_finite=False)
        r = float(np.max(np.abs(evals)))
        
        if r < best_radius:
            best_radius = r
            # Reconstruct full signs array for output
            signs = np.ones(E, dtype=np.int8)
            for j in range(k):
                if (mask >> j) & 1:
                    signs[non_tree[j]] = -1
            best_signs = signs.copy()

    # Safety fallback: random search with explicit connectivity test.
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(200000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=E)
            if not _lift_connected(n, edges, signs):
                continue
            # Reuse logic for radius
            A = np.zeros((n, n), dtype=np.float64)
            for idx in range(E):
                u, v = edges[idx]
                s = float(signs[idx])
                A[u, v] = s
                A[v, u] = s
            evals = sp_eigvalsh(A, check_finite=False)
            r = float(np.max(np.abs(evals)))
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    return best_signs.astype(np.int64)
