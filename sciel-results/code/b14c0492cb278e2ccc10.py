import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    # BFS to find spanning tree
    visited = [False] * n
    visited[0] = True
    queue = deque([0])
    tree_edges = set()

    while queue:
        u = queue.popleft()
        for v, eidx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(eidx)
                queue.append(v)

    non_tree_edges = [i for i in range(E) if i not in tree_edges]
    num_non_tree = len(non_tree_edges)

    # Build fundamental cycles
    def get_path_in_tree(u, v):
        visited_set = {u}
        queue2 = deque([(u, [])])
        while queue2:
            node, path = queue2.popleft()
            if node == v:
                return path
            for neighbor, eidx in adj[node]:
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    queue2.append((neighbor, path + [eidx]))
        return None

    fundamental_cycles = []
    for eidx in non_tree_edges:
        u, v = edges[eidx]
        path = get_path_in_tree(u, v)
        if path is not None:
            fundamental_cycles.append(path + [eidx])

    # Build GF(2) incidence matrix M: num_cycles x num_edges
    num_cycles = num_non_tree
    M = np.zeros((num_cycles, num_edges), dtype=np.int8)
    for j, cycle_edges in enumerate(fundamental_cycles):
        for e in cycle_edges:
            M[j, e] = 1

    # Fast connectivity check for the 2-lift
    def is_lift_connected(signs):
        # Build lift adjacency using bitmasks for speed
        # Vertex (v, s) -> index v*2+s
        lift_adj = [[] for _ in range(2 * n)]
        for eidx, (u, v) in enumerate(edges):
            s = signs[eidx]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)

        # BFS
        vis = [False] * (2 * n)
        q = deque([0])
        vis[0] = True
        cnt = 1
        while q:
            node = q.popleft()
            for nb in lift_adj[node]:
                if not vis[nb]:
                    vis[nb] = True
                    cnt += 1
                    q.append(nb)
        return cnt == 2 * n

    def compute_radius(signs):
        A_mat = np.zeros((n, n), dtype=np.float64)
        for eidx, (u, v) in enumerate(edges):
            s = float(signs[eidx])
            A_mat[u, v] = s
            A_mat[v, u] = s
        eigenvalues = np.linalg.eigvalsh(A_mat)
        return np.max(np.abs(eigenvalues))

    # Solve M x = b over GF(2) - optimized
    def solve_gf2(M_mat, b_vec):
        A = np.hstack([M_mat.copy(), b_vec.reshape(-1, 1)])
        n_rows, n_cols = M_mat.shape
        rank = 0
        pivot_cols = []
        for col in range(n_cols):
            pivot_row = -1
            for row in range(rank, n_rows):
                if A[row, col] == 1:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
            for row in range(n_rows):
                if row != rank and A[row, col] == 1:
                    A[row] = (A[row] + A[rank]) % 2
            pivot_cols.append(col)
            rank += 1
            if rank == n_rows:
                break

        # Check consistency
        for row in range(rank, n_rows):
            if A[row, n_cols] == 1:
                return None

        x = np.zeros(n_cols, dtype=np.int8)
        for i, col in enumerate(pivot_cols):
            x[col] = A[i, n_cols]
        return x

    # Strategy: enumerate all 2^{m+1} switching classes
    # Prioritize balanced configurations (all cycle products = -1)
    best_radius = float('inf')
    best_signs = None

    num_classes = 1 << num_non_tree

    # Generate masks in order of "balancedness" - prefer all -1 cycle products
    # All -1 means b = [1, 1, ..., 1] which is mask = (1 << num_non_tree) - 1
    all_neg_mask = (1 << num_non_tree) - 1

    # Order: start with all_neg_mask, then flip one bit, then two, etc.
    # For efficiency, just iterate all but check all_neg first
    masks_to_try = [all_neg_mask]
    for mask in range(num_classes):
        if mask != all_neg_mask:
            masks_to_try.append(mask)

    for mask in masks_to_try:
        if time.time() > deadline:
            break

        b = np.zeros(num_cycles, dtype=np.int8)
        for j in range(num_cycles):
            if (mask >> j) & 1:
                b[j] = 1

        x = solve_gf2(M, b)
        if x is None:
            continue

        signs = np.ones(num_edges, dtype=np.int8)
        signs[x == 1] = -1

        if not is_lift_connected(signs):
            continue

        radius = compute_radius(signs)

        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    # Local search polish: try flipping individual signs
    if best_signs is not None and time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for eidx in range(E):
                if time.time() > deadline - 2.0:
                    break
                trial = best_signs.copy()
                trial[eidx] = -trial[eidx]
                if not is_lift_connected(trial):
                    continue
                r = compute_radius(trial)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = trial
                    improved = True

    if best_signs is not None:
        return best_signs
    return np.ones(E, dtype=np.int8)
