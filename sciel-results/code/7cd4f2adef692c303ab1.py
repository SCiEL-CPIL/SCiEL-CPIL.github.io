"""R_4(C6): 4-colouring of K_n with no monochromatic 6-cycle.
Vectorized C6-count objective + violation-driven local search with annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# ---------- fast lexicographic edge indexing ----------
def _edge_offsets(n):
    # offset[i] = number of edges (i',j') with i' < i
    return np.array([i * (n - 1) - i * (i - 1) // 2 for i in range(n)], dtype=np.int64)

def _edge_matrix(n):
    # E[i, j] = index of edge (i, j) for i < j ; -1 otherwise
    E = np.full((n, n), -1, dtype=np.int64)
    off = _edge_offsets(n)
    for i in range(n):
        for j in range(i + 1, n):
            E[i, j] = off[i] + (j - i - 1)
    return E

# ---------- vectorized monochromatic C6 counter ----------
def _count_c6(color, n, E):
    """Count distinct 6-cycles in the graph whose edges all have given colour."""
    if n < 6:
        return 0
    m = n * (n - 1) // 2
    # adjacency matrix A (symmetric, 0/1) for this colour
    A = np.zeros((n, n), dtype=np.int64)
    rows = np.repeat(np.arange(n), n - 1 - np.arange(n))
    cols = np.concatenate([np.arange(i + 1, n) for i in range(n)])
    idx = E[rows, cols]
    A[rows, cols] = (color == colors_global[idx]).astype(np.int64)
    A[cols, rows] = A[rows, cols]

    # P = A^2  (number of length-2 paths between each pair)
    P = A @ A
    # B[v0, v3] = number of (v1, v2) with v0-v1-v2-v3 all in colour, v1 != v0, v2 != v3
    B = A @ P @ A
    # C[v0, v3] = number of (v4, v5) with v3-v4-v5-v0 all in colour, v4 != v3, v5 != v0
    C = A @ P @ A  # symmetric in the same sense; C[v0,v3] = # (v4,v5): v3~v4~v5~v0
    # For each (v0, v3): each v1 (v0~v1) and each v4 (v3~v4) with v1 != v4
    #   gives (B[v0,v3]) * (C[v0,v3]) - (number of shared middle) combinations.
    # Count = sum over v0<v3 of [ (A[v0,1]*...) ] -- use the standard formula:
    # total directed count = sum_{v0,v3} B[v0,v3] * C[v0,v3] - sum_{v0,v3} (shared)
    # where shared = number of (v1) with v0~v1 and v3~v1 and v1 adjacent such that
    #   v1 also = v4 (i.e. the two middle paths share the same vertex).
    # The shared term = (A @ A)[v0, v3] restricted to v1 != v0, v3, v2=v1... 
    # Simpler: shared = number of v1 with v0~v1~v3 and v3~v1~v0 = (A@A)[v0,v3] with v1!=v0,v3.
    #   Since v1~v0 and v1~v3 => v1 is a common neighbour; A@A[v0,v3] counts exactly
    #   common neighbours (v1) because v0~v1~v3. And we need v1 != v0, v3 (true, no loops).
    shared = P  # P[v0,v3] = # common neighbours = # v1 with v0~v1~v3
    # But B already requires v1!=v0 (v1~v0) and v2!=v3; C requires v4!=v3 and v5!=v0.
    # The overcount is exactly when v1 == v4 (the two length-2 paths share their start-mid).
    # That happens for each common neighbour v1 of v0 and v3: contributes 1 each.
    total = int(np.sum(B * C) - np.sum(shared * (A * A)))
    # each undirected 6-cycle is counted 12 times (6 rotations * 2 directions)
    return total // 12

# We compute C6 count for all 4 colours at once for speed.
def total_c6_count(colors_arr, n, E):
    if n < 6:
        return 0
    rows = np.repeat(np.arange(n), n - 1 - np.arange(n))
    cols = np.concatenate([np.arange(i + 1, n) for i in range(n)])
    idx = E[rows, cols]
    col = colors_arr[idx]
    total = 0
    for c in range(4):
        A = np.zeros((n, n), dtype=np.int64)
        A[rows, cols] = (col == c).astype(np.int64)
        A[cols, rows] = A[rows, cols]
        P = A @ A
        B = A @ P @ A
        C = B  # symmetric role
        shared = P
        total += int(np.sum(B * C) - np.sum(shared * (A * A)))
    return total // 12

def _find_one_c6(color, n, E, colors_arr):
    """Return a list of 6 vertex indices forming a C6 in `color`, or None."""
    if n < 6:
        return None
    rows = np.repeat(np.arange(n), n - 1 - np.arange(n))
    cols = np.concatenate([np.arange(i + 1, n) for i in range(n)])
    idx = E[rows, cols]
    A = np.zeros((n, n), dtype=np.int64)
    A[rows, cols] = (colors_arr[idx] == color).astype(np.int64)
    A[cols, rows] = A[rows, cols]
    # Build adjacency lists
    adj = [np.nonzero(A[v])[0].tolist() for v in range(n)]
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v2 in adj[v1]:
                if v2 in (v0,):
                    continue
                for v3 in adj[v2]:
                    if v3 in (v0, v1):
                        continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2, v3):
                            continue
                        for v5 in adj[v4]:
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if A[v5, v0]:
                                return [v0, v1, v2, v3, v4, v5]
    return None

# ---------- search ----------
def _local_search(colors_arr, n, E, deadline, rng, max_iter):
    """Minimize total C6 count. Returns (colors_arr, min_count)."""
    cur = int(total_c6_count(colors_arr, n, E))
    best = colors_arr.copy()
    best_c = cur
    temp = 1.0
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if cur == 0:
            break
        # pick a violating colour
        viol = [c for c in range(4) if _has_c6_fast(c, n, E, colors_arr)]
        if not viol:
            break
        c = viol[rng.randrange(len(viol))]
        cyc = _find_one_c6(c, n, E, colors_arr)
        if cyc is None:
            # no cycle found by explicit search; fall back to random edge in colour c
            break
        # choose an edge of the cycle to recolour
        e = cyc[rng.randrange(6)]
        u, v = cyc[e], cyc[(e + 1) % 6]
        if u > v:
            u, v = v, u
        pos = int(E[u, v])
        old = int(colors_arr[pos])
        new = rng.randrange(4)
        while new == old:
            new = rng.randrange(4)
        colors_arr[pos] = new
        newc = int(total_c6_count(colors_arr, n, E))
        if newc < cur or (newc == cur and rng.random() < temp):
            cur = newc
            if cur < best_c:
                best_c = cur
                best = colors_arr.copy()
        else:
            colors_arr[pos] = old  # revert
        temp = max(0.02, temp * 0.9995)
    if best_c < cur:
        colors_arr[:] = best
    return colors_arr, best_c

def _has_c6_fast(color, n, E, colors_arr):
    rows = np.repeat(np.arange(n), n - 1 - np.arange(n))
    cols = np.concatenate([np.arange(i + 1, n) for i in range(n)])
    idx = E[rows, cols]
    A = np.zeros((n, n), dtype=np.int64)
    A[rows, cols] = (colors_arr[idx] == color).astype(np.int64)
    A[cols, rows] = A[rows, cols]
    P = A @ A
    B = A @ P @ A
    shared = P
    val = int(np.sum(B * B) - np.sum(shared * (A * A)))
    return val // 12 > 0

def construct():
    global colors_global
    start = time.time()
    deadline = start + 140.0  # 10s safety margin before the 150s kill
    rng = random.Random(12345)

    # guaranteed valid fallback (seed's circular-distance K9)
    n0 = 9
    best_n = n0
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]

    for n in range(10, 21):
        if time.time() > deadline - 5.0:
            break
        num_edges = n * (n - 1) // 2
        E = _edge_matrix(n)
        found = False
        # budget for this n
        n_deadline = min(deadline, start + (140.0 - (n - 10) * 6.0))
        if n_deadline <= start + 2.0:
            break
        max_restarts = 40
        for attempt in range(max_restarts):
            if time.time() > n_deadline - 1.0:
                break
            colors_arr = np.array([rng.randrange(4) for _ in range(num_edges)], dtype=np.int64)
            colors_arr, cnt = _local_search(colors_arr, n, E, n_deadline, rng, max_iter=4000)
            if cnt == 0:
                # verify independently
                if total_c6_count(colors_arr, n, E) == 0:
                    best_n = n
                    best_colors = colors_arr.tolist()
                    found = True
                    break
        if found:
            continue
        else:
            break
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
