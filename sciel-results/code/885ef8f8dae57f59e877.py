"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    # Evaluator kills at 90s wall-clock; keep a safety margin.
    DEADLINE = 84.0

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m  # base graph has 2m vertices

    # Precompute adjacency lists for fast connectivity checks on the lift.
    # For each base edge index, store (u, v).
    uv = [(int(u), int(v)) for (u, v) in edges]

    # Build base adjacency structure (edge index per vertex) for lift BFS.
    # base_adj[v] = list of (neighbor_vertex, edge_index)
    base_adj = [[] for _ in range(n_vertices)]
    for idx, (u, v) in enumerate(uv):
        base_adj[u].append((v, idx))
        base_adj[v].append((u, idx))

    def spectral_radius(signs):
        """Spectral radius of the signed base adjacency matrix (2m x 2m)."""
        A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for idx, (u, v) in enumerate(uv):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def is_connected(signs):
        """Check the two-lift is connected via BFS on 2m lift vertices.
        Lift vertex encoding: (v, side) -> v*2 + side.
        Edge with sign s: if s>0 joins equal sides, if s<0 joins opposite sides.
        """
        n_lift = n_vertices * 2
        seen = np.zeros(n_lift, dtype=bool)
        dq = deque()
        start = 0
        seen[start] = True
        dq.append(start)
        count = 1
        while dq:
            node = dq.popleft()
            v = node // 2
            side = node % 2
            for (nb, idx) in base_adj[v]:
                s = signs[idx]
                if s > 0:
                    nside = side
                else:
                    nside = 1 - side
                nn = nb * 2 + nside
                if not seen[nn]:
                    seen[nn] = True
                    count += 1
                    dq.append(nn)
                    if count == n_lift:
                        return True
        return count == n_lift

    def evaluate(signs):
        """Return spectral radius if the lift is valid (connected), else inf."""
        if not is_connected(signs):
            return float('inf')
        return spectral_radius(signs)

    # 3-regular check: prism graph is 3-regular by construction, so the lift is
    # automatically 3-regular. Only connectivity is the non-trivial constraint.

    # ---------- Search ----------
    best_signs = None
    best_radius = float('inf')

    def try_candidate(signs):
        nonlocal best_signs, best_radius
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    def local_search(signs):
        """Steepest-descent local search: repeatedly flip the edge giving the
        best valid improvement until no improvement found or deadline."""
        nonlocal best_signs, best_radius
        cur = signs.copy()
        cur_r = evaluate(cur)
        if not np.isfinite(cur_r):
            return
        if cur_r < best_radius:
            best_radius = cur_r
            best_signs = cur.copy()

        # Cache current signs as int array for speed.
        s = cur.astype(np.int64)
        cur_r = best_radius
        while True:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            best_i = -1
            best_new_r = cur_r
            best_new_s = None
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break
                # flip
                s[i] = -s[i]
                r = evaluate(s)
                s[i] = -s[i]  # revert
                if r < best_new_r - 1e-12:
                    best_new_r = r
                    best_i = i
                    best_new_s = s.copy()
            if best_i < 0:
                break
            s = best_new_s
            cur_r = best_new_r
            if cur_r < best_radius:
                best_radius = cur_r
                best_signs = s.copy()
            improved = True

    # Multi-restart random search with local search.
    rng = np.random.default_rng(7)

    # First, try to find any valid connected lift quickly with a heuristic:
    # all +1 signs give a connected lift? Let's test.
    s0 = np.ones(n_edges, dtype=np.int64)
    try_candidate(s0)

    # Now multi-restart.
    n_restarts = 400
    for ridx in range(n_restarts):
        if time.time() - start_time > DEADLINE:
            break
        signs = rng.choice([-1, 1], size=n_edges).astype(np.int64)
        # Only proceed to local search if connected (cheap check).
        if is_connected(signs):
            local_search(signs)

    # If we still have no valid lift (shouldn't happen since all-+1 is connected),
    # fall back to all +1.
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=np.int64)

    # Final high-precision polish on the best solution.
    if time.time() - start_time < DEADLINE - 1.0:
        local_search(best_signs.astype(np.int64))

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
