"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit

    # Precompute edge endpoints
    eu = np.array([e[0] for e in edges], dtype=int)
    ev = np.array([e[1] for e in edges], dtype=int)

    # Classify edges: top cycle, bottom cycle, matching
    cycle1_idx = []
    cycle2_idx = []
    match_idx = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle1_idx.append(idx)
        elif u >= m and v >= m:
            cycle2_idx.append(idx)
        else:
            match_idx.append(idx)

    def fast_connected(signs):
        """Return True if the 2-lift is connected (BFS on 2n vertices)."""
        # Build adjacency of the lift: vertex (v, s) -> (v', s')
        # For edge i with sign s_i: (u, s) connects to (v, s) if s_i=+1, else (v, s^1)
        n_lift = 2 * n
        adj = [[] for _ in range(n_lift)]
        for i in range(num_edges):
            sgn = signs[i]
            u, v = eu[i], ev[i]
            if sgn == 1:
                # equal sides
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # opposite sides
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        # BFS from vertex 0
        visited = bytearray(n_lift)
        visited[0] = 1
        q = deque([0])
        cnt = 1
        while q:
            x = q.popleft()
            for y in adj[x]:
                if not visited[y]:
                    visited[y] = 1
                    cnt += 1
                    q.append(y)
        return cnt == n_lift

    best_signs = None
    best_radius = float('inf')

    def evaluate(signs):
        nonlocal best_signs, best_radius
        if not fast_connected(signs):
            return None
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
            return radius
        except Exception:
            return None

    def make_pattern(c1, c2, mt):
        signs = np.ones(num_edges, dtype=int)
        for k, idx in enumerate(cycle1_idx):
            signs[idx] = c1[k % len(c1)]
        for k, idx in enumerate(cycle2_idx):
            signs[idx] = c2[k % len(c2)]
        for k, idx in enumerate(match_idx):
            signs[idx] = mt[k % len(mt)]
        return signs

    def local_search(start_signs, max_iters=400, allow_double=False):
        nonlocal best_signs, best_radius
        cur = start_signs.copy()
        cur_radius = evaluate(cur)
        if cur_radius is None:
            return
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            improved = False
            # single flips
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                trial = cur.copy()
                trial[i] = -trial[i]
                tr = evaluate(trial)
                if tr is not None and tr < cur_radius - 1e-12:
                    cur = trial
                    cur_radius = tr
                    improved = True
                    break
            # double flips
            if not improved and allow_double:
                for i in range(num_edges):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i + 1, num_edges):
                        if time.time() > deadline - 2:
                            break
                        trial = cur.copy()
                        trial[i] = -trial[i]
                        trial[j] = -trial[j]
                        tr = evaluate(trial)
                        if tr is not None and tr < cur_radius - 1e-12:
                            cur = trial
                            cur_radius = tr
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break

    # Phase 1: structured patterns (diverse, often valid)
    base_patterns = [
        [1],
        [-1],
        [1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [1, 1, 1, -1],
        [1, -1, 1, 1],
    ]
    for pt in base_patterns:
        for pb in base_patterns:
            for pm in base_patterns:
                if time.time() > deadline - 12:
                    break
                signs = make_pattern(pt, pb, pm)
                evaluate(signs)

    # Phase 2: random sampling
    rng = np.random.default_rng(42)
    num_random = 1500 if m <= 5 else 400
    for _ in range(num_random):
        if time.time() > deadline - 12:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)

    # Phase 3: local search from best
    if best_signs is not None:
        local_search(best_signs, max_iters=300, allow_double=False)

    # Phase 4: random restarts + local search
    num_restarts = 40 if m <= 5 else 15
    for _ in range(num_restarts):
        if time.time() > deadline - 3:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
        if best_signs is not None:
            local_search(best_signs, max_iters=80, allow_double=True)

    if best_signs is None:
        return np.ones(num_edges, dtype=int)

    return best_signs
# EVOLVE-BLOCK-END
