import time
import random
import sys
import numpy as np
from itertools import combinations

def construct():
    n = 16
    target = 65
    deadline = time.time() + 112.0  # safety margin under 150s wall-clock
    rng = random.Random(42)

    # Precompute all possible triangles (triples)
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    num_tri = len(triangles)

    # Precompute triangle edge sets for fast lookup
    tri_edges = {}
    for (u, v, w) in triangles:
        tri_edges[(u, v, w)] = frozenset([(min(u,v),max(u,v)), (min(v,w),max(v,w)), (min(u,w),max(u,w))])

    def _score_from_adj(adj, deg):
        best = 0
        for (u, v, w) in triangles:
            if adj[u][v] and adj[v][w] and adj[u][w]:
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
        return best

    def _score_from_edges(edges):
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        return _score_from_adj(adj, deg)

    def _valid(edges, n, target):
        if len(edges) != target:
            return False
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or v < 0 or u >= n or v >= n:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                return False
            seen.add((a, b))
        return True

    def _seed_edges(n):
        # K_{8,8} + one intra edge -> 65 edges, score 24
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return edges

    best_edges = _seed_edges(n)
    best_score = _score_from_edges(best_edges)

    def try_improve(edges, score):
        nonlocal best_edges, best_score
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            return True
        return False

    def _edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def _compute_triangles_adj(adj):
        """Return set of triangles present in adj."""
        tris = set()
        for (u, v, w) in triangles:
            if adj[u][v] and adj[v][w] and adj[u][w]:
                tris.add((u, v, w))
        return tris

    # --- Enhanced local search with 1-edge and 2-edge swaps ---
    def local_search(edges, max_iter=200):
        nonlocal best_score, best_edges
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1

        score = _score_from_adj(adj, deg)
        try_improve(edges, score)

        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1

            # Find the worst triangle
            worst_tri = None
            worst_s = 0
            for (u, v, w) in triangles:
                if adj[u][v] and adj[v][w] and adj[u][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > worst_s:
                        worst_s = s
                        worst_tri = (u, v, w)

            if worst_tri is None:
                break

            u, v, w = worst_tri
            found_move = False

            # Try 1-edge swaps: remove one edge from worst triangle, add another
            for a, b in [(u, v), (v, w), (u, w)]:
                if not adj[a][b]:
                    continue
                best_candidate = None
                best_ns = score
                for c in range(n):
                    if c == a or c == b or adj[a][c]:
                        continue
                    for d in range(n):
                        if d == c or d == a or d == b or adj[c][d]:
                            continue
                        new_deg = deg[:]
                        new_deg[a] -= 1
                        new_deg[b] -= 1
                        new_deg[c] += 1
                        new_deg[d] += 1

                        new_adj = [row[:] for row in adj]
                        new_adj[a][b] = new_adj[b][a] = False
                        new_adj[c][d] = new_adj[d][c] = True
                        ns = _score_from_adj(new_adj, new_deg)
                        if ns < best_ns:
                            best_ns = ns
                            best_candidate = (c, d, a, b, new_adj, new_deg)

                if best_candidate:
                    c, d, a2, b2, new_adj, new_deg = best_candidate
                    adj = new_adj
                    deg = new_deg
                    score = best_ns
                    improved = True
                    found_move = True
                    try_improve(_edges_from_adj(adj), score)
                    break

            # If no 1-edge swap worked, try 2-edge swaps
            if not found_move:
                # Remove two edges, add two edges
                edge_pairs = list(combinations([(u, v), (v, w), (u, w)], 2))
                for (e1, e2) in edge_pairs:
                    a1, b1 = e1
                    a2, b2 = e2
                    if not adj[a1][b1] or not adj[a2][b2]:
                        continue
                    best_candidate = None
                    best_ns = score
                    # Try all pairs of new edges
                    for c1 in range(n):
                        if c1 in (a1, b1, a2, b2) or adj[a1][c1] or adj[a2][c1]:
                            continue
                        for d1 in range(n):
                            if d1 == c1 or d1 in (a1, b1, a2, b2) or adj[c1][d1]:
                                continue
                            for c2 in range(n):
                                if c2 in (a1, b1, a2, b2, c1, d1) or adj[a1][c2] or adj[a2][c2] or adj[c1][c2] or adj[d1][c2]:
                                    continue
                                for d2 in range(n):
                                    if d2 == c2 or d2 in (a1, b1, a2, b2, c1, d1, c2) or adj[c1][d2] or adj[d1][d2] or adj[c2][d2]:
                                        continue
                                    new_deg = deg[:]
                                    new_deg[a1] -= 1
                                    new_deg[b1] -= 1
                                    new_deg[a2] -= 1
                                    new_deg[b2] -= 1
                                    new_deg[c1] += 1
                                    new_deg[d1] += 1
                                    new_deg[c2] += 1
                                    new_deg[d2] += 1

                                    new_adj = [row[:] for row in adj]
                                    new_adj[a1][b1] = new_adj[b1][a1] = False
                                    new_adj[a2][b2] = new_adj[b2][a2] = False
                                    new_adj[c1][d1] = new_adj[d1][c1] = True
                                    new_adj[c2][d2] = new_adj[d2][c2] = True
                                    ns = _score_from_adj(new_adj, new_deg)
                                    if ns < best_ns:
                                        best_ns = ns
                                        best_candidate = (c1, d1, c2, d2, new_adj, new_deg)

                    if best_candidate:
                        c1, d1, c2, d2, new_adj, new_deg = best_candidate
                        adj = new_adj
                        deg = new_deg
                        score = best_ns
                        improved = True
                        found_move = True
                        try_improve(_edges_from_adj(adj), score)
                        break

            # If still no improvement, try random restarts within this local search
            if not found_move and iter_count % 10 == 0:
                # Perturb: flip a random edge
                all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                rng.shuffle(all_pairs)
                for (c, d) in all_pairs:
                    if adj[c][d]:
                        adj[c][d] = adj[d][c] = False
                        deg[c] -= 1
                        deg[d] -= 1
                    else:
                        adj[c][d] = adj[d][c] = True
                        deg[c] += 1
                        deg[d] += 1
                    break
                score = _score_from_adj(adj, deg)
                improved = True

        return edges

    # --- Final polish: aggressive hill-climbing on best solution ---
    def polish(edges):
        nonlocal best_score, best_edges
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1

        score = _score_from_adj(adj, deg)
        try_improve(edges, score)

        # Repeatedly try all possible single-edge swaps
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for (u, v) in combinations(range(n), 2):
                if not adj[u][v]:
                    continue
                for (c, d) in combinations(range(n), 2):
                    if c == u or c == v or d == u or d == v or adj[c][d]:
                        continue
                    new_deg = deg[:]
                    new_deg[u] -= 1
                    new_deg[v] -= 1
                    new_deg[c] += 1
                    new_deg[d] += 1

                    new_adj = [row[:] for row in adj]
                    new_adj[u][v] = new_adj[v][u] = False
                    new_adj[c][d] = new_adj[d][c] = True
                    ns = _score_from_adj(new_adj, new_deg)
                    if ns < score:
                        adj = new_adj
                        deg = new_deg
                        score = ns
                        improved = True
                        try_improve(_edges_from_adj(adj), score)
                        break
                if improved:
                    break

        return edges

    # --- Erdős–Gallai check ---
    def is_graphical(deg_seq):
        d = sorted(deg_seq, reverse=True)
        if sum(d) != 2 * target:
            return False
        if any(x < 0 or x > n - 1 for x in d):
            return False
        for k in range(1, n + 1):
            lhs = sum(d[:k])
            rhs = k * (k - 1) + sum(min(d[i], k) for i in range(k, n))
            if lhs > rhs:
                return False
        return True

    # --- Random Havel–Hakimi ---
    def realize_degree_sequence(deg_seq, rng):
        """Randomly realize a degree sequence via Havel–Hakimi."""
        remaining = list(enumerate(deg_seq))
        remaining.sort(key=lambda x: -x[1])

        adj = [[False] * n for _ in range(n)]

        def hh(remaining):
            if not remaining:
                return True
            if all(d == 0 for _, d in remaining):
                return True
            max_idx = max(range(len(remaining)), key=lambda i: remaining[i][1])
            v, dv = remaining.pop(max_idx)
            if dv < 0 or dv > len(remaining):
                return False
            if dv == 0:
                return hh(remaining)

            candidates = list(range(len(remaining)))
            candidates.sort(key=lambda i: -remaining[i][1])

            if dv <= len(candidates):
                chosen = candidates[:dv]
            else:
                return False

            rng.shuffle(chosen)

            for idx in chosen:
                u, du = remaining[idx]
                if du <= 0:
                    return False
                adj[v][u] = adj[u][v] = True
                remaining[idx] = (u, du - 1)

            return hh(remaining)

        rem_copy = list(remaining)
        if not hh(rem_copy):
            return None
        return adj

    # --- Main search loop ---
    # First, try the seed with enhanced local search
    local_search(list(best_edges), max_iter=300)

    # Generate diverse degree sequences to try
    deg_patterns = []

    # 14 of 8, 2 of 9
    base = [8] * 16
    for i in range(2):
        base[i] = 9
    deg_patterns.append(base[:])

    # 15 of 8, 1 of 10
    deg_patterns.append([8] * 15 + [10])

    # 2 of 7, 12 of 8, 2 of 10
    deg_patterns.append([7] * 2 + [8] * 12 + [10] * 2)

    # 1 of 7, 14 of 8, 1 of 11
    deg_patterns.append([7] + [8] * 14 + [11])

    # Add random patterns
    for _ in range(15):
        deg_seq = [8] * 16
        remaining = 130 - sum(deg_seq)
        while remaining != 0:
            i = rng.randint(0, 15)
            if remaining > 0 and deg_seq[i] < 15:
                deg_seq[i] += 1
                remaining -= 1
            elif remaining < 0 and deg_seq[i] > 0:
                deg_seq[i] -= 1
                remaining += 1
            elif remaining > 0:
                break
            elif remaining < 0:
                break
        if sum(deg_seq) == 130 and is_graphical(deg_seq):
            deg_patterns.append(deg_seq)

    # Remove duplicates
    unique_patterns = []
    seen = set()
    for p in deg_patterns:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)

    # Try each pattern
    for deg_seq in unique_patterns:
        if time.time() >= deadline - 10:
            break
        if not is_graphical(deg_seq):
            continue

        for attempt in range(2):
            if time.time() >= deadline - 10:
                break
            adj = realize_degree_sequence(deg_seq, rng)
            if adj is None:
                continue
            edges = _edges_from_adj(adj)
            if len(edges) != target:
                continue
            local_search(edges, max_iter=200)

    # Random restarts with random edge sets
    while time.time() < deadline - 15:
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        edges = [list(p) for p in all_pairs[:target]]
        if not _valid(edges, n, target):
            continue
        local_search(edges, max_iter=150)

    # Final polish on best solution
    if time.time() < deadline - 5:
        polish(best_edges)

    if not _valid(best_edges, n, target):
        best_edges = _seed_edges(n)
        best_score = _score_from_edges(best_edges)

    return {'n': n, 'edges': best_edges}
