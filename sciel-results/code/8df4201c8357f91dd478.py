"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin (evaluator kills at 150s)

    # Precompute all 60 distinct 6-cycles on 6 labeled vertices
    _cycles_cache = None
    def _get_cycles():
        nonlocal _cycles_cache
        if _cycles_cache is None:
            cycles = []
            seen = set()
            for perm in permutations(range(1, 6)):
                cycle = (0,) + perm
                edges = frozenset(frozenset((cycle[i], cycle[(i+1) % 6])) for i in range(6))
                if edges not in seen:
                    seen.add(edges)
                    cycles.append(cycle)
            _cycles_cache = cycles
        return _cycles_cache

    def _edge_index(i, j, n):
        """Compute index for edge (i,j) in lexicographic order."""
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def _build_adj_structures(colors, n):
        adj = [[-1] * n for _ in range(n)]
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
                idx += 1
        return adj, adj_list

    def _find_c6(adj, adj_list, n):
        cycles = _get_cycles()
        for c in range(4):
            for subset in combinations(range(n), 6):
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return c, subset, cycle
        return None

    def _verify_full(colors, n):
        if n < 6:
            return True
        adj, adj_list = _build_adj_structures(colors, n)
        return _find_c6(adj, adj_list, n) is None

    def _local_search_fix(colors, n, deadline, max_iters=10000):
        adj, adj_list = _build_adj_structures(colors, n)
        for it in range(max_iters):
            if time.time() > deadline:
                break
            result = _find_c6(adj, adj_list, n)
            if result is None:
                return colors, True
            c, subset, cycle = result
            # Try flipping each edge of the violating C6 to a different color
            fixed = False
            for k in range(6):
                u = subset[cycle[k]]
                w = subset[cycle[(k + 1) % 6]]
                if u > w:
                    u, w = w, u
                old_c = adj[u][w]
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    adj_list[old_c][u].discard(w)
                    adj_list[old_c][w].discard(u)
                    adj[u][w] = new_c
                    adj[w][u] = new_c
                    adj_list[new_c][u].add(w)
                    adj_list[new_c][w].add(u)
                    idx = _edge_index(u, w, n)
                    colors[idx] = new_c
                    fixed = True
                    break
                if fixed:
                    break
            if not fixed:
                # Random flip on the violating cycle
                k = np.random.randint(6)
                u = subset[cycle[k]]
                w = subset[cycle[(k + 1) % 6]]
                if u > w:
                    u, w = w, u
                old_c = adj[u][w]
                new_c = np.random.randint(0, 4)
                while new_c == old_c:
                    new_c = np.random.randint(0, 4)
                adj_list[old_c][u].discard(w)
                adj_list[old_c][w].discard(u)
                adj[u][w] = new_c
                adj[w][u] = new_c
                adj_list[new_c][u].add(w)
                adj_list[new_c][w].add(u)
                idx = _edge_index(u, w, n)
                colors[idx] = new_c
        return colors, _verify_full(colors, n)

    def _constrained_backtrack(n_target, deadline):
        """
        Backtracking with strong incremental C6 detection.
        bad[c][i] = set of j < i such that assigning color c to (i, v) would
        create a C6 (because there is a color-c path of length 4 from i to j
        in the current subgraph, and (j, v) is already color c).
        When we assign (i, v) = c, we also update bad for all h < i using the
        new color-c edges from v.
        """
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        # bad[c][i] = set of j < i that would form C6 with (i, v) colored c
        bad = [[set() for _ in range(n_target)] for _ in range(4)]

        def _update_bad_for_v(v, c, i):
            """After setting (i, v) = c, update bad for all h < i."""
            # v is now a color-c neighbor of i.
            # For each h < i, if there is a color-c path of length 3 from h to i
            # (h-a-b-i), then (h, v)=c would create h-v-i-b-a-h? Let's verify.
            # Path h-a-b-i (3 edges) + edge i-v + edge v-h = cycle h-a-b-i-v-h (5 edges) = C5.
            # That's C5, not C6. We need path of length 4: h-a-b-e-i (4 edges) + i-v + v-h = C6.
            # So we need a color-c path of length 4 from h to i.
            # Also: if (h, i) is color c AND there is a color-c path of length 3 from h to i
            # through other vertices, then h-v + v-i + path = C5. Not C6.
            # Let me reconsider: C6 through v and i.
            # v - i - x - y - z - w - v, all color c.
            # So we need a color-c path of length 4 from i to w (i-x-y-z-w), and (w, v) = c.
            # So bad[c][i] should contain w if (w, v)=c and there's a color-c path of length 4 from i to w.
            # When we add edge (i, v)=c, for each h < i:
            #   We need to know if there's a color-c path of length 4 from h to i.
            #   If yes, and (h, v)=c, then h-v-i-...-h is a C6? 
            #   h-v (edge) + v-i (new edge) + i-...-h (path length 4) = h-v-i-a-b-c-h? That's 6 edges only if path is length 4: h-v-i-a-b-c-h is 6 edges = C6. Yes!
            #   Wait: h-v (1), v-i (2), i-a (3), a-b (4), b-c (5), c-h (6). Yes, C6.
            #   But we need the path i-a-b-c-h to be length 4 (4 edges: i-a, a-b, b-c, c-h) and all color c, and h != v, and the path must be simple (no repeated vertices except h and i endpoints... actually h and i are endpoints of the path, v is separate).
            #   So: for each h < i, if (h, v) == c AND there exists a simple color-c path of length 4 from i to h in the current graph (before adding (i,v)), then (h, v)=c creates a C6, so h should be in bad[c][i].
            #   But wait, bad[c][i] is used to check if assigning (i, v)=c is valid. The check is: for all d in adj_list[c][v] with d < i, if d is in bad[c][i], then invalid.
            #   Actually, the check should be: assigning (i, v)=c creates a C6 iff there exists d < i with (d, v)=c and a color-c path of length 4 from i to d (in the graph without (i,v)).
            #   So bad[c][i] = {d < i : (d, v)=c AND path4_c(i, d) exists}.
            #   But (d, v)=c is already known from adj_list. So bad[c][i] = {d < i : d in adj_list[c][v] AND path4_c(i, d)}.
            #   We compute path4_c(i, d) for all d < i.
            pass

        def _path4_exists(c, src, dst, v, adj_list, adj):
            """Check if there is a simple color-c path of length 4 from src to dst,
            using only vertices < v (i.e., not using v)."""
            # Path: src - a - b - e - dst, all color c, all distinct, all < v.
            for a in adj_list[c][src]:
                if a >= v or a == dst:
                    continue
                for b in adj_list[c][a]:
                    if b >= v or b == src or b == dst or b == a:
                        continue
                    for e in adj_list[c][b]:
                        if e >= v or e == src or e == dst or e == a or e == b:
                            continue
                        if e == dst:
                            return True
                        if dst in adj_list[c][e]:
                            # e - dst is an edge, but we need e to BE dst for length 4.
                            # Path src-a-b-e-dst is length 4 if dst == e.
                            # If dst is neighbor of e but dst != e, that's length 5.
                            pass
            return False

        def _search(v):
            if time.time() > deadline:
                return False
            if v == n_target:
                return True
            if v == 0:
                return _search(v + 1)

            # Precompute path4 from each i < v to each d < v (for use in bad sets)
            # path4[i][d] = True if color-c path of length 4 from i to d exists (for each c)
            # But we need it per color. Let's compute path4_c[i][d] for all c, i, d < v.
            path4 = [[[False] * v for _ in range(v)] for _ in range(4)]
            for c in range(4):
                for i in range(v):
                    if not adj_list[c][i]:
                        continue
                    for d in range(v):
                        if i == d:
                            continue
                        found = False
                        for a in adj_list[c][i]:
                            if a == d or a >= v:
                                continue
                            for b in adj_list[c][a]:
                                if b == i or b == d or b == a or b >= v:
                                    continue
                                for e in adj_list[c][b]:
                                    if e == i or e == d or e == a or e == b or e >= v:
                                        continue
                                    if e == d:
                                        found = True
                                        break
                                    # e is not d, but is d a neighbor of e? No, we need path length exactly 4.
                                if found:
                                    break
                            if found:
                                break
                        path4[c][i][d] = found

            # Now build bad[c][i] for all c, i < v
            for c in range(4):
                for i in range(v):
                    bad[c][i].clear()
                    for d in range(i):
                        if d in adj_list[c][v] and path4[c][i][d]:
                            bad[c][i].add(d)

            def _assign(idx):
                if time.time() > deadline:
                    return False
                if idx == v:
                    return _search(v + 1)
                i = idx
                for c in range(4):
                    # Check if (i, v) = c is valid
                    valid = True
                    for d in range(i):
                        if d in adj_list[c][v] and d in bad[c][i]:
                            valid = False
                            break
                    if not valid:
                        continue

                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)

                    # Update bad for all h < i: adding (i, v)=c may create new C6 threats.
                    # For each h < i, if (h, v)=c and there's now a color-c path of length 4 from h to i,
                    # then h should be added to bad[c][i]... wait, bad[c][h] is for (h, v).
                    # Actually, we need to update bad[c][h] for h < i: if (h, v)=c and there's a
                    # color-c path of length 4 from h to i (now that (i,v) is color c), then...
                    # No, bad[c][h] is used when we later check (h, v). But (h, v) is already assigned.
                    # The bad set is only relevant for the current vertex v. Since we're assigning
                    # edges to v in order i=0,1,...,v-1, and we've already fixed (h,v) for h < i,
                    # we don't need to re-check them. But we DO need to ensure that future assignments
                    # (k, v) for k > i don't create C6s involving the new edge (i, v).
                    # A C6 involving (k, v) and (i, v) would need a color-c path of length 4 from k to i.
                    # So we need to update bad[c][k] for k > i? No, bad[c][k] is computed when we reach
                    # vertex v in _search, before assigning any edges. But we're in the middle of
                    # assigning edges to v. The bad sets were precomputed assuming no edges to v.
                    # This is a problem: the bad sets are stale after we assign (i, v).
                    # 
                    # Fix: Instead of precomputing bad once, we should check validity incrementally.
                    # When assigning (i, v) = c, we need to check if it creates a C6 with any
                    # already-assigned (d, v) = c for d < i. A C6 would be: d - v - i - a - b - e - d
                    # (all color c). So we need a color-c path of length 4 from i to d in the current
                    # graph (which includes all edges among vertices < v, plus edges (h, v) for h < i).
                    # But the path from i to d cannot use v (since v is the new vertex and the path
                    # must be in the subgraph induced by vertices < v, plus possibly edges to v? No,
                    # the C6 is d-v-i-a-b-e-d, so the path i-a-b-e-d is in the subgraph on vertices
                    # {i, a, b, e, d} which are all < v. So the path does NOT use v.
                    # Therefore, the path4 check only depends on edges among vertices < v, which are
                    # fixed. So the precomputed path4 is still valid!
                    # The only thing that changes is which (d, v) edges are color c, but we check
                    # that directly via adj_list[c][v].
                    # So the validity check is correct: for d < i, if (d, v)=c and path4_c(i, d), invalid.
                    # And we don't need to update bad sets because path4 doesn't change.

                    if _assign(idx + 1):
                        return True

                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)
                return False

            return _assign(0)

        if _search(0):
            colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    colors.append(adj[i][j])
            return colors
        return None

    # Start with small n and try to grow
    best_n = 5
    best_colors = [0] * (best_n * (best_n - 1) // 2)

    # Try to construct valid colorings for increasing n
    for target_n in range(6, 21):
        if time.time() > deadline - 10:
            break

        success = False

        # Try constrained backtracking first
        if target_n <= 17:
            result = _constrained_backtrack(target_n, deadline - 20)
            if result is not None:
                if _verify_full(result, target_n):
                    best_n = target_n
                    best_colors = result
                    success = True

        if not success:
            # Try random + local search
            m = target_n * (target_n - 1) // 2
            max_restarts = 30 if target_n <= 12 else 15
            for restart in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                if restart == 0:
                    colors = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            c = (i * 3 + j * 5) % 4
                            colors.append(c)
                else:
                    colors = np.random.randint(0, 4, size=m).tolist()
                colors, valid = _local_search_fix(colors, target_n, deadline - 5, max_iters=5000)
                if valid:
                    best_n = target_n
                    best_colors = colors
                    success = True
                    break

        if not success:
            break

    # Final safety: ensure we have a valid result
    if best_n >= 6 and not _verify_full(best_colors, best_n):
        best_n = 5
        best_colors = [0] * (best_n * (best_n - 1) // 2)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
