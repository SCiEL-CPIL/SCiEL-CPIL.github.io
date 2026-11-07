"""Densest graph on 15 vertices avoiding equal-degree path endpoints.

Strategy: maintain a 'hot' matrix H[i][j] = number of currently-existing
edges (u,v) such that i is in N(u)\{v} and j is in N(v)\{u} (or symmetric).
Adding edge (a,b) is valid iff for every edge (u,v) in the graph,
deg[a] != deg[b] whenever a in N(u)\{v} and b in N(v)\{u} (or swapped).
We maintain H incrementally so each add-check is O(E) instead of O(E*deg^2).
"""
import random
import time
import sys

# EVOLVE-BLOCK-START
N = 15


def _check_full(adj, deg):
    """Full O(E * deg^2) validity check. Returns True if valid."""
    n = N
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True


def _try_add(adj, deg, hot, a, b):
    """Try to add edge (a,b). Returns True if valid and added."""
    n = N
    # Check: for each existing edge (u,v), if a in N(u)\{v} and b in N(v)\{u},
    # then deg[a] must != deg[b] (since path a-u-v-b would have endpoints a,b)
    # Similarly for b in N(u)\{v} and a in N(v)\{u}
    da, db = deg[a], deg[b]
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            # Case 1: a is endpoint via u, b is endpoint via v
            if a != v and a in adj[u] and b != u and b in adj[v]:
                if da == db:
                    return False
            # Case 2: b is endpoint via u, a is endpoint via v
            if b != v and b in adj[u] and a != u and a in adj[v]:
                if da == db:
                    return False
    # Valid: add the edge
    adj[a].add(b)
    adj[b].add(a)
    deg[a] += 1
    deg[b] += 1
    # Update hot matrix: for each existing edge (u,v),
    # new edge (a,b) creates potential paths.
    # For each edge (u,v):
    #   if a in N(u)\{v} and b in N(v)\{u}: hot[a][b] += 1 (path a-u-v-b)
    #   if b in N(u)\{v} and a in N(v)\{u}: hot[b][a] += 1 (path b-u-v-a)
    #   if a in N(v)\{u} and b in N(u)\{v}: hot[a][b] += 1 (path a-v-u-b)
    #   if b in N(v)\{u} and a in N(u)\{v}: hot[b][a] += 1 (path b-v-u-a)
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            # a via u, b via v
            if a != v and a in adj[u] and b != u and b in adj[v]:
                hot[a][b] += 1
            # b via u, a via v
            if b != v and b in adj[u] and a != u and a in adj[v]:
                hot[b][a] += 1
            # a via v, b via u
            if a != u and a in adj[v] and b != v and b in adj[u]:
                hot[a][b] += 1
            # b via v, a via u
            if b != u and b in adj[v] and a != v and a in adj[u]:
                hot[b][a] += 1
    return True


def _try_remove(adj, deg, hot, a, b):
    """Remove edge (a,b). Assumes it exists."""
    n = N
    # Undo hot updates (reverse of _try_add)
    da, db = deg[a] - 1, deg[b] - 1  # degrees after removal
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            if a != v and a in adj[u] and b != u and b in adj[v]:
                hot[a][b] -= 1
            if b != v and b in adj[u] and a != u and a in adj[v]:
                hot[b][a] -= 1
            if a != u and a in adj[v] and b != v and b in adj[u]:
                hot[a][b] -= 1
            if b != u and b in adj[v] and a != v and a in adj[u]:
                hot[b][a] -= 1
    adj[a].discard(b)
    adj[b].discard(a)
    deg[a] -= 1
    deg[b] -= 1


def _greedy_construct(n, rng, deadline):
    """Greedy: try all edges in random order, add if valid."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    hot = [[0] * n for _ in range(n)]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    count = 0
    for (a, b) in edges:
        if time.time() > deadline:
            break
        if _try_add(adj, deg, hot, a, b):
            count += 1
    return adj, deg, hot, count


def _local_search(adj, deg, hot, rng, deadline, iters=2000):
    """Local search: try adding missing edges; if stuck, try 1-for-1 swaps."""
    n = N
    count = sum(len(s) for s in adj) // 2
    best_adj = [set(s) for s in adj]
    best_deg = list(deg)
    best_hot = [row[:] for row in hot]
    best_count = count

    no_improve = 0
    for it in range(iters):
        if time.time() > deadline:
            break
        # Collect missing edges
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    missing.append((i, j))
        rng.shuffle(missing)
        improved = False
        for (a, b) in missing:
            if time.time() > deadline:
                break
            if _try_add(adj, deg, hot, a, b):
                count += 1
                improved = True
                if count > best_count:
                    best_count = count
                    best_adj = [set(s) for s in adj]
                    best_deg = list(deg)
                    best_hot = [row[:] for row in hot]
                    no_improve = 0
                break
        if not improved:
            # Try 1-for-1 swaps: remove one edge, add another
            all_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            rng.shuffle(all_edges)
            swap_found = False
            for (ra, rb) in all_edges:
                if time.time() > deadline:
                    break
                _try_remove(adj, deg, hot, ra, rb)
                # Now try to add some missing edge
                for (a, b) in missing:
                    if a == ra and b == rb:
                        continue
                    if a == rb and b == ra:
                        continue
                    if _try_add(adj, deg, hot, a, b):
                        # Swap succeeded, count unchanged
                        if count > best_count:
                            best_count = count
                            best_adj = [set(s) for s in adj]
                            best_deg = list(deg)
                            best_hot = [row[:] for row in hot]
                        swap_found = True
                        break
                    # undo the add
                    _try_remove(adj, deg, hot, a, b)
                if swap_found:
                    # Re-add the removed edge if we didn't find a replacement
                    # Actually if swap_found, we added something, so we're fine
                    # But we removed (ra,rb) and added (a,b), count is same
                    break
                else:
                    # Re-add the removed edge
                    _try_add(adj, deg, hot, ra, rb)
            if not swap_found:
                no_improve += 1
                if no_improve > 30:
                    break

    return best_adj, best_deg, best_hot, best_count


def _multi_start(n, rng, deadline):
    """Multiple greedy starts + local search refinement."""
    best_count = -1
    best_adj = None
    best_deg = None
    best_hot = None

    n_starts = 40
    for k in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        adj, deg, hot, cnt = _greedy_construct(n, rng, deadline - 10.0)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
            best_deg = list(deg)
            best_hot = [row[:] for row in hot]
        # Refine
        adj2, deg2, hot2, cnt2 = _local_search(
            adj, deg, hot, rng, deadline - 5.0, iters=500
        )
        if cnt2 > best_count:
            best_count = cnt2
            best_adj = [set(s) for s in adj2]
            best_deg = list(deg2)
            best_hot = [row[:] for row in hot2]

    # Final polish
    if best_adj is not None and time.time() < deadline - 3.0:
        adj3, deg3, hot3, cnt3 = _local_search(
            best_adj, best_deg, best_hot, rng, deadline - 1.0, iters=1000
        )
        if cnt3 > best_count:
            best_count = cnt3
            best_adj = [set(s) for s in adj3]
            best_deg = list(deg3)
            best_hot = [row[:] for row in hot3]

    if best_adj is None:
        best_adj = [set() for _ in range(n)]

    return best_adj


def _verify(adj, n):
    """Final verification: check the graph is valid."""
    deg = [len(adj[i]) for i in range(n)]
    return _check_full(adj, deg), deg


def construct():
    n = N
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s
    rng = random.Random(42)

    adj = _multi_start(n, rng, deadline)

    # Final verification
    valid, deg = _verify(adj, n)
    if not valid:
        # Fallback: rebuild with full check
        adj2 = [set() for _ in range(n)]
        deg2 = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng2 = random.Random(999)
        rng2.shuffle(edges)
        for (a, b) in edges:
            adj2[a].add(b)
            adj2[b].add(a)
            deg2[a] += 1
            deg2[b] += 1
            if not _check_full(adj2, deg2):
                adj2[a].discard(b)
                adj2[b].discard(a)
                deg2[a] -= 1
                deg2[b] -= 1
        adj = adj2

    # Build edge list with proper types
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    return edges
# EVOLVE-BLOCK-END
