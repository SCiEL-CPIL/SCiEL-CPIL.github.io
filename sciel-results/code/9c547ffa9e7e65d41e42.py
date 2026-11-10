"""Find dense triangle-free 4-chromatic graphs on 16 vertices via exact backtracking."""
import time
import random
from itertools import combinations

# Deadline guard: keep a safety margin before the 150s wall-clock limit
_DEADLINE = 130.0  # seconds
_START_TIME = None


def _is_triangle_free(adj):
    """Check if adjacency matrix represents a triangle-free graph."""
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True


def _is_3_colorable(adj):
    """Check if graph with adjacency matrix is 3-colorable via backtracking."""
    n = len(adj)
    color = [-1] * n

    def backtrack(v):
        if v == n:
            return True
        for c in range(3):
            ok = True
            for u in range(v):
                if adj[v][u] and color[u] == c:
                    ok = False
                    break
            if ok:
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = -1
        return False

    return backtrack(0)


def _greedy_build(start_edges, n=16, max_edges=120, time_limit=None):
    """
    Start from initial edges, then greedily add edges that maintain triangle-freeness
    and 4-chromaticity. Returns list of edges.
    """
    adj = [[False] * n for _ in range(n)]
    for u, v in start_edges:
        adj[u][v] = adj[v][u] = True

    # Check initial is triangle-free
    if not _is_triangle_free(adj):
        return start_edges

    # Check initial is not 3-colorable
    if _is_3_colorable(adj):
        return start_edges

    # Build list of candidate non-edges in random order
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                candidates.append((i, j))
    random.shuffle(candidates)

    current_edges = len(start_edges)

    for (u, v) in candidates:
        if time_limit is not None and time.time() - time_limit > 0:
            break
        # Check triangle-freeness: adding (u,v) creates a triangle iff there exists w
        # adjacent to both u and v
        creates_triangle = False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                creates_triangle = True
                break
        if creates_triangle:
            continue

        # Try adding edge
        adj[u][v] = adj[v][u] = True
        if not _is_3_colorable(adj):
            current_edges += 1
        else:
            adj[u][v] = adj[v][u] = False

    # Collect edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges


def _local_search(edges, n=16, time_limit=None):
    """
    Try to add more edges via local search: try adding each non-edge,
    and if it breaks 3-colorability, accept. Also try edge swaps.
    """
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True

    improved = True
    while improved:
        if time_limit is not None and time.time() - time_limit > 0:
            break
        improved = False
        # Try adding each non-edge
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    continue
                # Check triangle-freeness
                creates_triangle = False
                for w in range(n):
                    if w != i and w != j and adj[i][w] and adj[j][w]:
                        creates_triangle = True
                        break
                if creates_triangle:
                    continue
                adj[i][j] = adj[j][i] = True
                if not _is_3_colorable(adj):
                    improved = True
                else:
                    adj[i][j] = adj[j][i] = False

    # Collect edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                result.append([i, j])
    return result


def _groetzsch_edges():
    """Return edges of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges


def _random_start(n=16):
    """Generate a random triangle-free 4-chromatic graph as a starting point."""
    # Start with Grötzsch graph on 0..10, then randomly add edges among 11..15
    # and between them and 0..10
    edges = _groetzsch_edges()
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True

    # Try to add random edges involving vertices 11-15
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                candidates.append((i, j))
    random.shuffle(candidates)

    for (u, v) in candidates:
        # Check triangle-freeness
        creates_triangle = False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                creates_triangle = True
                break
        if creates_triangle:
            continue
        adj[u][v] = adj[v][u] = True
        if not _is_3_colorable(adj):
            pass  # Keep it
        else:
            adj[u][v] = adj[v][u] = False

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                result.append([i, j])
    return result


def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + _DEADLINE

    n = 16
    best_edges = _groetzsch_edges()
    best_count = len(best_edges)

    # Strategy 1: Greedy from Grötzsch core
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        # Randomize the order of vertex relabeling to get different greedy paths
        perm = list(range(n))
        random.shuffle(perm)
        # Remap edges
        remapped = []
        for u, v in best_edges:
            remapped.append([perm[u], perm[v]])
        # Ensure no self-loops (shouldn't happen with a permutation)
        result = _greedy_build(remapped, n=n)
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result

    # Strategy 2: Local search on best found
    if time.time() < deadline - 10:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            perm = list(range(n))
            random.shuffle(perm)
            remapped = []
            for u, v in best_edges:
                remapped.append([perm[u], perm[v]])
            result = _local_search(remapped, n=n, time_limit=deadline - 5)
            if len(result) > best_count:
                best_count = len(result)
                best_edges = result

    # Strategy 3: Random starts with greedy
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        start = _random_start(n=n)
        result = _greedy_build(start, n=n)
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result
        # Also try local search on the random start
        if time.time() < deadline - 5:
            result2 = _local_search(start, n=n, time_limit=deadline - 5)
            if len(result2) > best_count:
                best_count = len(result2)
                best_edges = result2

    # Final local search on the best
    if time.time() < deadline - 3:
        final = _local_search(best_edges, n=n, time_limit=deadline - 3)
        if len(final) > best_count:
            best_edges = final
            best_count = len(final)

    return best_edges
