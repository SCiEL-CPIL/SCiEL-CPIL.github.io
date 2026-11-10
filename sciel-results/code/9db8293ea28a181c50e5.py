"""Few cycle lengths under degree and girth constraints."""
import time
import random
from itertools import combinations, permutations

# EVOLVE-BLOCK-START

_N = 12
_DEADLINE = 85.0  # safety margin before 90s kill

def _cycle_lengths(adj):
    """Return set of lengths of all simple cycles in undirected graph."""
    lengths = set()
    n = len(adj)
    for start in range(n):
        for end in range(start + 1, n):
            if end in adj[start]:
                continue
            # DFS paths from start to end of length >= 2 (edges)
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                for nb in adj[node]:
                    if nb == start:
                        if len(path) >= 2:
                            lengths.add(len(path) + 1)
                    elif nb == end and len(path) >= 2:
                        lengths.add(len(path) + 1)
                    elif nb not in visited and nb > start:
                        # Only allow nodes > start (except endpoints) to avoid dup cycles
                        stack.append((nb, path + [nb], visited | {nb}))
    return lengths

def _valid(edges):
    """Check all constraints."""
    n = _N
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n or u < 0 or v < 0:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d < 3:
            return False
    # triangle-free
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
    # girth >= 4 (no 3-cycles already checked; need no shorter, but min is 3)
    # must contain a cycle
    # check connectivity of at least one cycle: just check cycle exists
    # BFS for a cycle
    visited = [False] * n
    for s in range(n):
        if not visited[s]:
            stack = [(s, -1, 0)]
            visited[s] = True
            while stack:
                node, par, dist = stack.pop()
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append((nb, node, dist + 1))
                    elif nb != par:
                        # found a cycle
                        return True
    return False

def _fallback():
    """The seed graph: two 6-cycles with matching edges (cubical-like)."""
    es = set()
    for i in range(6):
        for u, v in [(i, (i + 1) % 6), (i + 6, (i + 1) % 6 + 6), (i, i + 6)]:
            es.add(tuple(sorted((u, v))))
    return [list(e) for e in sorted(es)]

def _build_bipartite(perm):
    """Build a 3-regular bipartite graph from a permutation of 6 (complement of matching in K6,6)."""
    edges = []
    for i in range(6):
        for j in range(6):
            if i != perm[j]:
                edges.append([i, j + 6])
    return edges

def _evaluate(edges):
    if not _valid(edges):
        return None
    adj = [set() for _ in range(_N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    lengths = _cycle_lengths(adj)
    return len(lengths)

def construct():
    t0 = time.time()
    best_edges = None
    best_score = float('inf')

    # Try bipartite 3-regular graphs: complement of perfect matching in K6,6
    # Each is determined by a permutation of 6 (360 total)
    from itertools import permutations as _perms
    for perm in _perms(range(6)):
        if time.time() > _DEADLINE:
            break
        edges = _build_bipartite(perm)
        score = _evaluate(edges)
        if score is not None and score < best_score:
            best_score = score
            best_edges = edges
        if best_score == 1:
            break

    if best_edges is not None:
        # convert to proper types
        result = []
        for u, v in best_edges:
            result.append([int(u), int(v)])
        return result

    return _fallback()

# EVOLVE-BLOCK-END
