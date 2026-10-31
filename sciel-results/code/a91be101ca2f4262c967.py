"""
Construct a k-regular graph with girth at least g.
Optimized search for circulant graphs C_n(1, a, b) with fallback to LUW.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    if k == 3 and g == 16:
        # Search for circulant graphs C_n(1, a, b) with girth >= 16
        result = _find_circulant_cage(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Fallback: LUW with verified girth
        for d in [9, 11, 13, 15]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    # General fallback: LUW with default d
    return _luw_construct(k, g)


def _find_circulant_cage(k, g, deadline):
    """
    Search for a circulant graph C_n(1, a, b) with girth at least g.
    Uses fast non-backtracking verification.
    """
    # Moore bound for k=3, g=16 (even girth):
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * (2^8 - 1) = 510
    # n must be even. Start from 512.
    min_n = 512
    max_n = 1000

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        # For each n, search for valid (a, b) pairs
        # Heuristic: try pairs that avoid small cycles
        # a should be "far" from 1, and b from 1 and a
        # Good candidates: a ~ n/3, b ~ n/2 (but not too close)

        # Generate candidate (a, b) pairs with heuristic ordering
        candidates = _generate_candidates(n)

        for a, b in candidates:
            if time.time() > deadline - 3:
                return None

            # Fast girth check using non-backtracking walks from vertex 0
            if _has_short_cycle_fast(n, a, b, g):
                continue

            # Construct edges
            edges = []
            for i in range(n):
                for shift in [1, a, b]:
                    j = (i + shift) % n
                    if i < j:
                        edges.append((i, j))

            return n, edges

    return None


def _generate_candidates(n):
    """
    Generate candidate (a, b) pairs for C_n(1, a, b) in order of promise.
    Pairs that avoid obvious short cycles are tried first.
    """
    candidates = []
    half = n // 2

    for a in range(2, half):
        for b in range(a + 1, half):
            # Quick rejection: avoid pairs that create 4-cycles
            # 4-cycle if any of ±1±1, ±1±a, ±1±b, ±a±a, ±a±b, ±b±b = ±1, ±a, ±b (mod n)
            # i.e., if a ≡ ±2, ±(a±1), etc.
            if _quick_reject(n, a, b):
                continue
            candidates.append((a, b))

    return candidates


def _quick_reject(n, a, b):
    """
    Quick rejection for pairs that definitely create short cycles.
    Checks for 4-cycles and some 6-cycle conditions.
    """
    # 4-cycle conditions for C_n(1, a, b):
    # A 4-cycle exists if there are steps s1, s2, s3, s4 in {±1, ±a, ±b}
    # with s1+s2+s3+s4 = 0 and no two consecutive steps are inverses.
    # Simple sufficient conditions for 4-cycle:
    # - a ≡ 2 (mod n): then 0-1-2-3-2-1-0... wait, that's not right.
    #   0 -a-> a, a -(a)-> 2a... no.
    #   Actually: 0 -1-> 1, 1 -a-> 1-a, (1-a) +a -> 1, 1 +1 -> 2. Not a cycle.
    #   Let me think again. A 4-cycle: v0-v1-v2-v3-v0.
    #   v1 = v0 ± s1, v2 = v1 ± s2, v3 = v2 ± s3, v0 = v3 ± s4.
    #   So ±s1 ± s2 ± s3 ± s4 = 0.
    #   With steps from {1, a, b}, a 4-cycle can be:
    #   1 + a + (-1) + (-a) = 0 -> this is just backtracking, not a 4-cycle.
    #   1 + a + (-b) + (-(a-b))... no, steps must be from the generator set.
    #   Actually for a 4-cycle we need s1+s2+s3+s4=0 where each si is in {±1,±a,±b}
    #   and the path doesn't immediately backtrack.
    #   Simple cases:
    #   - 1 + 1 + (-1) + (-1) = 0 -> backtracking
    #   - 1 + a + (-1) + (-a) = 0 -> backtracking
    #   - 1 + a + (-b) + (b-a) -> b-a must be in {±1,±a,±b}
    #   - a + a + (-a) + (-a) = 0 -> backtracking
    #   - 1 + a + a + (-1-2a) -> -1-2a must be in {±1,±a,±b}
    #
    # Let me just check: does any pair of distinct generators sum to ±1, ±a, or ±b?
    # If s1 + s2 = s3 where s1,s2,s3 are distinct generators (up to sign),
    # then 0 -s1-> s1 -s2-> s1+s2 = s3 -s3-> 0 gives a 4-cycle (if no backtracking).
    # But we need s1 ≠ -s2 (no immediate backtrack) and the path to be valid.
    #
    # Simpler: check if 2a ≡ ±1, ±a, ±b (mod n), or a+b ≡ ±1, ±a, ±b, etc.
    # Actually the cleanest check: for the set G = {1, -1, a, -a, b, -b},
    # check if any g1 + g2 = g3 for distinct g1, g2, g3 in G (with g1 ≠ -g2).
    # This is a sufficient condition for a 4-cycle.

    G = [1, -1, a, -a, b, -b]
    G_mod = [x % n for x in G]

    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            if G_mod[i] == (-G_mod[j]) % n:  # backtracking
                continue
            s = (G_mod[i] + G_mod[j]) % n
            # Check if s is in G_mod
            for k in range(6):
                if k == i or k == j:
                    continue
                if s == G_mod[k]:
                    return True
    return False


def _has_short_cycle_fast(n, a, b, g):
    """
    Fast girth check for C_n(1, a, b) using non-backtracking walks from vertex 0.
    Returns True if a cycle shorter than g is found, False otherwise.
    Uses the fact that the graph is vertex-transitive, so checking from 0 suffices.
    """
    # Non-backtracking BFS: state is (vertex, previous_vertex)
    # dist[(u, v)] = length of shortest non-backtracking walk from 0 to u, arriving from v
    # A cycle of length L < g exists if there's an edge (u, v) such that
    # the shortest non-backtracking walk from 0 to u (arriving from some w ≠ v)
    # plus the edge to v gives a path, and then we check if v can reach back to 0
    # without backtracking...

    # Actually, simpler approach: BFS with parent tracking.
    # dist[u] = shortest distance from 0
    # parent[u] = the vertex we came from
    # A cycle exists if there's an edge (u, v) where parent[u] != v and parent[v] != u
    # and dist[u] + dist[v] + 1 < g.
    # But this can give false positives (the paths might share vertices).
    # For a rigorous check, we need non-backtracking walks.

    # Let's do a proper non-backtracking BFS.
    # State: (current, previous) where previous is the vertex we came from.
    # Start: (0, -1) meaning we're at 0 with no previous.
    # From state (u, prev), we can go to (v, u) for each neighbor v of u where v != prev.
    # We track dist[(u, prev)] = length of walk.
    # A cycle of length L is found if we have a state (u, prev) and u is adjacent to
    # the starting vertex 0 (or more generally, if we can close a cycle).
    # Actually, a cycle exists if there's a non-backtracking walk from 0 back to 0
    # of length L < g. But non-backtracking walks from 0 to 0 of length L correspond
    # to cycles of length L (for L >= 3).

    # More precisely: a cycle of length L >= 3 exists iff there's a non-backtracking
    # walk of length L from 0 to 0. For L < g, we need to check if such a walk exists.

    # BFS over states (u, prev):
    # dist[(u, prev)] = min length of non-backtracking walk from 0 to u arriving from prev
    # Initialize: dist[(0, -1)] = 0
    # Transition: from (u, prev), for each neighbor v of u with v != prev:
    #   new_state = (v, u)
    #   new_dist = dist[(u, prev)] + 1
    #   if new_dist < g and new_dist < dist.get(new_state, inf):
    #       dist[new_state] = new_dist
    #       enqueue(new_state)
    # A cycle of length L < g exists if dist[(0, u)] < g for some neighbor u of 0.
    # (This means there's a non-backtracking walk from 0 to 0 of length L via u.)

    neighbors = {1: a, a: b, b: 1}  # not quite, let me build adjacency

    def get_neighbors(u):
        return [(u + 1) % n, (u - 1) % n, (u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]

    # BFS
    # Use a dictionary for dist
    dist = {}
    queue = deque()
    start_state = (0, -1)
    dist[start_state] = 0
    queue.append(start_state)

    max_len = g - 1  # we're looking for cycles of length < g, so up to g-1

    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]

        if d >= max_len:
            continue

        for v in get_neighbors(u):
            if v == prev:
                continue  # non-backtracking
            new_state = (v, u)
            new_d = d + 1
            if new_d > max_len:
                continue
            if new_state not in dist or new_d < dist[new_state]:
                dist[new_state] = new_d
                # Check if we've found a cycle: v == 0 and u != -1 (i.e., we're returning to 0)
                if v == 0 and u != -1:
                    # Cycle of length new_d
                    return True
                queue.append(new_state)

    return False


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    Returns True if a short cycle is found.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])

    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
