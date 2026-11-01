"""
Voltage-graph lift of the Petersen graph over Z_m to construct a 3-regular graph
of girth at least 16 with as few vertices as possible.

The Petersen graph is the smallest 3-regular graph of girth 5 (the (3,5)-cage).
Lifting it over Z_m with carefully chosen edge voltages can produce larger
3-regular graphs with much larger girth. The girth of the lift is at least 16
if every non-backtracking closed walk of length < 16 in the base graph has a
voltage sum that is not 0 mod m.

Strategy:
  1. Fix the Petersen graph as base (10 vertices, 15 edges, 3-regular).
  2. Assign voltages from a small set of "generators" to each edge.
  3. For a given m, the lift has 10*m vertices.
  4. Use non-backtracking BFS from every vertex to certify girth >= g.
  5. Scan m from the Moore bound upward, trying different voltage assignments
     (e.g., based on powers of a primitive root, or random search).

The Moore bound for n(3,16) is 512. So we start scanning m from 512/10 = 51.2
upward, i.e., m >= 52. The total vertex count is 10*m.

We try several voltage assignment families and pick the smallest m that works.
"""

import time
import random
from collections import deque

_DEADLINE = None

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return True
    return False


# Petersen graph: vertices 0..9, edges
# Standard Petersen: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes i-(i+5)
PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
]

# Build adjacency for Petersen
PETERSEN_ADJ = [[] for _ in range(10)]
for u, v in PETERSEN_EDGES:
    PETERSEN_ADJ[u].append(v)
    PETERSEN_ADJ[v].append(u)

# For non-backtracking walks, we need to track (vertex, prev_vertex) states.
# The number of directed edges is 30.

def _build_nb_states():
    """Build non-backtracking state transitions for the Petersen graph.
    State: (u, v) meaning we arrived at v from u.
    Transition: (u, v) -> (v, w) for w in adj[v], w != u.
    """
    states = []
    state_id = {}
    for u in range(10):
        for v in PETERSEN_ADJ[u]:
            states.append((u, v))
            state_id[(u, v)] = len(states) - 1
    n_states = len(states)
    trans = [[] for _ in range(n_states)]
    for idx, (u, v) in enumerate(states):
        for w in PETERSEN_ADJ[v]:
            if w != u:
                trans[idx].append(state_id[(v, w)])
    return states, trans


_STATES, _TRANS = _build_nb_states()
N_STATES = len(_STATES)  # 30


def _girth_of_lift(m, voltages):
    """
    Compute the girth of the voltage lift using non-backtracking BFS from every vertex.
    voltages: list of 15 integers (mod m) for each edge in PETERSEN_EDGES order.
    Returns the girth (length of shortest cycle), or a large number if no cycle < some bound.
    We only need to check up to g-1 = 15.
    """
    # For the lift, a non-backtracking walk of length L corresponds to a sequence of
    # directed edges in the base graph, with voltage sum = 0 mod m for a closed walk.
    # The girth is the minimum L >= 3 such that there exists a non-backtracking closed
    # walk of length L with voltage sum 0 mod m.
    #
    # We can do BFS in the state space (base_state, voltage_mod_m) but that's 30*m states.
    # For m up to ~1000, that's 30000 states, very fast.
    #
    # Actually, we need the shortest cycle, which is the shortest non-backtracking closed
    # walk. We can BFS from each starting state, tracking (state, voltage) and the depth.
    # But we need to be careful: a cycle of length L means we return to the same (state, voltage)
    # after L steps. Actually, we return to the same vertex in the lift, which means same
    # base vertex AND same voltage offset.
    #
    # Let's think differently. The lift graph has 10*m vertices. Vertex (v, t) where v is
    # base vertex and t is in Z_m. Edge between (u, t) and (v, t + e_uv) for edge (u,v) with voltage e_uv.
    #
    # For girth checking, we do BFS from each vertex in the lift, but that's 10*m BFS calls
    # each on 10*m vertices. For m=100, that's 1000 vertices, 1000 BFS calls, each visiting
    # up to 1000 vertices = 1M operations per m. That's fine.
    #
    # But we can be smarter: use the non-backtracking approach on the base graph with voltage tracking.
    #
    # Let's just do BFS on the actual lift graph for simplicity, with early termination at depth 15.

    n = 10 * m
    # Build adjacency of the lift
    # Vertex id: v * m + t
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(PETERSEN_EDGES):
        vol = voltages[idx] % m
        for t in range(m):
            a = u * m + t
            b = v * m + (t + vol) % m
            adj[a].append(b)
            c = v * m + t
            d = u * m + (t - vol) % m
            adj[c].append(d)

    # BFS from each vertex, tracking depth, stop at depth 15
    # Use a global visited with depth to avoid re-BFSing
    # Actually, for girth we need the shortest cycle, so BFS from each vertex and find
    # the shortest return. But that's expensive. Instead, we can do a single BFS-like
    # exploration tracking (vertex, depth) and detect cycles.
    #
    # Simpler: for each vertex, BFS up to depth 15, and if we revisit a vertex at depth d
    # that was first seen at depth d', the cycle length is d - d' + ... no, that's not right
    # for simple cycles.
    #
    # The cleanest correct approach: non-backtracking BFS from each directed edge.
    # A cycle of length L corresponds to a non-backtracking walk of length L that returns
    # to the starting vertex (and the walk is closed).
    #
    # Let's do: for each starting vertex s, BFS in the lift up to depth 15, keeping
    # (vertex, parent) to avoid backtracking. If we reach s again at depth L >= 3, we have
    # a cycle of length L.
    #
    # To be efficient, we do this for all s but stop early if we find a cycle < 16.

    for s in range(n):
        # BFS from s, no backtracking
        # visited: vertex -> depth
        visited = {s: 0}
        queue = deque([(s, -1, 0)])  # (vertex, parent, depth)
        while queue:
            v, parent, depth = queue.popleft()
            if depth >= 15:
                continue
            for w in adj[v]:
                if w == parent:
                    continue
                if w == s and depth + 1 >= 3:
                    return depth + 1
                if w not in visited:
                    visited[w] = depth + 1
                    queue.append((w, v, depth + 1))
    return 100  # no cycle found up to depth 15


def _try_voltage_assignment(m, voltages):
    """Check if the given voltage assignment on the Petersen lift over Z_m has girth >= 16."""
    g = _girth_of_lift(m, voltages)
    return g >= 16


def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 140  # safety margin before 150s kill

    if k != 3:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction for general k
        return _luw_construct(k, g)

    # For k=3, use Petersen lift
    # Moore bound for n(3,16) is 512, so m >= 52 (10*m >= 512 -> m >= 51.2)
    # The Petersen lift has 10*m vertices.
    # We want the smallest m such that some voltage assignment gives girth >= 16.

    # Try different voltage assignment strategies:
    # 1. All edges get voltage = 1 (simplest)
    # 2. Powers of a generator
    # 3. Random search

    # First, try simple assignments for small m
    base_m = 52  # 10*52 = 520 >= 512

    # Try m from base_m upward
    best_result = None

    for m in range(base_m, 200):
        if _check_deadline():
            break

        # Try several voltage assignments
        # Assignment 1: all ones
        vols1 = [1] * 15
        if _try_voltage_assignment(m, vols1):
            best_result = (10 * m, _build_lift_edges(m, vols1))
            break

        # Assignment 2: alternating 1, 2, 3...
        vols2 = [(i % 3) + 1 for i in range(15)]
        if _try_voltage_assignment(m, vols2):
            best_result = (10 * m, _build_lift_edges(m, vols2))
            break

        # Assignment 3: use a primitive root-like sequence
        # Try small sets of generators
        for gen in range(2, min(m, 20)):
            if _check_deadline():
                break
            vols3 = [pow(gen, i, m) if m > 1 else 0 for i in range(15)]
            # Make sure voltages are non-zero
            if all(v != 0 for v in vols3):
                if _try_voltage_assignment(m, vols3):
                    best_result = (10 * m, _build_lift_edges(m, vols3))
                    break
        if best_result:
            break

        # Assignment 4: random
        for _ in range(20):
            if _check_deadline():
                break
            vols4 = [random.randint(1, m - 1) for _ in range(15)]
            if _try_voltage_assignment(m, vols4):
                best_result = (10 * m, _build_lift_edges(m, vols4))
                break
        if best_result:
            break

    if best_result is None:
        # Fallback: use a larger m with all-ones voltage (guaranteed to work for large m?
        # Actually all-ones gives girth = girth of Petersen = 5, so that won't work.)
        # Use the LUW construction as fallback
        return _luw_construct(k, g)

    n, edges = best_result
    # Verify
    if not _verify(k, g, n, edges):
        return _luw_construct(k, g)
    return n, edges


def _build_lift_edges(m, voltages):
    """Build the edge list for the Petersen lift over Z_m."""
    edges = []
    for idx, (u, v) in enumerate(PETERSEN_EDGES):
        vol = voltages[idx] % m
        for t in range(m):
            a = u * m + t
            b = v * m + (t + vol) % m
            edges.append((a, b))
    return edges


def _verify(k, g, n, edges):
    """Verify the graph is k-regular, simple, and has girth >= g."""
    if len(edges) != n * k // 2:
        return False
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Check girth with non-backtracking BFS from each vertex
    for s in range(n):
        visited = {s: 0}
        queue = deque([(s, -1, 0)])
        while queue:
            v, parent, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for w in adj[v]:
                if w == parent:
                    continue
                if w == s and depth + 1 >= 3:
                    if depth + 1 < g:
                        return False
                if w not in visited:
                    visited[w] = depth + 1
                    queue.append((w, v, depth + 1))
    return True


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
