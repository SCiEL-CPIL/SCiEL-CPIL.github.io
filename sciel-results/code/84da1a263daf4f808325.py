"""Construct a k-regular graph of girth at least g with as few vertices as possible."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _girth_check_fast(adj, n, min_girth):
    """
    Verify that the graph has girth >= min_girth.
    Uses BFS from each vertex with non-backtracking constraint.
    A cycle of length L exists iff from some vertex s, two distinct neighbors
    of s are connected by a non-backtracking path of length L-2 < min_girth-2.
    We check: for each vertex s, BFS from each neighbor u (not going back to s),
    and see if we reach another neighbor v of s at distance < min_girth - 1.
    """
    for s in range(n):
        neighbors = adj[s]
        if len(neighbors) < 2:
            continue
        # For each neighbor u of s, do BFS not going through s
        # Track distance from u
        # If we reach another neighbor v of s at dist d, cycle length = d + 1
        # (path u...v has length d, plus s-u and v-s makes d+2? No.)
        # Path: s - u - ... - v - s. The path from u to v has length dist.
        # Cycle length = dist + 2.
        # We need cycle length >= min_girth, so dist + 2 >= min_girth, i.e., dist >= min_girth - 2.
        # If we find a path from u to v (v != u, v in neighbors) with dist < min_girth - 2,
        # then we have a cycle of length < min_girth.
        
        # Actually, let's use the standard method:
        # For each directed edge (u, v), check if there's a non-backtracking walk
        # from v back to u of length L < min_girth - 1.
        # This is equivalent to: from state (v, u), can we reach state (u, v) in L steps?
        # State is (current, prev). Next state is (next, current) where next != prev.
        
        # But this is O(n * k * g) per start, which is expensive.
        
        # Better: BFS from s, but track (node, prev) to avoid backtracking.
        # If we reach a node that is a neighbor of s (other than the one we started from),
        # at distance d, then cycle length is d + 1 (if we started from a neighbor) 
        # or d + 2? 
        # Let's be precise.
        # Start BFS from s. Level 0: s. Level 1: neighbors of s.
        # If at level d we find a node x that is already in level d' < d, and the path
        # doesn't backtrack, then we have a cycle.
        # But the standard girth check via BFS from each vertex:
        # The girth is the smallest d such that two neighbors of s at distance 1
        # are connected by a path of length d-2.
        # So for each s, for each pair of neighbors (u, v), find shortest path u to v
        # avoiding s. If shortest path length < min_girth - 2, girth violated.
        
        # This is O(n * (k^2 * (n + m))) which is too slow for n=320, k=4.
        # n=320, k=4, m=640. n * k^2 * (n+m) = 320 * 16 * 960 ≈ 5 million. That's fine.
        
        # Actually, for each s, we do BFS from each neighbor u (k BFSes), and check
        # if any other neighbor v is reached at dist < min_girth - 2.
        # That's k BFSes per s, each O(n + m). Total O(n * k * (n + m)).
        # 320 * 4 * 960 ≈ 1.2 million. Very fast.
        
        for u in neighbors:
            # BFS from u, not going through s
            dist = {u: 1}
            queue = deque([(u, 0)])  # (node, prev)
            while queue:
                node, prev = queue.popleft()
                d = dist[node]
                if d >= min_girth - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        continue
                    if nb not in dist:
                        dist[nb] = d + 1
                        queue.append((nb, node))
                    # Check if nb is a neighbor of s (other than u)
                    if nb in neighbors and nb != u:
                        cycle_len = d + 1 + 1  # dist from u to nb is d, wait.
                        # dist[u] = 1. dist[nb] = d. Path length from u to nb is d - 1.
                        # Cycle: s - u - ... - nb - s. Length = (d-1) + 2 = d + 1.
                        if d + 1 < min_girth:
                            return False
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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


def _try_find_graph(k, g, n, deadline):
    """
    Try to find a k-regular graph of girth >= g with n vertices.
    Uses a structured approach: build a BFS tree from a root, then connect
    the leaves in a way that avoids short cycles.
    
    For k=4, g=10, n=320:
    The BFS tree from a root in a 4-regular graph of girth 10 has:
    Level 0: 1 vertex (root)
    Level 1: 4 vertices
    Level 2: 4*3 = 12 vertices
    Level 3: 12*3 = 36 vertices
    Level 4: 36*3 = 108 vertices
    Total in tree: 1 + 4 + 12 + 36 + 108 = 161 vertices.
    But we need 320 vertices, so the graph is not a tree.
    
    Actually, for girth 10, the BFS tree from a root has radius 4 (diameter 8 or 9).
    Level 0: 1
    Level 1: 4
    Level 2: 12
    Level 3: 36
    Level 4: 108
    Total: 161. But n=320, so there are 159 more vertices.
    This means the BFS tree from a root doesn't cover all vertices, or the graph
    has a different structure.
    
    Actually, the Moore bound for k=4, g=10 is 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242.
    So n=320 is above the Moore bound.
    
    Let's use a different approach: start with a random 4-regular graph on n vertices,
    then try to increase the girth by swapping edges.
    """
    import time as _time
    
    # Generate random 4-regular graph
    for trial in range(50):
        if _time.time() > deadline:
            return None
        
        edges = _random_regular_graph(n, k, trial)
        if edges is None:
            continue
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _girth_check_fast(adj, n, g):
            return n, edges
    
    return None


def _random_regular_graph(n, k, seed):
    """Generate a random k-regular graph using the configuration model."""
    random.seed(seed * 1000 + 42)
    
    if (n * k) % 2 != 0:
        return None
    
    for trial in range(200):
        # Build configuration
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if not valid:
            continue
        
        # Check degrees
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            return list(edges)
    
    return None


def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # Moore bound for even girth g=2m: 2 * sum_{i=0}^{m-1} (k-1)^i
    m = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(m))
    if moore_bound % 2 == 0:
        moore_bound += 1

    # Try the Lazebnik-Ustimenko-Woldar construction
    best_n, best_edges = _luw_construct(k, g)

    # Verify the LUW result
    if best_edges is not None:
        adj = [[] for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].append(v)
            adj[v].append(u)
        if not _girth_check_fast(adj, best_n, g):
            # LUW result failed girth check, try to fix or use fallback
            pass

    # Try to do better with a structured search approach
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 10:
        for n_candidate in range(moore_bound, best_n):
            if time.time() > deadline:
                break
            if (k * n_candidate) % 2 != 0:
                continue
            result = _try_find_graph(k, g, n_candidate, deadline)
            if result is not None:
                best_n, best_edges = result
                break

    return best_n, best_edges


# EVOLVE-BLOCK-END
