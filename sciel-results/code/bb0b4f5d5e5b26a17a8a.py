"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct(k, g):
    """Construct a k-regular graph with girth >= g using circulant graph search."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Moore bound lower bound
    if k == 2:
        n_min = g
    elif k == 3:
        # Moore bound for 3-regular: 2 * (3 * (3**(g//2 - 1) - 1)/2) for even g
        # For girth 16: 2 * (3 * (3^7 - 1)/2) = 3*(2187-1) = 6558... wait
        # Actually for 3-regular even girth g: n >= 2 * (3 * (3^(g/2 - 1) - 1) / (3-1))
        # = 2 * (3 * (3^7 - 1) / 2) = 3 * (2187 - 1) = 6558? That seems too high.
        # Let me use the standard Moore bound: for k-regular, girth g (even):
        # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * (k^((g/2)) - 1) / (k-1)
        half_g = g // 2
        n_min = 2 * (k ** half_g - 1) // (k - 1)
    else:
        half_g = g // 2
        n_min = 2 * (k ** half_g - 1) // (k - 1)

    # For k=3, g=16: n_min = 2 * (3^8 - 1) / 2 = 6560... that's the Moore bound
    # But the known best is 936, so the Moore bound is loose.
    # Let's search circulant graphs.

    best_n = None
    best_edges = None

    # Search circulant graphs C_n({a, b, c}) for k=3
    # A circulant graph on n vertices with connection set S has edges (i, i+s mod n) for s in S
    # For 3-regular: S = {a, -a, b} or {a, b, -b} etc. (need exactly 3 distinct neighbors)
    # Actually for 3-regular circulant: S must have 3 elements, no element is its own inverse (unless n even and s = n/2)
    # Simplest: S = {a, n-a, b} where b != a and b != n-a, and b != n-b (i.e., b != n/2)

    if k != 3:
        # Fallback to LUW for other k values
        return _luw_construct(k, g)

    # Search over n and connection sets
    # Start from reasonable range
    n_start = max(n_min, 40)
    n_end = min(2000, int(deadline * 200))  # rough upper bound based on time

    for n in range(n_start, n_end + 1):
        if time.time() > deadline - 5.0:
            break

        # For 3-regular circulant: S = {a, n-a, b} where:
        # - 1 <= a < n/2 (to avoid redundancy, a < n-a)
        # - 1 <= b < n/2, b != a, b != n/2 (to avoid self-inverse)
        # - b != n-a (redundant since b < n/2 < n-a when a < n/2)
        # Actually simpler: pick 3 distinct offsets s1, s2, s3 where s_i != n - s_j for all i,j
        # and no s_i = 0, and the set {s1, s2, s3} with their negatives gives exactly 3 neighbors.
        # For 3-regular: we need exactly 3 neighbors, so S ∪ (-S) = {±s1, ±s2, ±s3} but we want |S|=3
        # and S ∩ (-S) = empty. So S = {s1, s2, s3} where no si = n - sj.

        # Try S = {a, b, c} with 1 <= a < b < c < n/2, and c != n/2
        # This gives neighbors at ±a, ±b, ±c... wait that's 6 neighbors.
        # For 3-regular: S = {a, b, n/2} if n even, or S = {a, n-a, b}... 
        # Let me think again. In a circulant graph, vertex i connects to i+s mod n for s in S.
        # Degree = |S| if S contains no pairs {s, n-s}. 
        # For degree 3: S has 3 elements, none is the negative of another.
        # S = {a, b, c} with a, b, c all in 1..n-1, and {a,b,c} ∩ {n-a, n-b, n-c} = empty.
        # Simplest: a, b, c all in 1..(n-1)/2, and none equals n/2 (if n even).

        if n % 2 == 0:
            max_s = n // 2 - 1
        else:
            max_s = (n - 1) // 2

        if max_s < 3:
            continue

        # Sample some connection sets to keep time bounded
        # Try S = {a, b, c} for various combinations
        # To limit search: try a few patterns
        found = False
        for a in range(1, min(max_s, 20) + 1):
            if time.time() > deadline - 3.0:
                break
            for b in range(a + 1, min(max_s, a + 20) + 1):
                if time.time() > deadline - 3.0:
                    break
                for c in range(b + 1, min(max_s, b + 20) + 1):
                    if time.time() > deadline - 3.0:
                        break
                    S = [a, b, c]
                    # Check no s is negative of another: since all < n/2, this is automatic
                    # Build graph and check girth
                    n_test = n
                    adj = [[] for _ in range(n_test)]
                    for i in range(n_test):
                        for s in S:
                            j = (i + s) % n_test
                            if j not in adj[i]:
                                adj[i].append(j)
                            if i not in adj[j]:
                                adj[j].append(i)
                    # Check degree
                    if any(len(adj[i]) != 3 for i in range(n_test)):
                        continue
                    # Check girth using BFS from each vertex (limited depth)
                    girth = _compute_girth(adj, n_test, g)
                    if girth >= g:
                        # Found valid graph
                        if best_n is None or n_test < best_n:
                            best_n = n_test
                            best_edges = []
                            for i in range(n_test):
                                for j in adj[i]:
                                    if i < j:
                                        best_edges.append((i, j))
                            found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            # Keep searching for smaller n
            pass

    if best_n is not None:
        return best_n, best_edges

    # Fallback: use LUW construction
    return _luw_construct(k, g)


def _compute_girth(adj, n, target_g):
    """Compute girth of graph. Returns girth if < target_g, else target_g."""
    # BFS from each vertex, track shortest cycle
    # For efficiency, only search up to depth target_g//2
    max_depth = target_g // 2
    for start in range(n):
        # BFS tracking parent to detect cycles
        visited = [-1] * n  # visited[v] = depth
        parent = [-1] * n
        queue = [(start, 0)]
        visited[start] = 0
        found_cycle = None
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in adj[node]:
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    parent[neighbor] = node
                    queue.append((neighbor, depth + 1))
                elif neighbor != parent[node]:
                    # Cycle found: depth + 1 + visited[neighbor]
                    cycle_len = depth + 1 + visited[neighbor]
                    if found_cycle is None or cycle_len < found_cycle:
                        found_cycle = cycle_len
                    if found_cycle < target_g:
                        return found_cycle
        if found_cycle is not None and found_cycle < target_g:
            return found_cycle
    return target_g


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
# EVOLVE-BLOCK-END
