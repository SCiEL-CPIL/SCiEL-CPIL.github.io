"""Voltage-graph (cyclic cover) construction for the 5-regular girth-9 cage problem.

We lift the base 5-regular graph K_{5,5} (girth 4) by a cyclic group Z_m.  Each base
edge {u,v} carries a voltage sigma in Z_m.  A closed walk lifts to a cycle iff its
combined voltage is the identity.  We search over voltage assignments to find the
smallest m such that every non-backtracking closed walk of length < 9 in the base
graph has non-zero voltage, giving a 5-regular graph of girth >= 9 with 5*m vertices.
"""
import itertools
from collections import deque

_K = 5
_BASE_V = 10
_BASE_EDGES = [
    (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
]
_EDGE_IDX = {}
for _i, (_a, _b) in enumerate(_BASE_EDGES):
    _EDGE_IDX[(_a, _b)] = _i
    _EDGE_IDX[(_b, _a)] = _i

# Precompute all non-backtracking closed walks of length 4, 6, 8 in K_{5,5}
# as lists of edge indices. We use canonical form (sorted tuple of edge indices
# with the first edge appearing first in the walk) to deduplicate.

def _enumerate_nonbacktracking_closed_walks(L):
    """Enumerate all non-backtracking closed walks of length L in K_{5,5}.
    Returns a list of tuples of edge indices (canonical form)."""
    cset = set()
    # For each starting directed edge, do a DFS of non-backtracking walks
    # and check if they close back to start with length L.
    for a in range(_BASE_V):
        na = [b for b in range(_BASE_V) if (a, b) in _EDGE_IDX]
        for i1 in range(len(na)):
            b = na[i1]
            e1 = _EDGE_IDX[a, b]
            # Walk: a -> b, then continue for L-1 more steps
            # Non-backtracking: can't go back to a from b
            if L == 4:
                # b -> c -> d -> a, where c != a, d != b, d adjacent to a
                nb = [x for x in range(_BASE_V) if (b, x) in _EDGE_IDX and x != a]
                for c in nb:
                    e2 = _EDGE_IDX[b, c]
                    nc = [x for x in range(_BASE_V) if (c, x) in _EDGE_IDX and x != b]
                    for d in nc:
                        e3 = _EDGE_IDX[c, d]
                        if (d, a) in _EDGE_IDX and d != c:
                            e4 = _EDGE_IDX[d, a]
                            edges = (e1, e2, e3, e4)
                            # Canonical form: rotate so min edge is first, then pick direction
                            min_e = min(edges)
                            idx = edges.index(min_e)
                            rot = edges[idx:] + edges[:idx]
                            # Try both directions
                            rev = tuple(reversed(rot))
                            canon = min(rot, rev)
                            cset.add(canon)
            elif L == 6:
                nb = [x for x in range(_BASE_V) if (b, x) in _EDGE_IDX and x != a]
                for c in nb:
                    e2 = _EDGE_IDX[b, c]
                    nc = [x for x in range(_BASE_V) if (c, x) in _EDGE_IDX and x != b]
                    for d in nc:
                        e3 = _EDGE_IDX[c, d]
                        nd = [x for x in range(_BASE_V) if (d, x) in _EDGE_IDX and x != c]
                        for e in nd:
                            e4 = _EDGE_IDX[d, e]
                            ne = [x for x in range(_BASE_V) if (e, x) in _EDGE_IDX and x != d]
                            for f in ne:
                                e5 = _EDGE_IDX[e, f]
                                if (f, a) in _EDGE_IDX and f != e:
                                    e6 = _EDGE_IDX[f, a]
                                    edges = (e1, e2, e3, e4, e5, e6)
                                    min_e = min(edges)
                                    idx = edges.index(min_e)
                                    rot = edges[idx:] + edges[:idx]
                                    rev = tuple(reversed(rot))
                                    canon = min(rot, rev)
                                    cset.add(canon)
            elif L == 8:
                nb = [x for x in range(_BASE_V) if (b, x) in _EDGE_IDX and x != a]
                for c in nb:
                    e2 = _EDGE_IDX[b, c]
                    nc = [x for x in range(_BASE_V) if (c, x) in _EDGE_IDX and x != b]
                    for d in nc:
                        e3 = _EDGE_IDX[c, d]
                        nd = [x for x in range(_BASE_V) if (d, x) in _EDGE_IDX and x != c]
                        for e in nd:
                            e4 = _EDGE_IDX[d, e]
                            ne = [x for x in range(_BASE_V) if (e, x) in _EDGE_IDX and x != d]
                            for f in ne:
                                e5 = _EDGE_IDX[e, f]
                                nf = [x for x in range(_BASE_V) if (f, x) in _EDGE_IDX and x != e]
                                for g in nf:
                                    e6 = _EDGE_IDX[f, g]
                                    ng = [x for x in range(_BASE_V) if (g, x) in _EDGE_IDX and x != f]
                                    for h in ng:
                                        e7 = _EDGE_IDX[g, h]
                                        if (h, a) in _EDGE_IDX and h != g:
                                            e8 = _EDGE_IDX[h, a]
                                            edges = (e1, e2, e3, e4, e5, e6, e7, e8)
                                            min_e = min(edges)
                                            idx = edges.index(min_e)
                                            rot = edges[idx:] + edges[:idx]
                                            rev = tuple(reversed(rot))
                                            canon = min(rot, rev)
                                            cset.add(canon)
    return list(cset)


def _precompute_cycles():
    """Precompute all non-backtracking closed walks of lengths 4, 6, 8."""
    four = _enumerate_nonbacktracking_closed_walks(4)
    six = _enumerate_nonbacktracking_closed_walks(6)
    eight = _enumerate_nonbacktracking_closed_walks(8)
    return four, six, eight


def _voltage_of_walk(edges, sigma):
    """Compute the total voltage of a walk (sum of edge voltages mod nothing, just integer sum)."""
    return sum(sigma[e] for e in edges)


def _find_voltages(four, six, eight):
    """Search for voltage assignments sigma in Z_m (for various m) such that
    no non-backtracking closed walk of length 4, 6, or 8 has zero voltage.
    Returns (sigma, m) or None."""
    # First, we need to ensure 4-cycles have non-zero voltage.
    # A 4-cycle in K_{5,5} has 4 edges. Its voltage is the sum of 4 edge voltages.
    # For m >= 5, we can assign voltages 1..25 to the 25 edges.
    # Let's try small m values and search for valid sigma.
    
    for m in range(5, 500):
        # Try to find sigma in Z_m^25 such that for every cycle C of length 4,6,8:
        # sum_{e in C} sigma[e] != 0 (mod m)
        
        # Use a greedy/local search approach
        import random
        random.seed(42)
        
        best_sigma = None
        for attempt in range(200):
            if attempt == 0:
                sigma = [i % m for i in range(25)]
            else:
                sigma = [random.randint(0, m - 1) for _ in range(25)]
            
            # Check and fix 4-cycles
            for _ in range(100):
                bad = []
                for c in four:
                    if _voltage_of_walk(c, sigma) % m == 0:
                        bad.append(c)
                if not bad:
                    break
                # Fix by adjusting one edge in the first bad cycle
                c = bad[0]
                e = c[0]
                sigma[e] = (sigma[e] + 1) % m
            
            # Check 6-cycles
            for _ in range(100):
                bad = []
                for c in six:
                    if _voltage_of_walk(c, sigma) % m == 0:
                        bad.append(c)
                if not bad:
                    break
                c = bad[0]
                e = c[0]
                sigma[e] = (sigma[e] + 1) % m
            
            # Check 8-cycles
            for _ in range(100):
                bad = []
                for c in eight:
                    if _voltage_of_walk(c, sigma) % m == 0:
                        bad.append(c)
                if not bad:
                    break
                c = bad[0]
                e = c[0]
                sigma[e] = (sigma[e] + 1) % m
            
            # Final check
            ok = True
            for c in four:
                if _voltage_of_walk(c, sigma) % m == 0:
                    ok = False
                    break
            if ok:
                for c in six:
                    if _voltage_of_walk(c, sigma) % m == 0:
                        ok = False
                        break
            if ok:
                for c in eight:
                    if _voltage_of_walk(c, sigma) % m == 0:
                        ok = False
                        break
            if ok:
                best_sigma = sigma
                break
        
        if best_sigma is not None:
            return best_sigma, m
    
    return None


def _lift(sigma, m):
    """Lift K_{5,5} by Z_m using voltages sigma."""
    n = _BASE_V * m
    adj = [[] for _ in range(n)]
    for (u, v) in _BASE_EDGES:
        e = _EDGE_IDX[(u, v)]
        s = sigma[e] % m
        for f in range(m):
            a = u * m + f
            b = v * m + (f + s) % m
            adj[a].append(b)
            adj[b].append(a)
    return n, adj


def _verify(n, adj, g):
    """Verify the graph is k-regular, simple, and has girth >= g."""
    deg = [len(a) for a in adj]
    if any(d != _K for d in deg):
        return False
    
    # Check simplicity
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
    
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        for v in adj[s]:
            queue = deque()
            queue.append((v, s))
            dist = 1
            while queue and dist < g:
                size = len(queue)
                for _ in range(size):
                    node, prev = queue.popleft()
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        if nb == s:
                            return False
                        queue.append((nb, node))
                dist += 1
    return True


def _construct_voltage(k, g):
    """Construct a k-regular graph of girth >= g using voltage lift of K_{k,k}."""
    if k != _K or g != 9:
        return None
    
    four, six, eight = _precompute_cycles()
    result = _find_voltages(four, six, eight)
    if result is None:
        return None
    
    sigma, m = result
    n, adj = _lift(sigma, m)
    
    if not _verify(n, adj, g):
        return None
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return n, edges


def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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


def construct(k, g):
    res = _construct_voltage(k, g)
    if res is not None:
        return res
    return _construct_luw(k, g)
