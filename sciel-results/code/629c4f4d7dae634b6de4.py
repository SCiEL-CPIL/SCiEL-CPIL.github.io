import time
import random
import math
from collections import deque

_DEADLINE = 150.0
_SAFETY = 10.0
_START = None


def _now():
    return time.time()


def _time_left():
    if _START is None:
        return _DEADLINE
    return _DEADLINE - _SAFETY - (_now() - _START)


def _girth(adj, n):
    """adj: list of sorted neighbor lists. Returns girth (int)."""
    best = 10**9
    for s in range(n):
        seen = {}
        q = deque()
        for nb in adj[s]:
            seen[(s, nb)] = 1
            q.append((s, nb, 1))
        while q:
            prev, v, dist = q.popleft()
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s and dist + 1 >= 3:
                    cyc = dist + 1
                    if cyc < best:
                        best = cyc
                    continue
                key = (v, nb)
                if key in seen:
                    cyc = seen[key] + dist + 1
                    if cyc < best:
                        best = cyc
                    continue
                seen[key] = dist + 1
                q.append((v, nb, dist + 1))
    return best


def _random_k_regular(n, k, rng):
    if n * k % 2:
        return None
    stubs = [v for v in range(n) for _ in range(k)]
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            return None
        if b in adj[a]:
            return None
        adj[a].add(b)
        adj[b].add(a)
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return [sorted(a) for a in adj]


def _edges_from_adj(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _try_2switch(adj, rng):
    n = len(adj)
    edge_list = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
    if len(edge_list) < 2:
        return False
    for _ in range(60):
        i = rng.randrange(len(edge_list))
        j = rng.randrange(len(edge_list))
        if i == j:
            continue
        a, b = edge_list[i]
        c, d = edge_list[j]
        if len({a, b, c, d}) < 4:
            continue
        if (c in adj[a] or d in adj[b] or b in adj[c] or a in adj[d]):
            continue
        adj[a].remove(b); adj[b].remove(a)
        adj[a].append(c); adj[b].append(d)
        adj[c].append(a); adj[d].append(b)
        adj[a].sort(); adj[b].sort(); adj[c].sort(); adj[d].sort()
        return True
    return False


def _local_search(n, k, g, rng, time_budget):
    t0 = _now()
    adj = _random_k_regular(n, k, rng)
    if adj is None:
        return None
    cur_g = _girth(adj, n)
    best_adj = [list(a) for a in adj]
    best_g = cur_g
    while _now() - t0 < time_budget and _time_left() > 0:
        if cur_g >= g:
            best_adj = [list(a) for a in adj]
            best_g = cur_g
            break
        if _try_2switch(adj, rng):
            new_g = _girth(adj, n)
            if new_g >= cur_g:
                cur_g = new_g
                if cur_g > best_g:
                    best_g = cur_g
                    best_adj = [list(a) for a in adj]
            else:
                adj = _random_k_regular(n, k, rng)
                if adj is None:
                    break
                cur_g = _girth(adj, n)
    if best_g >= g:
        return best_adj
    return None


def _luw(k, g):
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


def _moore_bound(k, g):
    if g % 2 == 0:
        m = g // 2
        return 1 + k * sum((k - 1) ** i for i in range(m - 1))
    else:
        m = (g - 1) // 2
        return k * sum((k - 1) ** i for i in range(m))


def _valid_circulant_coeffs(k, g):
    """
    For a k-regular circulant on Z_n with step set S (|S| = k),
    non-backtracking walks of length L have a net displacement that is a
    sum of L steps from S with no two consecutive steps being inverses.
    A cycle of length L exists iff some non-backtracking walk of length L
    has net displacement 0 mod n.
    
    For k=4, S = {a, -a, b, -b}. We track (coeff_a, coeff_b) of the
    displacement. A walk is non-backtracking if consecutive steps aren't
    inverses. We enumerate all possible (coeff_a, coeff_b) pairs for
    non-backtracking walks of lengths 3 through g-1.
    
    The circulant has girth >= g iff for all such (ca, cb):
        ca*a + cb*b != 0 (mod n)
    """
    if k != 4:
        return None  # Only handle k=4 for now
    
    # Steps: 1->a, -1->-a, 2->b, -2->-b
    # State: (last_step, coeff_a, coeff_b)
    # We collect all (coeff_a, coeff_b) reachable by non-backtracking walks
    # of length L for L = 3, 4, ..., g-1
    
    valid_coeffs = set()
    
    # Length 1 states
    states = {(1, 1, 0), (-1, -1, 0), (2, 0, 1), (-2, 0, -1)}
    
    for L in range(2, g):
        new_states = set()
        for last_step, ca, cb in states:
            for step in (1, -1, 2, -2):
                if step == -last_step:
                    continue
                nca = ca + (1 if step == 1 else -1 if step == -1 else 0)
                ncb = cb + (1 if step == 2 else -1 if step == -2 else 0)
                new_states.add((step, nca, ncb))
        states = new_states
        if L >= 3:
            for _, ca, cb in states:
                valid_coeffs.add((ca, cb))
    
    return valid_coeffs


def _check_circulant(n, a, b, valid_coeffs):
    """Check if circulant with steps ±a, ±b on Z_n has girth >= g."""
    for ca, cb in valid_coeffs:
        if (ca * a + cb * b) % n == 0:
            return False
    return True


def _circulant_edges(n, a, b):
    """Build edge list for 4-regular circulant with steps ±a, ±b."""
    edges = set()
    for i in range(n):
        for step in (a, -a, b, -b):
            j = (i + step) % n
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    return list(edges)


def _search_circulants(k, g, n_range, valid_coeffs, deadline):
    """Search for valid 4-regular circulants in the given n range."""
    if k != 4 or valid_coeffs is None:
        return None
    
    for n in n_range:
        if _now() > deadline:
            break
        # a must be in [1, n-1], b in [1, n-1], a != b, a != n-b, etc.
        # For 4-regular: need a != 0, b != 0, a != b, a != -b mod n, 2a != 0, 2b != 0
        # So: a in [1, n-1], b in [1, n-1], a != b, a + b != 0 mod n,
        #     2a != 0 mod n, 2b != 0 mod n
        
        max_a = n - 1
        max_b = n - 1
        
        # Try random (a, b) pairs
        rng = random.Random(n * 7919 + 13)
        trials = 500
        for _ in range(trials):
            if _now() > deadline:
                break
            a = rng.randint(1, max_a)
            b = rng.randint(1, max_b)
            if a == b:
                continue
            if (a + b) % n == 0:
                continue
            if 2 * a % n == 0:
                continue
            if 2 * b % n == 0:
                continue
            # Also need a != 0 mod n (already ensured by range)
            
            if _check_circulant(n, a, b, valid_coeffs):
                edges = _circulant_edges(n, a, b)
                return (n, edges)
    
    return None


def construct(k, g):
    global _START
    _START = _now()
    deadline = _now() + _DEADLINE - _SAFETY
    rng = random.Random(12345)

    moore = _moore_bound(k, g)

    # Phase 1: Circulant search (very fast per candidate)
    # For k=4, g=10: Moore bound = 1 + 4*(1+3+9+27) = 1 + 4*40 = 161
    # Known optimal is 320. Search n from moore to ~400.
    if k == 4:
        valid_coeffs = _valid_circulant_coeffs(k, g)
        
        # Search range: from Moore bound up to a reasonable cap
        n_start = moore
        n_end = min(500, moore + 300)
        
        # Try n values in increasing order (smaller is better)
        # But also try some larger n values that might be more likely to work
        # Actually, for circulants, larger n gives more freedom for (a,b)
        # Let's try a mix: first small n, then if time, larger n
        
        # First pass: n from moore to moore+50 (most promising for small n)
        n_range_1 = range(n_start, min(n_start + 50, n_end + 1))
        result = _search_circulants(k, g, n_range_1, valid_coeffs, deadline)
        if result is not None:
            return result
        
        # Second pass: n from moore+50 to n_end
        if _time_left() > 10:
            n_range_2 = range(min(n_start + 50, n_end + 1), n_end + 1)
            result = _search_circulants(k, g, n_range_2, valid_coeffs, deadline)
            if result is not None:
                return result
        
        # Third pass: try larger n values (up to 600) with more trials
        if _time_left() > 20:
            n_range_3 = range(n_end + 1, min(n_end + 200, 600) + 1)
            result = _search_circulants(k, g, n_range_3, valid_coeffs, deadline)
            if result is not None:
                return result

    # Phase 2: Local search with random 4-regular graphs
    if _time_left() > 15:
        if k == 4 and g == 10:
            candidates = [320, 300, 280, 260, 243]
        else:
            candidates = []
            base = max(moore, k * (k - 1) + 1)
            for factor in (1.0, 1.1, 1.25, 1.5, 1.8):
                candidates.append(int(base * factor))
            candidates = sorted(set(candidates))
        
        best_result = None
        for n in candidates:
            if _time_left() < 15:
                break
            budget = min(20.0, max(3.0, _time_left() / 2.0))
            adj = _local_search(n, k, g, rng, budget)
            if adj is not None:
                if _girth(adj, n) >= g:
                    edges = _edges_from_adj(adj)
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
        
        if best_result is not None:
            return best_result

    # Phase 3: Fallback to LUW
    return _luw(k, g)
