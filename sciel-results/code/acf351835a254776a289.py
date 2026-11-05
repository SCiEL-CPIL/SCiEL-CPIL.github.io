"""Construct a 3-regular graph of girth >= g using circulant graph search."""
import math
import sys

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    candidate = n if n % 2 else n + 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate

def _compute_girth_circulant(n, a, b):
    """Compute exact girth of circulant graph C_n(a, b) using non-backtracking BFS.
    Vertices: 0..n-1. Edges: v connected to (v+a)%n, (v-a)%n, (v+b)%n, (v-b)%n.
    But since it's undirected and 3-regular, each vertex has neighbors:
      (v+a)%n, (v-a)%n, (v+b)%n  -- wait, that's 4 neighbors.
    Actually for 3-regular: generators {a, -a, b} where -a != a mod n (n odd).
    Neighbors of v: (v+a)%n, (v-a)%n, (v+b)%n. But (v-b)%n is also a neighbor.
    Hmm, for 3-regular we need exactly 3 distinct neighbors.
    If we use generators {a, b} with a != b, a != -b, b != -a, then neighbors are:
    (v+a), (v-a), (v+b), (v-b) -- that's 4. Not 3-regular.
    
    For 3-regular circulant: we need 3 generators where some are negatives.
    Standard: generators {a, b, c} where a = -c or similar.
    Simplest: generators {a, -a, b} where a != -a (n odd) and b != ±a.
    Then neighbors of v: (v+a), (v-a), (v+b). But we also need (v-b) to be connected?
    No -- in an undirected graph, if b is a generator, then -b is also a generator.
    So generators {a, -a, b, -b} gives 4 neighbors.
    
    For 3-regular: use generators {a, b, c} where one is the negative of another.
    E.g., {a, -a, b} but then -b must also be in the set for undirected, so {a, -a, b, -b} = 4.
    
    Alternative: {a, b, c} where c = -(a+b) or something. Or use {a, b, -a-b}.
    Then neighbors: v+a, v+b, v-a-b. And by symmetry, v-a, v-b, v+a+b are also neighbors
    of other vertices. This is 3-regular if a, b, a+b are all distinct and nonzero mod n.
    """
    # For 3-regular circulant: generators {a, b, c} with a+b+c = 0 mod n
    # Neighbors of v: (v+a)%n, (v+b)%n, (v+c)%n where c = (-a-b)%n
    # This is 3-regular if a, b, c are distinct and nonzero.
    
    c = (-a - b) % n
    if a == 0 or b == 0 or c == 0:
        return 0
    if len({a, b, c}) != 3:
        return 0
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for v in range(n):
        adj[v] = [(v + a) % n, (v + b) % n, (v + c) % n]
    
    # Non-backtracking BFS from each directed edge
    # State: (current_vertex, prev_vertex, depth)
    # Start: for each directed edge (u, v), state (v, u, 1)
    # If we reach (u, v, d) where d >= 3, we found a cycle of length d
    
    # To be efficient, we'll do BFS and track visited states
    # State is (current, prev) -- the last two vertices in the walk
    # We want the shortest cycle, so BFS by depth
    
    from collections import deque
    
    best_girth = float('inf')
    
    # For each starting directed edge
    for start_u in range(n):
        for start_v in adj[start_u]:
            # BFS from (start_v, start_u) at depth 1
            visited = set()
            queue = deque()
            queue.append((start_v, start_u, 1))
            visited.add((start_v, start_u))
            
            found = False
            while queue:
                curr, prev, depth = queue.popleft()
                
                # Check if we've returned to start_u
                if curr == start_u and depth >= 3:
                    best_girth = min(best_girth, depth)
                    found = True
                    break
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # no backtracking
                    state = (nxt, curr)
                    if state not in visited:
                        visited.add(state)
                        if depth + 1 >= best_girth:
                            continue
                        queue.append((nxt, curr, depth + 1))
            
            if found:
                # We can break early if we've found a small girth
                if best_girth <= 4:
                    break
        if best_girth <= 4:
            break
    
    return best_girth if best_girth != float('inf') else n  # fallback

def construct(k, g):
    """Construct a 3-regular graph with girth >= g."""
    assert k == 3, "This implementation only handles k=3"
    
    # Search over primes n and generator pairs (a, b)
    # n must be odd (for 3-regular circulant with a+b+c=0)
    # Lower bound on n from Moore bound: n >= 2*g^2 - 2*g + 2 for 3-regular
    # Actually Moore bound for 3-regular girth g: n >= 2*(3-1)^(g/2-1) for even g
    # For g=16: n >= 2*2^7 = 256 (rough)
    
    moore_bound = 2 * (2 ** (g // 2 - 1)) if g % 2 == 0 else 2 * (2 ** ((g - 1) // 2))
    
    best_n = None
    best_edges = None
    
    # Try primes starting from a reasonable lower bound
    # The known best is 936, so we should try to get close to that or better
    # Start from around 200-300 and go up
    
    n_start = max(moore_bound + 1, 101)
    
    # Try several primes
    n = _next_prime(n_start)
    max_primes_to_try = 200
    
    for trial in range(max_primes_to_try):
        if not _is_prime(n):
            n = _next_prime(n)
            continue
        
        # Try several generator pairs
        # Heuristic: a small, b around n/3 or n/4
        candidates = []
        for a in range(1, min(20, n // 4)):
            for b in range(a + 1, min(a + 50, n // 2)):
                c = (-a - b) % n
                if c in (0, a, b) or c <= b:  # avoid duplicates and zero
                    continue
                candidates.append((a, b))
        
        # Also try some structured candidates
        for a in [1, 2, 3]:
            for frac in [3, 4, 5, 7]:
                b = n // frac
                if b > a and b < n // 2:
                    c = (-a - b) % n
                    if c not in (0, a, b) and c > b:
                        candidates.append((a, b))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        found_for_n = False
        for a, b in candidates:
            girth = _compute_girth_circulant(n, a, b)
            if girth >= g:
                # Build edges
                c = (-a - b) % n
                edges = []
                for v in range(n):
                    for w in [(v + a) % n, (v + b) % n, (v + c) % n]:
                        if v < w:
                            edges.append((v, w))
                
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                
                found_for_n = True
                break
        
        if found_for_n:
            # We found a valid graph for this n; since we're going up in n,
            # this is the smallest for this prime. Check a few more primes to be safe.
            # Actually, since we want the smallest n, and we're trying primes in order,
            # the first prime that works gives us a candidate. But a larger prime might
            # have a smaller n? No, n is the number of vertices.
            # So we should keep going until we're confident we have the smallest.
            # For now, break after finding one and verify.
            pass
        
        n = _next_prime(n + 1)
        
        # Stop if we've gone too far
        if n > 2000:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the original Lazebnik-Ustimenko-Woldar construction
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
