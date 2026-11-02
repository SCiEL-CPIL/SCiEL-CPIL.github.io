"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _luw_graph(k, g, d_override=None, q_override=None):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges) if valid q-regular with girth >= g, else None."""
    q = q_override if q_override is not None else k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    elif q == 2:
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: x * y
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Build rule for LUW construction
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
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield (1 - side, tuple(y))

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
    
    n = len(order)
    return n, edges

def _verify_girth_fast(n, edges, k, g, timeout=5):
    """Fast girth verification using BFS from each vertex with early termination.
    Returns True if girth >= g, False otherwise."""
    start_time = time.time()
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check for multiple edges and self-loops
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Girth check: BFS from each vertex, track shortest cycle
    for s in range(n):
        if time.time() - start_time > timeout:
            return False  # timeout, assume invalid to be safe
        
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g - 1:
                continue  # No cycle of length < g can be formed from here
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Non-tree edge: cycle of length dist[u] + dist[v] + 1
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _cayley_field_graph(k, g):
    """Try to construct a k-regular Cayley graph on a finite field.
    
    Vertices are elements of F_q where q is a prime power >= k+1.
    We need a symmetric connection set S of size k such that
    all non-trivial sums of < g elements from S (with repetitions allowed,
    but no immediate backtracking) are non-zero.
    
    For simplicity, we try S = {a_1, ..., a_k} where the a_i are chosen
    to be powers of a primitive element, which often gives good girth.
    """
    start_time = time.time()
    
    def is_prime_power(n):
        """Check if n is a prime power, return (prime, power) or None."""
        if n < 2:
            return None
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                q = n
                while q % p == 0:
                    q //= p
                if q == 1:
                    return (p, int(n / (p ** (len(str(n)) - len(str(q))))))
                break
        # n is prime
        return (n, 1)
    
    def gf_mul(a, b, p, exp):
        """Multiply in GF(p^exp) using a fixed irreducible polynomial."""
        # For simplicity, only handle exp=1 (i.e., prime fields)
        if exp == 1:
            return (a * b) % p
        # For higher extensions, we'd need a specific irreducible poly
        # Skip for now
        return None
    
    # Try prime fields first
    for q in range(k + 1, k + 50):
        if time.time() - start_time > 30:
            break
        # Check if q is prime
        if q < 2:
            continue
        is_prime = True
        for i in range(2, int(q**0.5) + 1):
            if q % i == 0:
                is_prime = False
                break
        if not is_prime:
            continue
        
        # Try to find a symmetric connection set S of size k in F_q
        # S must be symmetric: if a in S then -a in S
        # For odd k, we need one element to be its own inverse (i.e., 0, but 0 not allowed)
        # So for odd k, we need pairs {a, -a} plus one more... but -a = a only if 2a=0
        # In a field of odd characteristic, -a = a only if a=0. So odd k is problematic.
        # Actually, we need |S| = k and S = -S. If k is odd, this is impossible in a field
        # of odd characteristic. So for odd k, we need a different approach or even characteristic.
        
        if q % 2 == 0:
            # In characteristic 2, -a = a, so any set is symmetric
            # Try S = {1, 2, 4, ..., 2^(k-1)}
            S = [1]
            for i in range(1, k):
                S.append((S[-1] * 2) % q)
            if len(set(S)) != k:
                continue
        else:
            # Odd characteristic: need pairs {a, -a}
            if k % 2 != 0:
                continue  # Can't have odd-sized symmetric set in odd char
            # Try S = {a_1, -a_1, a_2, -a_2, ...}
            S = []
            for a in range(1, q // 2):
                if len(S) >= k:
                    break
                S.append(a)
                S.append(q - a)
            if len(S) != k:
                continue
        
        # Check girth: no cycle of length < g
        # A cycle of length l corresponds to a_1 + ... + a_l = 0 where
        # a_i in S and a_{i+1} != -a_i (no backtracking)
        # For small g, we can check exhaustively
        if g <= 12:
            valid = True
            # Check all walks of length < g
            # This is expensive, so only do it for small q
            if q <= 50:
                # BFS from 0 to find shortest cycle back to 0
                # Actually, we need to check if there's any cycle of length < g
                # Use the connection set to build the graph and check girth
                adj = [[] for _ in range(q)]
                for u in range(q):
                    for a in S:
                        v = (u + a) % q
                        adj[u].append(v)
                
                # Check girth
                for s in range(q):
                    dist = [-1] * q
                    parent = [-1] * q
                    dist[s] = 0
                    queue = deque([s])
                    while queue:
                        u = queue.popleft()
                        du = dist[u]
                        if du >= g - 1:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = du + 1
                                parent[v] = u
                                queue.append(v)
                            elif v != parent[u] and u != parent[v]:
                                cycle_len = du + dist[v] + 1
                                if cycle_len < g:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Build edges
                    edges = []
                    for u in range(q):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return q, edges
    
    return None

def construct(k, g):
    """Portfolio search over explicit LUW-type constructions."""
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best = None
    best_n = float('inf')
    
    def try_candidate(n, edges, expected_degree):
        nonlocal best, best_n
        if n >= best_n:
            return
        # Quick degree check
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != expected_degree for dg in deg):
            return
        # Check for duplicate edges
        edge_set = set()
        for u, v in edges:
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return
            edge_set.add(key)
        # Verify girth
        if _verify_girth_fast(n, edges, expected_degree, g, timeout=3):
            best = (n, edges)
            best_n = n
    
    # Standard d for LUW: d = g-5 for odd g, or g-4 for even g
    d_std = g - 5 if (g - 5) % 2 == 1 else g - 4
    d_std = max(3, d_std)
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Candidate (q, d) pairs to try
    candidates = []
    # First: q = k with various d values
    for d in [d_std, d_std + 1, d_std + 2, d_std - 1]:
        if d >= 3:
            candidates.append((k, d))
    
    # Then: try q = next few primes >= k
    q = k
    while q < k + 20 and len(candidates) < 10:
        if is_prime(q) and q != k:
            for d in [d_std, d_std + 1]:
                if d >= 3:
                    candidates.append((q, d))
        q += 1
    
    # Also try q = 4 if k is small (GF(4) gives different structure)
    if k <= 4:
        candidates.append((4, d_std))
    
    # Try each candidate
    for q_val, d_val in candidates:
        if time.time() - start_time > deadline - 5:
            break
        try:
            result = _luw_graph(k, g, d_override=d_val, q_override=q_val)
            if result is not None:
                n, edges = result
                if q_val == k:
                    try_candidate(n, edges, k)
        except Exception:
            continue
    
    # Try Cayley field graph construction
    if time.time() - start_time < deadline - 10:
        try:
            cayley_result = _cayley_field_graph(k, g)
            if cayley_result is not None:
                n, edges = cayley_result
                try_candidate(n, edges, k)
        except Exception:
            pass
    
    if best is not None:
        return best
    
    # Fallback: return the standard LUW construction
    return _luw_graph(k, g)
# EVOLVE-BLOCK-END
