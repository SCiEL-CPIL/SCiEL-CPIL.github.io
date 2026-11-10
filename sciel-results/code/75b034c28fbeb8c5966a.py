import time
import random
import numpy as np
from scipy import sparse

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # Specific handling for the known difficult case k=3, g=16
    if k == 3 and g == 16:
        # Try SL(2, 8) Cayley graph (n=504)
        result = _try_cayley_sl2(8, k, g, deadline)
        if result is not None:
            return result
        
        # Try SL(2, 16) Cayley graph (n=4080)
        result = _try_cayley_sl2(16, k, g, deadline)
        if result is not None:
            return result

        # Fallback to LUW
        return _luw_construct(k, g)

    # General case
    return _luw_construct(k, g)

def _try_cayley_sl2(q, k, g, deadline):
    """
    Attempts to construct a k-regular Cayley graph of SL(2, q) with girth >= g.
    For k=3, we look for 3 involutions that generate the group.
    """
    if time.time() > deadline - 10:
        return None
        
    # Generate field elements
    # For q=2^m, we can represent elements as integers and use GF(2^m) arithmetic
    # But for small q like 8, 16, we can just use integers 0..q-1 with custom add/mul
    # However, SL(2, q) requires q to be a prime power.
    # For q=8, we need GF(8).
    # For q=16, we need GF(16).
    
    # Implementing GF(2^m) arithmetic
    if q == 8:
        # GF(8) with primitive polynomial x^3 + x + 1 (0b1011 = 11)
        # Actually, let's use a standard representation.
        # Elements are 0..7.
        # Multiplication table for GF(8) with poly x^3+x+1
        # alpha^3 = alpha + 1
        # We can just precompute or use bit operations.
        def gf_add(a, b): return a ^ b
        def gf_mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                if a & 8: # x^3
                    a ^= 11 # x^3 + x + 1
            return res
        def gf_inv(a):
            if a == 0: return 0
            # In GF(8), a^7 = 1
            return gf_mul(gf_mul(gf_mul(a, a), gf_mul(gf_mul(a, a), a)), gf_mul(gf_mul(a, a), a))
    elif q == 16:
        # GF(16) with poly x^4 + x + 1 (0b10011 = 19)
        def gf_add(a, b): return a ^ b
        def gf_mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                if a & 16:
                    a ^= 19
            return res
        def gf_inv(a):
            if a == 0: return 0
            # a^15 = 1
            a2 = gf_mul(a, a)
            a4 = gf_mul(a2, a2)
            a8 = gf_mul(a4, a4)
            a15 = gf_mul(a8, gf_mul(a4, gf_mul(a2, a)))
            return a15
    else:
        # Prime q
        def gf_add(a, b): return (a + b) % q
        def gf_sub(a, b): return (a - b) % q
        def gf_mul(a, b): return (a * b) % q
        def gf_inv(a):
            if a == 0: return 0
            return pow(a, q-2, q)

    # Represent matrices as tuples (a, b, c, d)
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            gf_add(gf_mul(a1, a2), gf_mul(b1, c2)),
            gf_add(gf_mul(a1, b2), gf_mul(b1, d2)),
            gf_add(gf_mul(c1, a2), gf_mul(d1, c2)),
            gf_add(gf_mul(c1, b2), gf_mul(d1, d2))
        )

    def mat_det(m):
        a, b, c, d = m
        return gf_sub(gf_mul(a, d), gf_mul(b, c))

    # Find all involutions in SL(2, q)
    # M^2 = I => trace(M) = 0 (in char 2) or trace(M) = +/- 2 (in char != 2)
    # For q even, trace(M) = a+d = 0 => d=a.
    # det(M) = a^2 - bc = 1 => a^2 + bc = 1 (since -1=1 in char 2? No, a^2-bc=1 => a^2+bc=1 if char 2? 
    # In char 2, a^2-bc = a^2+bc. So a^2+bc=1.
    
    involutions = []
    if q % 2 == 0:
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # d = a
                    # check det = 1
                    det = gf_sub(gf_mul(a, a), gf_mul(b, c))
                    if det == 1:
                        m = (a, b, c, a)
                        # Check M != I
                        if m != (1, 0, 0, 1):
                            involutions.append(m)
    else:
        # For odd q, M^2=I => (M-I)(M+I)=0.
        # Eigenvalues 1, -1. Trace 0.
        # a+d=0 => d=-a.
        # det = -a^2 - bc = 1 => a^2 + bc = -1.
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    d = q - a if a != 0 else 0 # -a mod q
                    det = gf_sub(gf_mul(a, d), gf_mul(b, c))
                    if det == 1:
                        m = (a, b, c, d)
                        if m != (1, 0, 0, 1):
                            involutions.append(m)

    if len(involutions) < k:
        return None

    # We need k generators. For k=3, we pick 3 involutions.
    # We need them to generate the group.
    # Group size for SL(2, q) is q(q^2-1).
    target_size = q * (q*q - 1)
    
    # Try random subsets of involutions
    # Limit attempts to avoid timeout
    max_attempts = 200
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        gens = random.sample(involutions, k)
        
        # Check if they generate the group
        # BFS from identity
        identity = (1, 0, 0, 1)
        visited = {identity}
        queue = [identity]
        edges_from_gen = [] # Store edges for the Cayley graph if successful
        
        # We need to build the graph structure
        # Vertex: group element
        # Edges: (v, v*g) for g in gens
        
        # First, verify generation
        generated = False
        for _ in range(10): # Limit BFS depth/expansion for speed check? No, just run BFS
            if len(visited) == target_size:
                generated = True
                break
            if not queue:
                break
            # Simple BFS
            next_queue = []
            for v in queue:
                for g in gens:
                    w = mat_mul(v, g)
                    if w not in visited:
                        visited.add(w)
                        next_queue.append(w)
            queue = next_queue
            if time.time() > deadline - 5:
                break
        
        if not generated or len(visited) != target_size:
            continue
            
        # If generated, build the graph
        # Map group elements to indices
        index_map = {elem: i for i, elem in enumerate(visited)}
        n = len(visited)
        edges = []
        for v in visited:
            idx_v = index_map[v]
            for g in gens:
                w = mat_mul(v, g)
                idx_w = index_map[w]
                if idx_v < idx_w:
                    edges.append((idx_v, idx_w))
        
        # Check girth
        if _has_girth_at_least_fast(edges, n, k, g):
            return n, edges

    return None

def _has_girth_at_least_fast(edges, n, k, g):
    """
    Fast girth check using non-backtracking walks.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for u in range(n):
        if len(adj[u]) != k:
            return False
        if len(set(adj[u])) != k:
            return False
        if u in adj[u]:
            return False

    # Build Hashimoto matrix
    dir_edges = []
    dir_idx = {}
    for u, v in edges:
        idx1 = len(dir_edges)
        dir_edges.append((u, v))
        dir_idx[(u, v)] = idx1
        idx2 = len(dir_edges)
        dir_edges.append((v, u))
        dir_idx[(v, u)] = idx2
    
    m = len(dir_edges)
    
    row = []
    col = []
    for i, (u, v) in enumerate(dir_edges):
        for w in adj[v]:
            if w != u:
                j = dir_idx[(v, w)]
                row.append(i)
                col.append(j)
                
    B = sparse.csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(m, m))
    
    power = B.copy()
    for L in range(2, g):
        power = power @ B
        if L >= 3:
            diag = power.diagonal()
            if np.any(diag != 0):
                return False
                
    return True

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph construction.
    """
    q = _next_prime_power(k)
    d = max(3, g // 2 + 1)
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
            
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
            
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    return n, edges

def _next_prime_power(k):
    if k <= 2: return 2
    if k <= 3: return 3
    if k <= 4: return 4
    if k <= 5: return 5
    if k <= 7: return 7
    if k <= 8: return 8
    if k <= 9: return 9
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True
    p = k
    while not is_prime(p):
        p += 1
    return p
