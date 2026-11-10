import time
import random
import numpy as np
from scipy import sparse

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Primary: Cayley graph on PSL(2,q) with algebraic girth guarantee.
    Fallback: LUW construction.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # Try Cayley graph construction first
    try:
        result = _cayley_psl2_construct(k, g)
        if result is not None:
            n, edges = result
            # Verify girth
            if _has_girth_at_least_fast(edges, n, g):
                return n, edges
    except Exception:
        pass

    # Fallback: LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        return _luw_construct(k, g)

def _cayley_psl2_construct(k, g):
    """
    Construct a k-regular Cayley graph on PSL(2, q) where q is a power of 2.
    
    PSL(2, q) has order q*(q^2-1)/gcd(2, q-1) = q*(q^2-1) for even q.
    For q=2^m, order = q*(q^2-1) = q^3 - q.
    
    We need a symmetric generating set S of size k (no identity, no involutions)
    such that no product of fewer than g elements from S equals identity.
    
    For k=3, g=16: try q=16 (order 4080) or q=32 (order 32736).
    """
    if k != 3:
        return None
    
    # For k=3, g=16, try specific q values
    # PSL(2, 16) has order 16*(256-1) = 4080
    # PSL(2, 32) has order 32*(1024-1) = 32736
    # PSL(2, 8) has order 8*(64-1) = 504
    
    q_candidates = [16, 32, 64]
    
    for q in q_candidates:
        if time.time() > deadline_global - 10:
            break
        try:
            result = _build_cayley_psl2(q, k, g)
            if result is not None:
                return result
        except Exception:
            continue
    
    return None

def _build_cayley_psl2(q, k, g):
    """
    Build Cayley graph on PSL(2,q) with k generators.
    For q = 2^m, elements are 2x2 matrices over GF(q) with det=1, modulo ±I.
    """
    m = q.bit_length() - 1  # q = 2^m
    
    # GF(2^m) arithmetic using a primitive polynomial
    # For m=4 (q=16): x^4 + x + 1
    # For m=5 (q=32): x^5 + x^3 + 1
    # For m=6 (q=64): x^6 + x + 1
    prim_poly = {
        4: 0b10011,  # x^4 + x + 1
        5: 0b101001, # x^5 + x^3 + 1
        6: 0b1000011, # x^6 + x + 1
        7: 0b10001001, # x^7 + x^4 + 1
        8: 0b100011101, # x^8 + x^4 + x^3 + x^2 + 1
    }.get(m, None)
    
    if prim_poly is None:
        return None
    
    # Precompute GF(q) tables
    gf_exp = [1] * (2 * q)
    gf_log = [0] * q
    gf_log[1] = 0
    x = 1
    for i in range(1, 2 * q):
        gf_exp[i] = x
        gf_log[x] = i
        # x = x * alpha mod prim_poly
        x <<= 1
        if x & (1 << m):
            x ^= prim_poly
    
    def gf_mul(a, b):
        if a == 0 or b == 0:
            return 0
        return gf_exp[(gf_log[a] + gf_log[b]) % (q - 1)]
    
    def gf_inv(a):
        if a == 0:
            return 0
        return gf_exp[(q - 1) - gf_log[a]]
    
    # Matrix operations over GF(q), 2x2 matrices represented as (a, b, c, d)
    # [[a, b], [c, d]]
    def mat_mul(M1, M2):
        a1, b1, c1, d1 = M1
        a2, b2, c2, d2 = M2
        return (gf_mul(a1, a2) ^ gf_mul(b1, c2),
                gf_mul(a1, b2) ^ gf_mul(b1, d2),
                gf_mul(c1, a2) ^ gf_mul(d1, c2),
                gf_mul(c1, b2) ^ gf_mul(d1, d2))
    
    def mat_det(M):
        a, b, c, d = M
        return gf_mul(a, d) ^ gf_mul(b, c)
    
    def mat_inv(M):
        a, b, c, d = M
        det = mat_det(M)
        if det == 0:
            return None
        inv_det = gf_inv(det)
        return (gf_mul(d, inv_det), gf_mul(b ^ b, inv_det),  # -b = b in GF(2)
                gf_mul(c ^ c, inv_det), gf_mul(a, inv_det))
    
    # Identity matrix
    I = (1, 0, 0, 1)
    
    # Generate candidate generators: matrices in PSL(2,q) that are not identity
    # and have no small order
    # For k=3, we need 3 involutions (or pairs of inverse elements)
    # In GF(2), -1 = 1, so PSL(2,q) = SL(2,q) for even q
    # A matrix has order 2 iff M^2 = I iff trace(M) = 0 (in GF(2), trace = a+d)
    
    # Find involutions: matrices with det=1 and trace=0
    # trace = a ^ d = 0, so d = a
    # det = a^2 ^ b*c = 1, so b*c = a^2 ^ 1
    # We need b, c != 0 (otherwise matrix is identity or not in SL)
    
    involutions = []
    for a in range(q):
        if a == 0:
            continue
        bc = gf_mul(a, a) ^ 1
        if bc == 0:
            continue
        for b in range(1, q):
            c = gf_mul(bc, gf_inv(b))
            if c == 0:
                continue
            M = (a, b, c, a)
            # Check det
            if mat_det(M) != 1:
                continue
            # Check it's not identity
            if M == I:
                continue
            # Check it's truly an involution: M^2 = I
            M2 = mat_mul(M, M)
            if M2 != I:
                continue
            # Avoid duplicates: in PSL(2,q) with even q, M and -M are same, but -M=M in GF(2)
            # So each involution is unique
            involutions.append(M)
    
    if len(involutions) < k:
        return None
    
    # For k=3, pick 3 involutions that generate a "free" product for girth g
    # Strategy: pick involutions from different "families" to maximize girth
    # Group involutions by their (b, c) pattern or some invariant
    
    # For the Cayley graph, generators are the involutions themselves
    # (since they are self-inverse)
    
    # Try random selections of k involutions and check girth
    best_result = None
    
    # Sort involutions to have deterministic ordering
    involutions.sort(key=lambda M: (M[0], M[1], M[2], M[3]))
    
    num_involutions = len(involutions)
    
    # Try several random selections
    max_tries = 50
    for trial in range(max_tries):
        if time.time() > deadline_global - 15:
            break
        
        # Pick k random distinct involutions
        indices = random.sample(range(num_involutions), k)
        generators = [involutions[i] for i in indices]
        
        # Build the Cayley graph
        # Vertices are elements of PSL(2,q) = SL(2,q) (since -I = I in GF(2))
        # Order = q*(q^2-1)
        # Enumerate all elements: all 2x2 matrices with det=1
        # a, b, c, d with a*d ^ b*c = 1
        # For each a, b, c: d = (1 ^ b*c) / a if a != 0, else c=0 and d arbitrary with b=0... 
        # Actually: if a=0, then b*c=1, so b,c != 0, d is free? No: det = 0*d - b*c = b*c = 1.
        # So d is free when a=0. That gives q choices for d.
        # If a != 0: d = (1 ^ b*c) / a. So for each a!=0, b, c: one d.
        # Total: (q-1)*q*q + q*q = q^2*(q-1) + q^2 = q^3 - q^2 + q^2 = q^3 - q. Correct.
        
        elements = []
        elem_index = {}
        
        # a = 0 case: b*c = 1, d free
        for b in range(1, q):
            c = gf_inv(b)
            for d in range(q):
                M = (0, b, c, d)
                elements.append(M)
                elem_index[M] = len(elements) - 1
        
        # a != 0 case
        for a in range(1, q):
            for b in range(q):
                for c in range(q):
                    bc = gf_mul(b, c)
                    d = gf_mul(1 ^ bc, gf_inv(a))
                    M = (a, b, c, d)
                    # Verify det
                    if mat_det(M) != 1:
                        continue
                    elements.append(M)
                    elem_index[M] = len(elements) - 1
        
        n = len(elements)
        
        # Build edges: for each vertex, connect to vertex * generator
        edges_set = set()
        for i, v in enumerate(elements):
            for gen in generators:
                w = mat_mul(v, gen)
                if w not in elem_index:
                    continue
                j = elem_index[w]
                if i < j:
                    edges_set.add((i, j))
        
        edges = list(edges_set)
        
        # Check degree regularity
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        if any(d != k for d in deg):
            continue
        
        # Check girth
        if _has_girth_at_least_fast(edges, n, g):
            return n, edges
    
    return None

def _random_regular_graph(n, k):
    """
    Generates a random k-regular graph with n vertices using the Configuration Model.
    Returns a list of edges (i, j) or None if self-loops/multi-edges occur.
    """
    if n * k < 4:
        return None
    if (n * k) % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    # Pair stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return None
        edges.add((u, v))
    
    return list(edges)

def _has_girth_at_least_fast(edges, n, g):
    """
    Fast girth check using non-backtracking walk matrix powers.
    """
    if n == 0:
        return True

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Build directed edges for non-backtracking
    dir_edges = []
    dir_index = {}
    for u, v in edges:
        idx1 = len(dir_edges)
        dir_edges.append((u, v))
        dir_index[(u, v)] = idx1
        idx2 = len(dir_edges)
        dir_edges.append((v, u))
        dir_index[(v, u)] = idx2

    m = len(dir_edges)  # = n * k

    # Build Hashimoto matrix B (sparse)
    row = []
    col = []
    for i, (u, v) in enumerate(dir_edges):
        for w in adj[v]:
            if w == u:
                continue
            j = dir_index[(v, w)]
            row.append(i)
            col.append(j)

    B = sparse.csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(m, m))

    # Check if diag(B^L) has any nonzero for L = 3, 4, ..., g-1
    power = B.copy()  # B^1
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
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
        
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

# Global deadline for time management
deadline_global = time.time() + 115.0

def _random_regular_graph(n, k):
    """
    Generates a random k-regular graph with n vertices using the Configuration Model.
    Returns a list of edges (i, j) or None if self-loops/multi-edges occur.
    """
    if n * k < 4:
        return None
    if (n * k) % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    # Pair stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return None
        edges.add((u, v))
    
    return list(edges)

def _has_girth_at_least_fast(edges, n, g):
    """
    Fast girth check using BFS from each vertex.
    For k-regular graph, check if any cycle shorter than g exists.
    Uses BFS with parent tracking to detect backtracking-free cycles.
    """
    if n == 0:
        return True

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Verify regularity and simplicity
    for u in range(n):
        if len(adj[u]) != k_global:
            return False
        if u in adj[u]:
            return False
        if len(adj[u]) != len(set(adj[u])):
            return False

    # For each vertex, do BFS up to depth (g-1)//2
    # If we find a cycle of length < g, return False
    max_depth = (g - 1) // 2  # For even g, cycles of length g have radius g/2
    
    # Use a more efficient approach: check for cycles using non-backtracking walks
    # For k=3, we can use a faster method
    
    # Build directed edges for non-backtracking
    dir_edges = []
    dir_index = {}
    for u, v in edges:
        idx1 = len(dir_edges)
        dir_edges.append((u, v))
        dir_index[(u, v)] = idx1
        idx2 = len(dir_edges)
        dir_edges.append((v, u))
        dir_index[(v, u)] = idx2

    m = len(dir_edges)  # = n * k

    # Build Hashimoto matrix B (sparse)
    row = []
    col = []
    for i, (u, v) in enumerate(dir_edges):
        for w in adj[v]:
            if w == u:
                continue
            j = dir_index[(v, w)]
            row.append(i)
            col.append(j)

    B = sparse.csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(m, m))

    # Check if diag(B^L) has any nonzero for L = 3, 4, ..., g-1
    power = B.copy()  # B^1
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
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
        
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

# Global variable for k used in _has_girth_at_least_fast
k_global = 3
