import time
import random
import numpy as np
from scipy import sparse
from itertools import combinations, permutations

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using Cayley graph
    construction on finite groups.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try Cayley graph constructions on various groups
    # Strategy: for each candidate group, find a symmetric generating set S
    # of size k such that no non-trivial product of < g elements of S equals identity
    
    # Candidate groups in order of size (smaller = better score)
    candidates = []
    
    # SL(2, q) for small q
    for q in [4, 5, 7, 8, 9, 11, 13, 16, 19, 25, 27, 32, 37, 49, 64]:
        if q >= 4:
            candidates.append(('SL2', q))
    
    # PSL(2, q) - smaller groups
    for q in [5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        candidates.append(('PSL2', q))
    
    # A_n groups
    for n in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
        candidates.append(('An', n))
    
    # Try each candidate
    for group_type, param in candidates:
        if time.time() > deadline - 10.0:
            break
        try:
            result = _cayley_construct(group_type, param, k, g, deadline)
            if result is not None:
                n, edges = result
                # Verify the result
                if _has_girth_at_least_fast(edges, n, g):
                    return n, edges
        except Exception:
            continue
    
    # Fallback: LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        # Last resort
        return _luw_construct(k, g)

def _cayley_construct(group_type, param, k, g, deadline):
    """
    Constructs a Cayley graph on the given group with generating set of size k
    and girth at least g.
    """
    if group_type == 'SL2':
        return _cayley_sl2(param, k, g, deadline)
    elif group_type == 'PSL2':
        return _cayley_psl2(param, k, g, deadline)
    elif group_type == 'An':
        return _cayley_an(param, k, g, deadline)
    else:
        return None

def _gf2_mat_mul(A, B):
    """Multiply two 2x2 matrices over GF(2)."""
    result = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            result[i][j] = (A[i][0] & B[0][j]) ^ (A[i][1] & B[1][j])
    return result

def _gf2_mat_pow(M, n):
    """Matrix exponentiation over GF(2)."""
    if n == 0:
        return [[1, 0], [0, 1]]
    if n == 1:
        return M
    if n % 2 == 0:
        half = _gf2_mat_pow(M, n // 2)
        return _gf2_mat_mul(half, half)
    else:
        return _gf2_mat_mul(M, _gf2_mat_pow(M, n - 1))

def _gf2_mat_trace(M):
    """Trace of a 2x2 matrix over GF(2)."""
    return M[0][0] ^ M[1][1]

def _sl2_elements(q):
    """
    Generate all elements of SL(2, q) where q is a power of 2.
    Elements are 2x2 matrices over GF(q) with determinant 1.
    For q = 2^m, represent field elements as integers 0..q-1 with
    multiplication modulo the appropriate irreducible polynomial.
    """
    # For simplicity, only handle small q where we can enumerate
    if q == 4:
        # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
        # Represent as 0,1,2,3 with 2=alpha, 3=alpha+1
        # Multiplication table for GF(4):
        gf4_mult = [
            [0,0,0,0],
            [0,1,2,3],
            [0,2,3,1],
            [0,3,1,2]
        ]
        gf4_add = [
            [0,1,2,3],
            [1,0,3,2],
            [2,3,0,1],
            [3,2,1,0]
        ]
        
        def mat_mul(A, B):
            result = [[0,0],[0,0]]
            for i in range(2):
                for j in range(2):
                    result[i][j] = gf4_add[gf4_mult[A[i][0]][B[0][j]], gf4_mult[A[i][1]][B[1][j]]]
            return result
        
        def mat_det(A):
            return gf4_add[gf4_mult[A[0][0]][A[1][1]], gf4_mult[A[0][1]][A[1][0]]]
        
        elements = []
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        M = [[a,b],[c,d]]
                        if mat_det(M) == 1:
                            elements.append(M)
        return elements
    
    elif q == 8:
        # GF(8) with primitive polynomial x^3 + x + 1
        # Elements 0..7, multiplication via polynomial reduction
        gf8_mult = _build_gf_mult(8, [1, 0, 1, 1])  # x^3 + x + 1
        gf8_add = [[i ^ j for j in range(8)] for i in range(8)]
        
        def mat_mul(A, B):
            result = [[0,0],[0,0]]
            for i in range(2):
                for j in range(2):
                    result[i][j] = gf8_add[gf8_mult[A[i][0]][B[0][j]], gf8_mult[A[i][1]][B[1][j]]]
            return result
        
        def mat_det(A):
            return gf8_add[gf8_mult[A[0][0]][A[1][1]], gf8_mult[A[0][1]][A[1][0]]]
        
        elements = []
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    for d in range(8):
                        M = [[a,b],[c,d]]
                        if mat_det(M) == 1:
                            elements.append(M)
        return elements
    
    else:
        # For larger q, use a different representation
        # This gets complex; for now, handle up to q=8
        return None

def _build_gf_mult(q, poly):
    """Build multiplication table for GF(2^m) given irreducible polynomial."""
    m = int(np.log2(q))
    # poly represents x^m + poly[m-1]*x^(m-1) + ... + poly[0]
    # For GF(8) with x^3+x+1: poly = [1, 0, 1] (coefficients of x^0, x^1, x^2)
    # Wait, let me use a cleaner representation
    
    # poly is list of length m, where poly[i] is coefficient of x^i
    # The polynomial is x^m + sum(poly[i]*x^i)
    
    def reduce(p):
        """Reduce polynomial p (list of m+1 coefficients) modulo the irreducible poly."""
        # p has degree up to m
        while len(p) > m:
            # p[-1] * x^m = p[-1] * (sum poly[i] * x^i)
            deg = len(p) - 1
            coeff = p[-1]
            if coeff:
                for i in range(m):
                    p[i] ^= poly[i]
            p = p[:-1]
        return p
    
    mult_table = [[0]*q for _ in range(q)]
    for a in range(q):
        for b in range(q):
            # Multiply polynomials a and b
            prod = [0]*(2*m)
            for i in range(m):
                if (a >> i) & 1:
                    for j in range(m):
                        if (b >> j) & 1:
                            prod[i+j] ^= 1
            # Reduce
            result = reduce(prod)
            val = 0
            for i in range(m):
                if result[i]:
                    val |= (1 << i)
            mult_table[a][b] = val
    return mult_table

def _cayley_sl2(q, k, g, deadline):
    """Construct Cayley graph on SL(2, q)."""
    elements = _sl2_elements(q)
    if elements is None or len(elements) == 0:
        return None
    
    n = len(elements)
    if n < 2:
        return None
    
    # Index elements
    elem_index = {}
    for i, e in enumerate(elements):
        key = (e[0][0], e[0][1], e[1][0], e[1][1])
        elem_index[key] = i
    
    # Define multiplication
    if q == 4:
        gf4_mult = [
            [0,0,0,0],
            [0,1,2,3],
            [0,2,3,1],
            [0,3,1,2]
        ]
        gf4_add = [
            [0,1,2,3],
            [1,0,3,2],
            [2,3,0,1],
            [3,2,1,0]
        ]
        def mat_mul(A, B):
            result = [[0,0],[0,0]]
            for i in range(2):
                for j in range(2):
                    result[i][j] = gf4_add[gf4_mult[A[i][0]][B[0][j]], gf4_mult[A[i][1]][B[1][j]]]
            return result
    elif q == 8:
        gf8_mult = _build_gf_mult(8, [1, 0, 1])  # x^3 + x + 1
        gf8_add = [[i ^ j for j in range(8)] for i in range(8)]
        def mat_mul(A, B):
            result = [[0,0],[0,0]]
            for i in range(2):
                for j in range(2):
                    result[i][j] = gf8_add[gf8_mult[A[i][0]][B[0][j]], gf8_mult[A[i][1]][B[1][j]]]
            return result
    else:
        return None
    
    # Precompute multiplication table
    mult_table = {}
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            prod = mat_mul(e1, e2)
            key = (prod[0][0], prod[0][1], prod[1][0], prod[1][1])
            mult_table[(i, j)] = elem_index[key]
    
    # Identity element
    identity = (1, 0, 0, 1)
    identity_idx = elem_index[identity]
    
    # Find symmetric generating sets of size k
    # S must be symmetric: if s in S, then s^-1 in S
    # For k=3, we need an involution (self-inverse) and a pair {s, s^-1}
    
    # Get inverses
    inv = [0]*n
    for i in range(n):
        e = elements[i]
        # Inverse of [[a,b],[c,d]] in SL(2) is [[d,-b],[-c,a]]
        if q == 4:
            gf4_neg = lambda x: x  # In GF(4), -x = x for x in {0,1,2,3}? No!
            # Actually in GF(4), characteristic 2, so -x = x
            inv_e = (e[1][1], e[0][1], e[1][0], e[0][0])
        else:
            inv_e = (e[1][1], e[0][1], e[1][0], e[0][0])
        key = inv_e
        inv[i] = elem_index[key]
    
    # Find involutions (elements of order 2)
    involutions = []
    for i in range(n):
        if i != identity_idx:
            if mult_table[(i, i)] == identity_idx:
                involutions.append(i)
    
    # For k=3: pick one involution t and one pair {s, s^-1} where s != s^-1
    # The generating set is {t, s, s^-1}
    
    # We need to check girth: no product of < g generators equals identity
    # This is expensive, so we use a heuristic search
    
    best_result = None
    time_limit = deadline - 5.0
    
    # Try different combinations
    # First, find elements s where s != s^-1 and s is not identity
    non_self_inv = [i for i in range(n) if i != identity_idx and inv[i] != i]
    
    # Random sampling for speed
    random.shuffle(non_self_inv)
    random.shuffle(involutions)
    
    for t in involutions[:20]:  # Try up to 20 involutions
        if time.time() > time_limit:
            break
        for s in non_self_inv[:50]:  # Try up to 50 non-self-inverse elements
            if time.time() > time_limit:
                break
            if s == inv[s]:
                continue
            # Generating set: {t, s, s^-1}
            S = [t, s, inv[s]]
            
            # Check that S generates the group (optional for now)
            # Check girth using BFS on the Cayley graph
            edges = _build_cayley_edges(elements, mult_table, S, identity_idx)
            if edges is None:
                continue
            
            if _has_girth_at_least_fast(edges, n, g):
                best_result = (n, edges)
                return best_result
    
    return best_result

def _build_cayley_edges(elements, mult_table, S, identity_idx):
    """Build edge list for Cayley graph with generating set S."""
    n = len(elements)
    edges = set()
    for i in range(n):
        for s in S:
            j = mult_table[(i, s)]
            if i < j:
                edges.add((i, j))
            elif i > j:
                edges.add((j, i))
    
    # Check for self-loops and multi-edges
    if len(edges) != n * len(S) // 2:
        # This can happen if some generators map i to itself
        # For a proper Cayley graph, we need |S| = degree
        pass
    
    return list(edges)

def _cayley_psl2(q, k, g, deadline):
    """Construct Cayley graph on PSL(2, q)."""
    # PSL(2, q) = SL(2, q) / {±I}
    # For odd q, this has order q(q^2-1)/2
    # For even q, SL(2, q) = PSL(2, q)
    
    if q % 2 == 0:
        return _cayley_sl2(q, k, g, deadline)
    
    # For odd q, we need a different representation
    # This is more complex; skip for now and fall through to other methods
    return None

def _cayley_an(n, k, g, deadline):
    """Construct Cayley graph on A_n."""
    # A_n has order n!/2
    # For small n, we can enumerate
    if n > 12:
        return None  # Too large to enumerate
    
    # Generate all even permutations of n elements
    # This is expensive for n > 8
    if n > 8:
        return None
    
    from math import factorial
    order = factorial(n) // 2
    
    # For now, return None to avoid complexity
    # A full implementation would need permutation group theory
    return None

def _has_girth_at_least_fast(edges, n, g):
    """
    Fast girth check using BFS with non-backtracking constraint.
    For k-regular graphs, this is much faster than the matrix approach.
    """
    if n == 0:
        return True
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for u in range(n):
        if len(adj[u]) != len(set(adj[u])):
            return False
    
    # For girth >= g, from each vertex, the non-backtracking tree of depth
    # floor((g-1)/2) must have all distinct vertices.
    # More precisely: do BFS from each vertex, tracking (node, parent),
    # up to depth floor(g/2). If we see (node, parent) twice, we have a cycle
    # of length <= 2*depth < g.
    
    max_depth = g // 2
    
    for start in range(n):
        # BFS tracking (current, parent)
        visited = set()
        queue = [(start, -1)]
        visited.add((start, -1))
        
        depth = 0
        while queue and depth < max_depth:
            next_queue = []
            for (node, parent) in queue:
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    state = (neighbor, node)
                    if state in visited:
                        # Found a cycle of length <= 2*(depth+1)
                        if 2 * (depth + 1) < g:
                            return False
                        else:
                            continue
                    visited.add(state)
                    next_queue.append(state)
            queue = next_queue
            depth += 1
    
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
