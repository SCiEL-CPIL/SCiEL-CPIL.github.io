"""Cayley graph on SL(2,q) for the (3,16) cage problem."""
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = None
    best_edges = None
    
    # Try SL(2,q) for various q
    # Order of SL(2,q) = q(q^2-1)
    # q=16: 16*255=4080
    # q=32: 32*1023=32736
    # q=9: 9*80=720
    # q=25: 25*624=15600
    
    for q in [16, 32, 9, 25]:
        if time.time() > deadline - 25:
            break
        try:
            result = _try_sl2(q, k, g, deadline)
            if result is not None:
                n, edges = result
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            continue
    
    if best_n is not None:
        return best_n, best_edges
    
    return _luw_fallback(k, g)


def _try_sl2(q, k, g, deadline):
    """Construct Cayley graph on SL(2,q)."""
    if q == 4:
        return _try_sl2_gf4(k, g, deadline)
    elif q == 9:
        return _try_sl2_gf9(k, g, deadline)
    elif q == 16:
        return _try_sl2_gf16(k, g, deadline)
    elif q == 25:
        return _try_sl2_gf25(k, g, deadline)
    elif q == 32:
        return _try_sl2_gf32(k, g, deadline)
    else:
        return _try_sl2_prime(q, k, g, deadline)


def _try_sl2_gf16(k, g, deadline):
    """SL(2,16), order 4080."""
    # GF(16) = GF(2)[x]/(x^4+x+1)
    # Elements are polynomials of degree < 4 over GF(2)
    # Represent as 4-bit integers
    # Primitive polynomial: x^4 + x + 1
    
    # Precompute GF(16) tables
    # alpha = x (the root of the primitive polynomial)
    # alpha^4 = alpha + 1
    
    # Build log/exp tables for GF(16)
    exp_table = [0] * 16  # exp_table[i] = alpha^i
    log_table = [0] * 16  # log_table[alpha^i] = i
    
    exp_table[0] = 1
    log_table[1] = 0
    for i in range(1, 15):
        # alpha^i = alpha^(i-1) * alpha
        # Multiplication by alpha: shift left, if bit 4 is set, XOR with 0x11 (x^4+x+1 -> x^4 = x+1)
        val = exp_table[i-1] << 1
        if val >= 16:
            val ^= 0x11  # x^4 + x + 1 = 10011 in binary
        exp_table[i] = val
        log_table[val] = i
    
    def gf_add(a, b):
        return a ^ b
    
    def gf_mul(a, b):
        if a == 0 or b == 0:
            return 0
        return exp_table[(log_table[a] + log_table[b]) % 15]
    
    def gf_inv(a):
        if a == 0:
            return 0
        return exp_table[(15 - log_table[a]) % 15]
    
    # Generate SL(2,16)
    elements = []
    elem_to_idx = {}
    I = (1, 0, 0, 1)
    
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
                    det = gf_add(gf_mul(a, d), gf_mul(b, c))  # ad - bc = ad + bc in GF(2)
                    if det == 1:
                        elem = (a, b, c, d)
                        if elem not in elem_to_idx:
                            elem_to_idx[elem] = len(elements)
                            elements.append(elem)
    
    n_elem = len(elements)
    if n_elem != 4080:
        return None
    
    def multiply(x, y):
        a, b, c, d = x
        e, f, g2, h = y
        return (
            gf_add(gf_mul(a, e), gf_mul(b, g2)),
            gf_add(gf_mul(a, f), gf_mul(b, h)),
            gf_add(gf_mul(c, e), gf_mul(d, g2)),
            gf_add(gf_mul(c, f), gf_mul(d, h))
        )
    
    # Find order 2 elements (trace = 0, i.e., a+d = 0, so a=d)
    order2 = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add(a, d)
        if tr == 0 and elem != I:
            order2.append(elem)
    
    # Candidates for non-order-2 generator
    candidates = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add(a, d)
        if elem != I and tr != 0:
            candidates.append(elem)
    
    import random
    random.seed(42)
    
    # Try several generator pairs
    tried = set()
    best_result = None
    
    n_tries = 200
    for _ in range(n_tries):
        if time.time() > deadline - 30:
            break
        
        s = random.choice(order2)
        t = random.choice(candidates)
        
        key = (s, t)
        if key in tried:
            continue
        tried.add(key)
        
        # t inverse
        a, b, c, d = t
        # inv = [[d, -b], [-c, a]] = [[d, b], [c, a]] in GF(2)
        t_inv = (d, b, c, a)
        
        # Check that s, t, t_inv are distinct and not identity
        if s == t or s == t_inv or t == t_inv:
            continue
        
        # Build adjacency
        adj = [set() for _ in range(n_elem)]
        
        for i, v in enumerate(elements):
            vs = multiply(v, s)
            vt = multiply(v, t)
            vti = multiply(v, t_inv)
            
            j_s = elem_to_idx.get(vs)
            j_t = elem_to_idx.get(vt)
            j_ti = elem_to_idx.get(vti)
            
            if j_s is not None and j_s != i:
                adj[i].add(j_s)
            if j_t is not None and j_t != i:
                adj[i].add(j_t)
            if j_ti is not None and j_ti != i:
                adj[i].add(j_ti)
        
        # Check degree
        valid = True
        for i in range(n_elem):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        # Check girth efficiently
        min_cycle = _check_girth_fast(adj, n_elem, g)
        
        if min_cycle >= g:
            edges = []
            for i in range(n_elem):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if best_result is None or n_elem < best_result[0]:
                best_result = (n_elem, edges)
            
            # Found a valid one, return it
            return best_result
    
    return best_result


def _check_girth_fast(adj, n, g):
    """Check if girth >= g using non-backtracking BFS.
    Returns shortest cycle length or g if none found shorter."""
    # For each vertex, do non-backtracking BFS up to depth g-1
    # Use a global visited array with timestamps for efficiency
    
    # State: (vertex, prev_vertex)
    # We'll do BFS from each vertex
    
    # Optimization: use arrays instead of sets
    # visited[prev][v] = depth, or use a 2D array
    
    # For n=4080, a 2D array of size n*n = 16M entries is too much
    # Use a dict per BFS or a flat array
    
    # Alternative: since degree is 3, the non-backtracking tree has at most 2 branches
    # at each level. Total states up to depth d: 1 + 2 + 2*2 + ... + 2^d = 2^(d+1) - 1
    # For d=15: 65535 states. That's manageable.
    
    for s in range(n):
        # BFS from s
        # Queue of (vertex, prev, depth)
        # Visited: set of (vertex, prev)
        visited = set()
        queue = deque()
        
        for u in adj[s]:
            visited.add((u, s))
            queue.append((u, s, 1))
        
        while queue:
            v, prev, depth = queue.popleft()
            
            if depth >= g - 1:
                continue
            
            for w in adj[v]:
                if w == prev:
                    continue
                
                if w == s:
                    return depth + 1
                
                if (w, v) not in visited:
                    visited.add((w, v))
                    queue.append((w, v, depth + 1))
    
    return g


def _try_sl2_gf9(k, g, deadline):
    """SL(2,9) = A6, order 720."""
    # GF(9) = GF(3)[x]/(x^2+1) since x^2+1 is irreducible over GF(3)
    # alpha^2 = -1 = 2 in GF(3)
    # Elements: a + b*alpha, a,b in {0,1,2}
    # Represent as (a, b)
    
    def gf_add(p, q_):
        return ((p[0] + q_[0]) % 3, (p[1] + q_[1]) % 3)
    
    def gf_mul(p, q_):
        a, b = p
        c, d = q_
        # (a + b*alpha)(c + d*alpha) = ac + (ad+bc)*alpha + bd*alpha^2
        # = (ac + 2*bd) + (ad+bc)*alpha
        return ((a*c + 2*b*d) % 3, (a*d + b*c) % 3)
    
    def gf_inv(p):
        a, b = p
        # inv = (a - b*alpha) / (a^2 + b^2) = (a - b*alpha) / (a^2 + 2*b^2)
        denom = (a*a + 2*b*b) % 3
        if denom == 0:
            return (0, 0)
        # In GF(3), inv(1)=1, inv(2)=2
        inv_denom = denom  # since 1^-1=1, 2^-1=2 in GF(3)
        return ((a * inv_denom) % 3, (3 - b * inv_denom) % 3)
    
    # Generate SL(2,9)
    elements = []
    elem_to_idx = {}
    I = ((1, 0), (0, 0), (0, 0), (1, 0))
    
    # This is getting complex with nested tuples. Let me use a different representation.
    # Represent matrix entries as integers 0-8 (flattening (a,b) -> a + 3*b)
    
    def to_int(p):
        return p[0] + 3 * p[1]
    
    def from_int(x):
        return (x % 3, x // 3)
    
    def gf_add_int(a, b):
        return to_int(gf_add(from_int(a), from_int(b)))
    
    def gf_mul_int(a, b):
        return to_int(gf_mul(from_int(a), from_int(b)))
    
    def gf_inv_int(a):
        return to_int(gf_inv(from_int(a)))
    
    elements = []
    elem_to_idx = {}
    I = (1, 0, 0, 1)  # [[1,0],[0,1]] in GF(9) ints
    
    for a in range(9):
        for b in range(9):
            for c in range(9):
                for d in range(9):
                    det = gf_add_int(gf_mul_int(a, d), gf_mul_int(b, c))
                    if det == 1:
                        elem = (a, b, c, d)
                        if elem not in elem_to_idx:
                            elem_to_idx[elem] = len(elements)
                            elements.append(elem)
    
    n_elem = len(elements)
    if n_elem != 720:
        return None
    
    def multiply(x, y):
        a, b, c, d = x
        e, f, g2, h = y
        return (
            gf_add_int(gf_mul_int(a, e), gf_mul_int(b, g2)),
            gf_add_int(gf_mul_int(a, f), gf_mul_int(b, h)),
            gf_add_int(gf_mul_int(c, e), gf_mul_int(d, g2)),
            gf_add_int(gf_mul_int(c, f), gf_mul_int(d, h))
        )
    
    # Find order 2 elements
    order2 = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add_int(a, d)
        if tr == 0 and elem != I:
            order2.append(elem)
    
    candidates = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add_int(a, d)
        if elem != I and tr != 0:
            candidates.append(elem)
    
    import random
    random.seed(42)
    
    tried = set()
    best_result = None
    
    for _ in range(300):
        if time.time() > deadline - 30:
            break
        
        s = random.choice(order2)
        t = random.choice(candidates)
        
        key = (s, t)
        if key in tried:
            continue
        tried.add(key)
        
        a, b, c, d = t
        t_inv = (d, gf_add_int(0, b), gf_add_int(0, c), a)
        
        if s == t or s == t_inv or t == t_inv:
            continue
        
        adj = [set() for _ in range(n_elem)]
        
        for i, v in enumerate(elements):
            vs = multiply(v, s)
            vt = multiply(v, t)
            vti = multiply(v, t_inv)
            
            j_s = elem_to_idx.get(vs)
            j_t = elem_to_idx.get(vt)
            j_ti = elem_to_idx.get(vti)
            
            if j_s is not None and j_s != i:
                adj[i].add(j_s)
            if j_t is not None and j_t != i:
                adj[i].add(j_t)
            if j_ti is not None and j_ti != i:
                adj[i].add(j_ti)
        
        valid = True
        for i in range(n_elem):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        min_cycle = _check_girth_fast(adj, n_elem, g)
        
        if min_cycle >= g:
            edges = []
            for i in range(n_elem):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if best_result is None or n_elem < best_result[0]:
                best_result = (n_elem, edges)
            return best_result
    
    return best_result


def _try_sl2_gf25(k, g, deadline):
    """SL(2,25), order 15600."""
    # GF(25) = GF(5)[x]/(x^2+1) since x^2+1 is irreducible over GF(5)? 
    # Check: x^2+1 = 0 mod 5. x=0:1, x=1:2, x=2:0! So x^2+1 = (x-2)(x+2) over GF(5). Not irreducible.
    # Try x^2+2: x=0:2, x=1:3, x=2:1, x=3:1, x=4:3. No root. Irreducible.
    # alpha^2 = -2 = 3 in GF(5)
    
    def gf_add_int(a, b):
        # a, b in 0..24, represent as a = a0 + 5*a1, b = b0 + 5*b1
        a0, a1 = a % 5, a // 5
        b0, b1 = b % 5, b // 5
        return (a0 + b0) % 5 + 5 * ((a1 + b1) % 5)
    
    def gf_mul_int(a, b):
        a0, a1 = a % 5, a // 5
        b0, b1 = b % 5, b // 5
        # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + (a0*b1+a1*b0)*alpha + a1*b1*alpha^2
        # = (a0*b0 + 3*a1*b1) + (a0*b1+a1*b0)*alpha
        r0 = (a0 * b0 + 3 * a1 * b1) % 5
        r1 = (a0 * b1 + a1 * b0) % 5
        return r0 + 5 * r1
    
    def gf_inv_int(a):
        a0, a1 = a % 5, a // 5
        # inv = (a0 - a1*alpha) / (a0^2 + 3*a1^2)
        denom = (a0 * a0 + 3 * a1 * a1) % 5
        if denom == 0:
            return 0
        # inv in GF(5): 1->1, 2->3, 3->2, 4->4
        inv_map = {1: 1, 2: 3, 3: 2, 4: 4}
        inv_denom = inv_map[denom]
        r0 = (a0 * inv_denom) % 5
        r1 = (5 - a1 * inv_denom) % 5
        return r0 + 5 * r1
    
    elements = []
    elem_to_idx = {}
    I = (1, 0, 0, 1)
    
    for a in range(25):
        for b in range(25):
            for c in range(25):
                for d in range(25):
                    det = gf_add_int(gf_mul_int(a, d), gf_mul_int(b, c))
                    if det == 1:
                        elem = (a, b, c, d)
                        if elem not in elem_to_idx:
                            elem_to_idx[elem] = len(elements)
                            elements.append(elem)
    
    n_elem = len(elements)
    if n_elem != 15600:
        return None
    
    def multiply(x, y):
        a, b, c, d = x
        e, f, g2, h = y
        return (
            gf_add_int(gf_mul_int(a, e), gf_mul_int(b, g2)),
            gf_add_int(gf_mul_int(a, f), gf_mul_int(b, h)),
            gf_add_int(gf_mul_int(c, e), gf_mul_int(d, g2)),
            gf_add_int(gf_mul_int(c, f), gf_mul_int(d, h))
        )
    
    order2 = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add_int(a, d)
        if tr == 0 and elem != I:
            order2.append(elem)
    
    candidates = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add_int(a, d)
        if elem != I and tr != 0:
            candidates.append(elem)
    
    import random
    random.seed(42)
    
    tried = set()
    best_result = None
    
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        
        s = random.choice(order2)
        t = random.choice(candidates)
        
        key = (s, t)
        if key in tried:
            continue
        tried.add(key)
        
        a, b, c, d = t
        t_inv = (d, gf_add_int(0, b), gf_add_int(0, c), a)
        
        if s == t or s == t_inv or t == t_inv:
            continue
        
        adj = [set() for _ in range(n_elem)]
        
        for i, v in enumerate(elements):
            vs = multiply(v, s)
            vt = multiply(v, t)
            vti = multiply(v, t_inv)
            
            j_s = elem_to_idx.get(vs)
            j_t = elem_to_idx.get(vt)
            j_ti = elem_to_idx.get(vti)
            
            if j_s is not None and j_s != i:
                adj[i].add(j_s)
            if j_t is not None and j_t != i:
                adj[i].add(j_t)
            if j_ti is not None and j_ti != i:
                adj[i].add(j_ti)
        
        valid = True
        for i in range(n_elem):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        min_cycle = _check_girth_fast(adj, n_elem, g)
        
        if min_cycle >= g:
            edges = []
            for i in range(n_elem):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if best_result is None or n_elem < best_result[0]:
                best_result = (n_elem, edges)
            return best_result
    
    return best_result


def _try_sl2_gf32(k, g, deadline):
    """SL(2,32), order 32736."""
    # GF(32) = GF(2)[x]/(x^5+x^3+1)
    # Primitive polynomial: x^5 + x^3 + 1
    
    exp_table = [0] * 32
    log_table = [0] * 32
    
    exp_table[0] = 1
    log_table[1] = 0
    for i in range(1, 31):
        val = exp_table[i-1] << 1
        if val >= 32:
            val ^= 0x13  # x^5+x^3+1 = 10011 in binary
        exp_table[i] = val
        log_table[val] = i
    
    def gf_add(a, b):
        return a ^ b
    
    def gf_mul(a, b):
        if a == 0 or b == 0:
            return 0
        return exp_table[(log_table[a] + log_table[b]) % 31]
    
    def gf_inv(a):
        if a == 0:
            return 0
        return exp_table[(31 - log_table[a]) % 31]
    
    elements = []
    elem_to_idx = {}
    I = (1, 0, 0, 1)
    
    for a in range(32):
        for b in range(32):
            for c in range(32):
                for d in range(32):
                    det = gf_add(gf_mul(a, d), gf_mul(b, c))
                    if det == 1:
                        elem = (a, b, c, d)
                        if elem not in elem_to_idx:
                            elem_to_idx[elem] = len(elements)
                            elements.append(elem)
    
    n_elem = len(elements)
    if n_elem != 32736:
        return None
    
    def multiply(x, y):
        a, b, c, d = x
        e, f, g2, h = y
        return (
            gf_add(gf_mul(a, e), gf_mul(b, g2)),
            gf_add(gf_mul(a, f), gf_mul(b, h)),
            gf_add(gf_mul(c, e), gf_mul(d, g2)),
            gf_add(gf_mul(c, f), gf_mul(d, h))
        )
    
    order2 = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add(a, d)
        if tr == 0 and elem != I:
            order2.append(elem)
    
    candidates = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add(a, d)
        if elem != I and tr != 0:
            candidates.append(elem)
    
    import random
    random.seed(42)
    
    tried = set()
    best_result = None
    
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        
        s = random.choice(order2)
        t = random.choice(candidates)
        
        key = (s, t)
        if key in tried:
            continue
        tried.add(key)
        
        a, b, c, d = t
        t_inv = (d, b, c, a)
        
        if s == t or s == t_inv or t == t_inv:
            continue
        
        adj = [set() for _ in range(n_elem)]
        
        for i, v in enumerate(elements):
            vs = multiply(v, s)
            vt = multiply(v, t)
            vti = multiply(v, t_inv)
            
            j_s = elem_to_idx.get(vs)
            j_t = elem_to_idx.get(vt)
            j_ti = elem_to_idx.get(vti)
            
            if j_s is not None and j_s != i:
                adj[i].add(j_s)
            if j_t is not None and j_t != i:
                adj[i].add(j_t)
            if j_ti is not None and j_ti != i:
                adj[i].add(j_ti)
        
        valid = True
        for i in range(n_elem):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        min_cycle = _check_girth_fast(adj, n_elem, g)
        
        if min_cycle >= g:
            edges = []
            for i in range(n_elem):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if best_result is None or n_elem < best_result[0]:
                best_result = (n_elem, edges)
            return best_result
    
    return best_result


def _try_sl2_gf4(k, g, deadline):
    """SL(2,4) = A5, order 60."""
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(a, b): return a ^ b
    def gf_mul(a, b): return mul_table[a][b]
    def gf_inv(a):
        if a == 0: return 0
        return {1: 1, 2: 3, 3: 2}[a]
    
    elements = []
    elem_to_idx = {}
    I = (1, 0, 0, 1)
    
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    det = gf_add(gf_mul(a, d), gf_mul(b, c))
                    if det == 1:
                        elem = (a, b, c, d)
                        if elem not in elem_to_idx:
                            elem_to_idx[elem] = len(elements)
                            elements.append(elem)
    
    n_elem = len(elements)
    if n_elem != 60:
        return None
    
    def multiply(x, y):
        a, b, c, d = x
        e, f, g2, h = y
        return (
            gf_add(gf_mul(a, e), gf_mul(b, g2)),
            gf_add(gf_mul(a, f), gf_mul(b, h)),
            gf_add(gf_mul(c, e), gf_mul(d, g2)),
            gf_add(gf_mul(c, f), gf_mul(d, h))
        )
    
    order2 = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add(a, d)
        if tr == 0 and elem != I:
            order2.append(elem)
    
    candidates = []
    for elem in elements:
        a, b, c, d = elem
        tr = gf_add(a, d)
        if elem != I and tr != 0:
            candidates.append(elem)
    
    import random
    random.seed(42)
    
    tried = set()
    best_result = None
    
    for _ in range(500):
        if time.time() > deadline - 30:
            break
        
        s = random.choice(order2)
        t = random.choice(candidates)
        
        key = (s, t)
        if key in tried:
            continue
        tried.add(key)
        
        a, b, c, d = t
        t_inv = (d, gf_add(0, b), gf_add(0, c), a)
        
        if s == t or s == t_inv or t == t_inv:
            continue
        
        adj = [set() for _ in range(n_elem)]
        
        for i, v in enumerate(elements):
            vs = multiply(v, s)
            vt = multiply(v, t)
            vti = multiply(v, t_inv)
            
            j_s = elem_to_idx.get(vs)
            j_t = elem_to_idx.get(vt)
            j_ti = elem_to_idx.get(vti)
            
            if j_s is not None and j_s != i:
                adj[i].add(j_s)
            if j_t is not None and j_t != i:
                adj[i].add(j_t)
            if j_ti is not None and j_ti != i:
                adj[i].add(j_ti)
        
        valid = True
        for i in range(n_elem):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        min_cycle = _check_girth_fast(adj, n_elem, g)
        
        if min_cycle >= g:
            edges = []
            for i in range(n_elem):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if best_result is None or n_elem < best_result[0]:
                best_result = (n_elem, edges)
            return best_result
    
    return best_result


def _try_sl2_prime(q, k, g, deadline):
    """SL(2,q) for prime q."""
    I = (1, 0, 0, 1)
    
    elements = []
    elem_to_idx = {}
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    det = (a * d - b * c) % q
                    if det == 1:
                        elem = (a, b, c, d)
                        if elem not in elem_to_idx:
                            elem_to_idx[elem] = len(elements)
                            elements.append(elem)
    
    n_elem = len(elements)
    if n_elem == 0:
        return None
    
    def multiply(x, y):
        a, b, c, d = x
        e, f, g2, h = y
        return (
            (a * e + b * g2) % q,
            (a * f + b * h) % q,
            (c * e + d * g2) % q,
            (c * f + d * h) % q
        )
    
    order2 = []
    for elem in elements:
        a, b, c, d = elem
        tr = (a + d) % q
        if tr == 0 and elem != I:
            order2.append(elem)
    
    candidates = []
    for elem in elements:
        a, b, c, d = elem
        tr = (a + d) % q
        if elem != I and tr != 0:
            candidates.append(elem)
    
    import random
    random.seed(42)
    
    tried = set()
    best_result = None
    
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        
        s = random.choice(order2)
        t = random.choice(candidates)
        
        key = (s, t)
        if key in tried:
            continue
        tried.add(key)
        
        a, b, c, d = t
        # t_inv = [[d, -b], [-c, a]] mod q
        t_inv = (d, (q - b) % q, (q - c) % q, a)
        
        if s == t or s == t_inv or t == t_inv:
            continue
        
        adj = [set() for _ in range(n_elem)]
        
        for i, v in enumerate(elements):
            vs = multiply(v, s)
            vt = multiply(v, t)
            vti = multiply(v, t_inv)
            
            j_s = elem_to_idx.get(vs)
            j_t = elem_to_idx.get(vt)
            j_ti = elem_to_idx.get(vti)
            
            if j_s is not None and j_s != i:
                adj[i].add(j_s)
            if j_t is not None and j_t != i:
                adj[i].add(j_t)
            if j_ti is not None and j_ti != i:
                adj[i].add(j_ti)
        
        valid = True
        for i in range(n_elem):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        min_cycle = _check_girth_fast(adj, n_elem, g)
        
        if min_cycle >= g:
            edges = []
            for i in range(n_elem):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if best_result is None or n_elem < best_result[0]:
                best_result = (n_elem, edges)
            return best_result
    
    return best_result


def _luw_fallback(k, g):
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
