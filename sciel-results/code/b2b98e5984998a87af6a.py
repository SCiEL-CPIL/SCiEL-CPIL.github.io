import time
import math
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Tries Cayley graphs on SL(2, q) for small q.
    """
    start_time = time.time()
    deadline = start_time + 140

    if k != 3:
        # Fallback for other k, though task is specific to k=3
        # For now, return a dummy or try LUW if available
        return _lnw_construct(k, g)

    # Try Cayley graphs on SL(2, q)
    # q must be a power of 2 for the simple field arithmetic in char 2
    for q in [16, 32, 64, 128]:
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges, girth = _construct_cayley_sl2(q)
        except Exception:
            continue
            
        if girth >= g:
            return n, edges

    # Fallback to LUW if Cayley failed (should not happen for q=128)
    # But LUW is slow. We'll try a small d.
    # If we get here, we might fail.
    # Let's try d=10 for LUW (n=3^10=59049)
    if time.time() < deadline - 20:
        try:
            n, edges = _lnw_construct_params(3, 10)
            adj = _edges_to_adj(edges, n)
            girth = _calculate_girth_fast(adj, n) # Use fast girth check
            if girth >= g:
                return n, edges
        except Exception:
            pass

    # Last resort: Return a very large LUW graph? 
    # Or raise error? 
    # The task says "invalid result... scores 0".
    # We must return something valid.
    # SL(2, 128) should definitely have girth > 16.
    # If it didn't, we have a problem.
    # But we looped up to 128.
    
    # If all else fails, return a trivial graph? No, must be girth g.
    # We assume SL(2, 128) works.
    raise RuntimeError("Failed to construct graph")

def _construct_cayley_sl2(q):
    """
    Construct 3-regular Cayley graph on SL(2, q) (char 2).
    Returns (n, edges, girth).
    """
    # 1. Generate elements and index
    # Element (a, b, c) -> d is determined.
    # We map (a, b, c) to an integer ID.
    # a in 0..q-1, b in 0..q-1, c in 0..q-1.
    # Total q^3 tuples. But only q(q^2-1) are valid.
    
    # Precompute inverses for non-zero elements in GF(q) (char 2)
    # Since q is power of 2, GF(q) is not just integers mod q for q>2.
    # Wait, the previous code used `pow(i, q-2, q)`. This only works if q is prime.
    # For q=16, 32, etc., we need finite field arithmetic.
    # This is a CRITICAL issue in the original code!
    # `pow(i, q-2, q)` gives the inverse in Z_q, which is NOT the same as GF(2^m) for m>1.
    # We MUST implement GF(2^m) arithmetic.
    
    # Let's implement a simple GF(2^m) class or functions.
    # For q=16, m=4. Primitive polynomial: x^4 + x + 1 (0b10011 = 19)
    # For q=32, m=5. Primitive polynomial: x^5 + x^3 + 1 (0b101001 = 41)
    # For q=64, m=6. Primitive polynomial: x^6 + x + 1 (0b1000011 = 67)
    # For q=128, m=7. Primitive polynomial: x^7 + x + 1 (0b10000011 = 131)
    
    m = int(math.log2(q))
    # Find a primitive polynomial
    # We can hardcode for 16, 32, 64, 128
    if m == 4:
        poly = 0b10011
    elif m == 5:
        poly = 0b101001
    elif m == 6:
        poly = 0b1000011
    elif m == 7:
        poly = 0b10000011
    else:
        # Try to find one or use a standard one
        # For simplicity, we assume m <= 7 for this task
        poly = 0 # Error
        
    # GF(2^m) operations
    def gf_add(x, y):
        return x ^ y
        
    def gf_mul(x, y):
        res = 0
        while y:
            if y & 1:
                res ^= x
            y >>= 1
            x <<= 1
            if x >= q:
                x ^= poly
        return res
        
    def gf_inv(x):
        if x == 0:
            return 0
        # x^(q-2)
        res = 1
        base = x
        exp = q - 2
        while exp:
            if exp & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            exp >>= 1
        return res

    # Generate all valid elements
    # Element is (a, b, c). d is derived.
    # We store elements in a list.
    # We also need a map from (a,b,c) to index.
    # Since q is small, we can use a dict.
    
    elements = []
    index = {}
    
    # Helper to get d
    def get_d(a, b, c):
        if a == 0:
            # bc must be 1. d is free? 
            # No, in the representation (a,b,c), if a=0, we have multiple d's?
            # Wait, the group element is the matrix.
            # If a=0, bc=1. The matrix is [[0, b], [c, d]].
            # Different d's give different matrices.
            # So (0, b, c) does NOT uniquely identify the element.
            # We need to include d in the key for a=0?
            # But for a!=0, d is unique.
            
            # To keep it simple, let's iterate over all matrices directly?
            # No, that's 4 loops.
            
            # Let's handle a!=0 and a==0 separately.
            pass
            
    # Re-strategize generation:
    # Case 1: a != 0.
    # a in 1..q-1. b in 0..q-1. c in 0..q-1.
    # d = (1 + bc) / a.
    # This gives q-1 * q * q elements.
    
    # Case 2: a == 0.
    # bc = 1. b in 1..q-1. c = inv(b).
    # d in 0..q-1.
    # This gives (q-1) * q elements.
    
    # Total q(q-1) + q(q-1) = 2q(q-1)?
    # Wait, q(q^2-1) = q(q-1)(q+1).
    # My count:
    # Case 1: (q-1) * q * q = q^2(q-1).
    # Case 2: (q-1) * q = q(q-1).
    # Total: q(q-1)(q+1) = q(q^2-1). Correct.
    
    # We need a unique ID for each element.
    # For a != 0, key is (a, b, c).
    # For a == 0, key is (0, b, c, d). Note c is determined by b.
    
    # Let's just create a list of tuples (a,b,c,d) and index them.
    
    def add_elem(a, b, c, d):
        key = (a, b, c, d)
        if key not in index:
            index[key] = len(elements)
            elements.append((a, b, c, d))
            
    # Case 1: a != 0
    inv_a = [0] * q
    for i in range(1, q):
        inv_a[i] = gf_inv(i)
        
    for a in range(1, q):
        ia = inv_a[a]
        for b in range(q):
            for c in range(q):
                # d = (1 + bc) * a^-1
                bc = gf_mul(b, c)
                num = gf_add(1, bc)
                d = gf_mul(num, ia)
                add_elem(a, b, c, d)
                
    # Case 2: a == 0
    for b in range(1, q):
        c = gf_inv(b)
        for d in range(q):
            add_elem(0, b, c, d)
            
    n = len(elements)
    
    # Define Generators S1, S2, S3
    # S1 = [[0, 1], [1, 0]]
    # S2 = [[1, 1], [0, 1]]
    # S3 = [[1, 0], [1, 1]]
    
    S1 = (0, 1, 1, 0)
    S2 = (1, 1, 0, 1)
    S3 = (1, 0, 1, 1)
    
    # Precompute neighbors
    # next[i][j] = index of elements[i] * S_j
    nexts = [[0]*3 for _ in range(n)]
    
    for i, (a1, b1, c1, d1) in enumerate(elements):
        for j, (a2, b2, c2, d2) in enumerate([S1, S2, S3]):
            # Multiply
            a = gf_add(gf_mul(a1, a2), gf_mul(b1, c2))
            b = gf_add(gf_mul(a1, b2), gf_mul(b1, d2))
            c = gf_add(gf_mul(c1, a2), gf_mul(d1, c2))
            d = gf_add(gf_mul(c1, b2), gf_mul(d1, d2))
            key = (a, b, c, d)
            nexts[i][j] = index[key]
            
    # Build edges
    edges = []
    seen = set()
    for u in range(n):
        for j in range(3):
            v = nexts[u][j]
            if u < v:
                edges.append((u, v))
                
    # Calculate Girth via BFS from 0
    girth = _calculate_girth_cayley(nexts, n)
    
    return n, edges, girth

def _calculate_girth_cayley(nexts, n):
    """
    Calculate girth of a Cayley graph given nexts table.
    BFS from 0.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = [0]
    head = 0
    min_cycle = n + 1
    
    while head < len(queue):
        u = queue[head]
        head += 1
        
        # If dist[u] >= min_cycle, we can't find a shorter cycle starting from further nodes?
        # Actually, if we find a cycle of length L, any cycle formed by nodes at dist >= L/2 
        # combined with others might be longer.
        # But BFS explores in order of distance.
        # If we find a cycle, it's the shortest one involving the "first" collision?
        # Not necessarily. We should check all edges.
        
        du = dist[u]
        
        for v_idx in nexts[u]:
            v = v_idx
            if dist[v] == -1:
                dist[v] = du + 1
                parent[v] = u
                queue.append(v)
            else:
                # If v is visited and not parent
                if v != parent[u]:
                    # Cycle length
                    # The path 0->...->u->v->...->0?
                    # No, the cycle is u-v plus the tree paths to u and v.
                    # Length = dist[u] + dist[v] + 1.
                    cycle_len = du + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        
    return min_cycle if min_cycle <= n else n + 1

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth_fast(adj, n):
    # For LUW fallback
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct_params(k, d):
    """
    LUW construction with fixed q=k and variable d.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule generation depends on d
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

def _lnw_construct(k, g):
    # Original fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
