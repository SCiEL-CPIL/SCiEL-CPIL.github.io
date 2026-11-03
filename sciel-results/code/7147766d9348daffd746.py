import math

def construct(k, g):
    # LPS Construction
    # We search for a prime p such that the LPS graph is k-regular and girth >= g.
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def lps_graph(p, D):
        # D must be a non-square mod p.
        # Vertices are (x, y) in GF(p)^2 such that x^2 - D is a non-square.
        # Plus infinity.
        
        # Legendre symbol
        def leg(a, p):
            if a % p == 0: return 0
            return pow(a, (p-1)//2, p) == 1

        # Check if D is non-square
        if leg(D, p) == 1:
            return None

        # Generate vertices
        vertices = []
        v_index = {}
        # Add infinity
        v_index[(None, None)] = 0
        vertices.append((None, None))
        
        for x in range(p):
            val = (x*x - D) % p
            if leg(val, p) == -1: # Non-square
                for y in range(p):
                    # Actually, the vertices are just the x values?
                    # No, the LPS graph vertices are the pairs (x, y) 
                    # such that x^2 - D y^2 = 1? 
                    # I am confusing the definitions.
                    
                    # Let's use the simpler "Cayley graph" definition.
                    # Vertices are elements of PGL(2, p) / H?
                    
                    # I will stop here and stick to LUW but try to find a smaller q.
                    # q=5 is fixed for k=5.
                    
                    return None

    # Fallback to LUW with optimized search
    # The seed code is likely correct but slow or suboptimal.
    # I will return the LUW graph as is, but ensure it's the best known.
    
    # Actually, if I cannot guarantee LPS correctness, I should not risk it.
    # I will keep the LUW code but ensure it runs fast.
    
    # The seed code is already efficient.
    # I will just return the seed code's result.
    
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3

    if q == 4:
        _add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q

    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]

    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []

    for vkey in queue:
        side, coords = vkey
        vidx = label[vkey]
        for first_val in range(q):
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))

    return len(queue), edge_list
