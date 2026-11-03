<<<<<<< SEARCH
def _construct_cage_voltage(k, g, deadline):
    """
    Construct the (4,10)-cage using the known algebraic construction.
    The (4,10)-cage has 320 vertices and can be constructed as a Cayley graph
    over Z_2 x Z_2 x Z_40 with appropriate generators, or via the Exoo et al.
    construction. We use a direct algebraic construction.
    """
    if k != 4 or g != 10:
        return None

    # The (4,10)-cage: 320 vertices, 4-regular, girth 10
    # Construction: Use the known structure from Exoo et al.
    # Vertices are (a, b, c) where a in Z_2, b in Z_2, c in Z_40
    # Generators: (0,0,1), (0,0,-1), (1,0,0), (0,1,1)
    # But we need to verify girth = 10

    n = 320
    group_size = 40

    def to_id(a, b, c):
        return (a * 2 + b) * group_size + c

    def from_id(x):
        c = x % group_size
        rest = x // group_size
        b = rest % 2
        a = rest // 2
        return (a, b, c)

    # Define the 4 generators
    generators = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 1)]

    edges = []
    seen_edges = set()
    for a in range(2):
        for b in range(2):
            for c in range(group_size):
                v = to_id(a, b, c)
                for gen in generators:
                    ga, gb, gc = gen
                    na = (a + ga) % 2
                    nb = (b + gb) % 2
                    nc = (c + gc) % group_size
                    u = to_id(na, nb, nc)
                    if v < u:
                        edge = (v, u)
                    else:
                        edge = (u, v)
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        edges.append(edge)

    # Verify the graph
    if _verify_girth(n, edges, k, g):
        return n, edges

    # If the simple Cayley graph doesn't work, try the known (4,10)-cage
    # from the literature. The cage is constructed as a 2-fold cover of
    # a specific 4-regular graph on 160 vertices.
    # Alternative: use the LUW construction which is known to work
    return None
=======
def _construct_cage_voltage(k, g, deadline):
    """
    Construct the (4,10)-cage using algebraic constructions.
    Tries multiple candidate Cayley graphs over abelian groups of order 320,
    verifying girth for each candidate.
    """
    if k != 4 or g != 10:
        return None

    n = 320

    def _try_cayley(apply_fn, num_gens=4):
        """Build edges from apply_fn(v, gen_idx) and verify girth."""
        edges = []
        seen_edges = set()
        for v in range(n):
            for gi in range(num_gens):
                u = apply_fn(v, gi)
                if u == v:
                    return None
                lo, hi = (v, u) if v < u else (u, v)
                if (lo, hi) not in seen_edges:
                    seen_edges.add((lo, hi))
                    edges.append((lo, hi))
        if len(edges) != n * k // 2:
            return None
        if _verify_girth(n, edges, k, g):
            return edges
        return None

    # Z_2 x Z_2 x Z_40: involutions are (1,0,0),(0,1,0),(0,0,20),(1,1,0),(1,0,20),(0,1,20),(1,1,20)
    # Need 4 that generate the group. (1,0,0) and (0,1,0) generate Z_2xZ_2, need something with odd c-component... 
    # but all involutions have even c-component (0 or 20). So can't reach all of Z_40!
    # This means Z_2 x Z_2 x Z_40 cannot have a Cayley graph with 4 involutory generators that's connected.
    # We need non-involutory generators. But then the graph isn't necessarily 4-regular in the simple sense.
    # Actually for Cayley graphs, generators don't need to be involutions. We just need 4 generators
    # (counting inverses). If a generator has order > 2, we need its inverse as a separate generator.
    # For 4-regular, we need exactly 4 generators where each generator's inverse is also in the set.
    # So either all are involutions, or we have 2 pairs of (g, g^-1).
    
    # Let's try 2 pairs: generators (a,b,c) and (a,b,-c) for some (a,b,c)
    # Plus 2 involutions from the Z_2xZ_2 part
    
    # Try: generators = {(1,0,0), (0,1,0), (0,0,1), (0,0,-1)}
    # This is the direct product Cayley graph of Z_2 x Z_2 x Z_40 with standard generators.
    # This has many 4-cycles (e.g., go (0,0,1), (1,0,0), (0,0,-1), (-1,0,0) back).
    # Girth will be 4. Not good.
    
    # Try: generators = {(1,0,0), (0,1,1), (0,1,-1), (1,0,20)}
    # Check: (1,0,0) is involution. (0,1,1)^2 = (0,0,2) not identity. So (0,1,1) and (0,1,-1) are a pair.
    # (1,0,20) is involution.
    # Do these generate the group? (1,0,0) gives a. (0,1,1) gives b and c=1. (0,1,-1) gives b and c=-1.
    # From (0,1,1) and (0,1,-1): sum = (0,0,0), diff = (0,0,2). So we get (0,0,2).
    # With (0,0,2) and (0,1,1): we can get (0,0,2k) for all k, and (0,1, 2k+1).
    # We can get all of Z_2 x Z_40. With (1,0,0) we get all of Z_2 x Z_2 x Z_40. Good.
    
    candidate_sets = [
        # Each entry is a list of 4 generators for Z_2 x Z_2 x Z_40
        [(1, 0, 0), (0, 1, 1), (0, 1, -1), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 1), (0, 1, -1), (0, 1, 20)],
        [(1, 0, 0), (0, 1, 1), (0, 1, -1), (1, 1, 0)],
        [(1, 0, 0), (0, 1, 2), (0, 1, -2), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 3), (0, 1, -3), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 4), (0, 1, -4), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 5), (0, 1, -5), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 6), (0, 1, -6), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 7), (0, 1, -7), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 8), (0, 1, -8), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 9), (0, 1, -9), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 10), (0, 1, -10), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 11), (0, 1, -11), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 12), (0, 1, -12), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 13), (0, 1, -13), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 14), (0, 1, -14), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 15), (0, 1, -15), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 16), (0, 1, -16), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 17), (0, 1, -17), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 18), (0, 1, -18), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 19), (0, 1, -19), (1, 0, 20)],
        [(1, 0, 0), (0, 1, 20), (0, 1, -20), (1, 0, 20)],  # (0,1,20) and (0,1,-20) same since -20=20 mod 40
    ]
    
    gs = 40
    for gens in candidate_sets:
        def apply(v, gi, _gens=gens):
            a, b, c = v // gs // 2, (v // gs) % 2, v % gs
            ga, gb, gc = _gens[gi]
            return ((a + ga) % 2) * 2 * gs + ((b + gb) % 2) * gs + (c + gc) % gs
        edges = _try_cayley(apply)
        if edges is not None:
            return n, edges

    # Try Z_2 x Z_2 x Z_2 x Z_40
    gs4 = 40
    candidate_sets_4 = [
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 1), (0, 0, 1, -1)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 2), (0, 0, 1, -2)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 3), (0, 0, 1, -3)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 4), (0, 0, 1, -4)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 5), (0, 0, 1, -5)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 6), (0, 0, 1, -6)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 7), (0, 0, 1, -7)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 8), (0, 0, 1, -8)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 9), (0, 0, 1, -9)],
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 10), (0, 0, 1, -10)],
    ]
    
    for gens in candidate_sets_4:
        def apply4(v, gi, _gens=gens):
            d = v % gs4
            c = (v // gs4) % 2
            b = (v // (2 * gs4)) % 2
            a = (v // (4 * gs4)) % 2
            ga, gb, gc, gd = _gens[gi]
            return ((a + ga) % 2) * 4 * gs4 + ((b + gb) % 2) * 2 * gs4 + ((c + gc) % 2) * gs4 + (d + gd) % gs4
        edges = _try_cayley(apply4)
        if edges is not None:
            return n, edges

    return None
>>>>>>> REPLACE

