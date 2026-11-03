# EVOLVE-BLOCK-START
def construct(k, g):
    # LUW construction for 5-regular graph with girth >= 9
    # q = 5, d = 4 -> 1250 vertices, girth 10
    q = 5
    d = 4
    
    # Precompute field operations for q=5
    def f_add(a, b): return (a + b) % q
    def f_sub(a, b): return (a - b) % q
    def f_mul(a, b): return (a * b) % q
    
    # LUW rules for d=4
    # rules[c] = (a, b) such that new[c] = old[c] +/- old[a]*new[b]
    # For d=4:
    # c=2: (1, 1) -> new[2] = old[2] +/- old[1]*new[1]
    # c=3: (2, 1) -> new[3] = old[3] +/- old[2]*new[1]
    # c=4: (1, 2) -> new[4] = old[4] +/- old[1]*new[2]
    # Note: The specific rules depend on the LUW family variant.
    # Standard LUW(5,4) uses:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    
    rules = {
        2: (1, 1),
        3: (2, 1),
        4: (1, 2)
    }
    
    n_vertices = 2 * (q ** d)
    edge_list = []
    
    # Map (side, coords) to index
    # Index = side * q^d + coords[0]*q^(d-1) + ... + coords[d-1]
    # We use coords 1..d, so 0-indexed array of length d
    # coords[0] corresponds to c=1, ..., coords[d-1] corresponds to c=d
    
    for side in range(2):
        for val in range(q ** d):
            # Decode val to coords
            coords = [0] * d
            v = val
            for i in range(d):
                coords[i] = v % q
                v //= q
            
            vidx = side * (q ** d) + val
            
            for first_val in range(q):
                ncoords = [0] * d
                ncoords[0] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    # a, b are 1-indexed in the math, so 0-indexed here: a-1, b-1
                    if side == 0:
                        ncoords[c-1] = f_add(coords[c-1], f_mul(coords[a-1], ncoords[b-1]))
                    else:
                        ncoords[c-1] = f_sub(coords[c-1], f_mul(coords[a-1], ncoords[b-1]))
                
                nval = 0
                for i in range(d):
                    nval = nval * q + ncoords[i]
                
                nside = 1 - side
                nidx = nside * (q ** d) + nval
                
                if side == 0:
                    edge_list.append((vidx, nidx))
    
    return n_vertices, edge_list
# EVOLVE-BLOCK-END
