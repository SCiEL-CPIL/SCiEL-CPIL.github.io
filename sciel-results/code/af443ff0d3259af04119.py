import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Lazebnik-Woldar construction for R_5(C4) = 26
    # Vertices: GF(25) ∪ {∞}, total 26 vertices
    # GF(25) = GF(5²), represented as pairs (a, b) where element = a + b*α, α² = α + 2 (irreducible x²-x-2 over GF(5))
    # Actually, x² - x - 2 = (x-2)(x+1) over GF(5), so that's not irreducible.
    # Need an irreducible polynomial of degree 2 over GF(5).
    # x² + 1: check if -1 is a square mod 5. 1²=1, 2²=4, 3²=4, 4²=1. So -1=4 is a square. Not irreducible.
    # x² + 2: check if -2=3 is a square. No. So x²+2 is irreducible over GF(5).
    # Let α² = -2 = 3 in GF(5).
    
    p = 5
    # Represent GF(25) elements as (a, b) meaning a + b*alpha, alpha^2 = 3 (i.e., -2)
    # alpha^2 = 3, so alpha^3 = 3*alpha, alpha^4 = 3*alpha^2 = 9 = 4, etc.
    
    def gf25_mul(x, y):
        """Multiply two elements in GF(25). x, y are (a, b) tuples."""
        a1, b1 = x
        a2, b2 = y
        # (a1 + b1*α)(a2 + b2*α) = a1*a2 + (a1*b2 + a2*b1)*α + b1*b2*α²
        # = a1*a2 + 3*b1*b2 + (a1*b2 + a2*b1)*α
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
    
    def gf25_add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def gf25_sub(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
    
    # Enumerate all 25 elements of GF(25)
    gf25_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Find a primitive element (generator of GF(25)*)
    # GF(25)* has order 24. We need an element of order 24.
    # Try each nonzero element and check its order.
    
    def gf25_pow(x, n):
        result = (1, 0)
        base = x
        while n > 0:
            if n % 2 == 1:
                result = gf25_mul(result, base)
            base = gf25_mul(base, base)
            n //= 2
        return result
    
    def gf25_order(x):
        """Compute the multiplicative order of x in GF(25)*."""
        # Order must divide 24
        # Check divisors of 24: 1, 2, 3, 4, 6, 8, 12, 24
        for d in [1, 2, 3, 4, 6, 8, 12, 24]:
            if gf25_pow(x, d) == (1, 0):
                return d
        return -1
    
    # Find primitive element
    primitive = None
    for elem in gf25_elements:
        if elem == (0, 0):
            continue
        if gf25_order(elem) == 24:
            primitive = elem
            break
    
    if primitive is None:
        # Fallback: should not happen
        return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
    
    # Generate all powers of primitive: g^0, g^1, ..., g^23
    # These are all nonzero elements
    powers = []
    current = (1, 0)
    for i in range(24):
        powers.append(current)
        current = gf25_mul(current, primitive)
    
    # Map each nonzero element to its discrete log (exponent)
    log_map = {}
    for i, elem in enumerate(powers):
        log_map[elem] = i
    
    # Now, we need to partition the 24 nonzero elements into 4 groups of 6
    # such that each group is a coset of a subgroup of order 6.
    # A subgroup of order 6 in GF(25)*: since 24 = 4*6, we need an element of order 6.
    # h = g^4 has order 6 (since g has order 24, g^4 has order 24/gcd(4,24) = 24/4 = 6)
    # The subgroup H = {g^0, g^4, g^8, g^12, g^16, g^20}
    # Cosets: g^0*H, g^1*H, g^2*H, g^3*H
    
    # Define the 5 colour classes for edges within GF(25):
    # Colour 0: edges where one endpoint is 0 (i.e., (0,0))
    # Colours 1-4: for edges between nonzero elements a, b, colour = (log(a) - log(b)) mod 4
    #   i.e., which coset of H the ratio a/b falls into
    
    # Wait, I need to verify this doesn't create C4s.
    # In this construction, a monochromatic C4 would require 4 vertices a,b,c,d
    # such that a-b, b-c, c-d, d-a are all the same colour.
    # For colours 1-4: this means a/b, b/c, c/d, d/a are all in the same coset of H.
    # The product (a/b)(b/c)(c/d)(d/a) = 1, so we need 4 elements of the same coset
    # whose product is 1. This is possible in general, so I need to verify the construction more carefully.
    
    # Actually, the standard Lazebnik-Woldar construction is known to work. Let me just implement it and verify.
    
    n = 26
    # Vertex 0 to 24: elements of GF(25)
    # Vertex 25: infinity
    
    # Assign colour to each pair (i, j) with i < j
    colors = []
    
    def get_color(i, j):
        """Get colour for edge between vertices i and j."""
        if i == 25 or j == 25:
            # Edge involving infinity
            other = j if i == 25 else i
            if other == 0:  # (0,0) element
                return 0
            else:
                # Colour based on which coset the element is in
                # log(other) mod 4 gives the coset
                return (log_map[gf25_elements[other]] % 4) + 1
        else:
            # Both in GF(25)
            elem_i = gf25_elements[i]
            elem_j = gf25_elements[j]
            if elem_i == (0, 0) or elem_j == (0, 0):
                return 0
            else:
                # Both nonzero, colour by coset of ratio
                log_i = log_map[elem_i]
                log_j = log_map[elem_j]
                return ((log_i - log_j) % 4) + 1
    
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(get_color(i, j))
    
    # Verify: check that no colour class contains a C4
    # A C4 in colour c means there exist 4 distinct vertices a,b,c,d
    # such that edges (a,b), (b,c), (c,d), (d,a) are all colour c.
    # Equivalently, two vertices have two common neighbours in colour c.
    
    def validate(colors, n):
        """Check that no colour class contains a C4."""
        # For each colour, build adjacency and check for C4
        adj = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                adj[colors[idx]].add((i, j))
                idx += 1
        
        for c in range(5):
            # For each pair of vertices, count common neighbours in colour c
            # If any pair has >= 2 common neighbours, there's a C4
            # More efficient: for each pair (u,v) in colour c, count common neighbours
            # A C4 exists if there are u,v with |N(u) ∩ N(v)| >= 2 in colour c
            # Actually, we need: exist distinct a,b,c,d with all 4 edges in colour c
            # This means a and c have two common neighbours b and d (or similar)
            
            # Build adjacency lists for colour c
            adj_c = [set() for _ in range(n)]
            for (u, v) in adj[c]:
                adj_c[u].add(v)
                adj_c[v].add(u)
            
            # For each pair of vertices, count common neighbours
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_c[u] & adj_c[v]
                    if len(common) >= 2:
                        return False
        return True
    
    if not validate(colors, n):
        # Fallback to K11
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    return {'n': n, 'colors': colors}
