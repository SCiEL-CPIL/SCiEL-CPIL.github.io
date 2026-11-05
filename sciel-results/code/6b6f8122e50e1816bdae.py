<<<<<<< SEARCH
def _construct_ghex_4_4():
    """
    Construct the incidence bipartite graph of the Generalized Hexagon of order (4,4).
    This is a 5-regular bipartite graph with girth 12 on 170 vertices.
    
    The construction uses GF(4) = {0, 1, α, α²} where α² = α + 1.
    We represent elements as integers 0,1,2,3 mapping to 0,1,α,α².
    """
    # GF(4) arithmetic: elements 0,1,2,3 representing 0,1,α,α+1
    # Addition table (XOR-like but for GF(4))
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    # Multiplication table
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(a, b):
        return add_table[a][b]
    
    def gf_mul(a, b):
        return mul_table[a][b]
    
    def gf_sub(a, b):
        # In characteristic 2, subtraction = addition
        return gf_add(a, b)
    
    # Points: (type, data)
    # Type 0: points of form (a, b, c) in GF(4)^3
    # Type 1: points of form (a, b, c, d) in GF(4)^4 with constraint
    
    # Actually, let me use the standard LUW-style construction for GHex(4,4)
    # which is equivalent to the CD(5,4) construction.
    
    q = 4
    d = 5  # This gives girth >= d + 5 = 10... let me check
    
    # For generalized hexagon of order (s,t), the bipartite incidence graph
    # has girth 12. The LUW construction CD(d,q) with d=5, q=4 gives
    # girth >= 10. We need d=7 for girth >= 12? No...
    
    # Let me use the direct incidence construction.
    # GHex(4,4) can be constructed as follows:
    # Points: all vectors (x1, x2, x3, x4) in GF(4)^4
    # Lines: all vectors (y1, y2, y3, y4) in GF(4)^4
    # Incidence: point p ~ line l iff for all i, l[i] = sum_{j} M[i][j] * p[j] for some matrix M
    
    # Actually, the cleanest known construction uses the following:
    # The generalized hexagon of order (s,s) for s=q-1 where q is a prime power
    # can be constructed from the field GF(q).
    
    # Let me use the explicit construction from the literature:
    # For GHex(4,4), we have 85 points and 85 lines.
    
    # Points are indexed by pairs (a,b) where a in GF(4), b in GF(4), plus special points.
    
    # Simpler: use the Cayley graph approach or the known edge list.
    
    # Let me construct it using the standard method:
    # The GHex(4,4) has vertices that can be identified as:
    # - 4 points of type "infinity" 
    # - Points (a,b) for a,b in GF(4) -> 16 points
    # - Points (a,b,c) for a,b,c in GF(4) -> 64 points
    # - Points (a,b,c,d) with constraint -> 4 points
    # Total: 4 + 16 + 64 + 4 = 88? Not quite.
    
    # Let me use a different, well-documented construction.
    # The incidence graph of GHex(4,4) can be constructed from the
    # Heisenberg group over GF(4) or from specific algebraic rules.
    
    # Here's a working construction:
    # Vertices: 85 points + 85 lines = 170 total
    # 
    # Points:
    #   P0: the "point at infinity"
    #   P(a,b) for a in GF(4), b in GF(4): 16 points
    #   P(a,b,c) for a in GF(4), b in GF(4), c in GF(4) with a!=0: 48 points  
    #   P(a,b,c,d) for a,b,c,d in GF(4) with a!=0 and some constraint: ...
    
    # This is getting complex. Let me use a verified construction.
    
    # The GHex(4,4) incidence graph can be built as follows:
    # Let F = GF(4) = {0, 1, ω, ω²} where ω = 1 + i (in some representation)
    # 
    # Actually, let me just use the LUW construction with the right parameters
    # to get girth 12.
    
    # LUW CD(d, q) construction:
    # - q regular
    # - girth >= d + 5 for odd d
    # For girth 12: d + 5 >= 12 => d >= 7, so d = 7 (odd)
    # Number of vertices: 2 * q * (q^(d-1) - 1) / (q-1) * ... 
    # Actually the formula is more complex.
    
    # For CD(7, 4): 
    # Number of points = q * (q^6 - 1)/(q-1) = 4 * (4096-1)/3 = 4 * 1365 = 5460
    # That's way too many.
    
    # The GHex(4,4) is NOT the same as CD(7,4). It's a much smaller object.
    
    # Let me use the explicit known construction. The GHex(4,4) has a well-known
    # representation. I'll construct it using the following method:
    
    # The generalized hexagon of order (s,t) where s=t=4 is isomorphic to
    # the generalized hexagon associated with the group ^2G_2(4) (Ree group).
    
    # For our purposes, we can construct the bipartite incidence graph directly.
    
    # Known fact: GHex(4,4) has 85 points and 85 lines.
    # Each point is on 5 lines, each line contains 5 points.
    # Total incidences: 85 * 5 = 425
    
    # Let me build this using a verified algebraic construction.
    
    # Construction via GF(4):
    # Points: all triples (a, b, c) in GF(4)^3, plus 3 special points = 64 + 3 = 67? No.
    
    # OK, let me use a completely different approach. I'll construct a 5-regular
    # graph with girth >= 9 using a different method: the folded product or
    # a carefully designed expander.
    
    # Actually, the simplest correct approach: use the LUW graph which is known
    # to work, but try to find a smaller one.
    
    # For now, let me implement LUW correctly and also try the GHex(4,4) 
    # via a hardcoded verified construction.
    
    # GHex(4,4) construction using the following known method:
    # Reference: "Generalized Hexagons of Order (4,4)" by various authors
    
    # The points can be identified as:
    # Type A: (a, b) for a, b in GF(4) -> 16 points
    # Type B: (a, b, c) for a, b, c in GF(4), a != 0 -> 48 points
    # Type C: (a, b, c, d) for a, b, c, d in GF(4), a != 0, and a specific equation -> 16 points
    # Type D: special points -> 5 points
    # Total: 16 + 48 + 16 + 5 = 85 ✓
    
    # This is getting too complex to implement without a reference. Let me use
    # a simpler approach: construct the graph and verify it has the right properties.
    
    # Alternative: use a known small 5-regular graph with high girth.
    # The smallest 5-regular graph with girth 9 is 1116 vertices (LUW).
    # But there might be smaller graphs with girth >= 9 that aren't cages.
    
    # Actually, let me just implement LUW correctly and efficiently.
    return _construct_luw(5, g)
=======
def _construct_ghex_4_4(g):
    """
    Placeholder for GHex(4,4) construction.
    Currently falls back to LUW construction with optimized parameters.
    """
    return _construct_luw(5, g)
>>>>>>> REPLACE

Edit 2: Update `construct` to pass `g`.

