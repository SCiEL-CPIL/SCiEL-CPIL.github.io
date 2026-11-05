"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need to colour edges of K16 such that every K4 has at least 5 distinct colours.
    # The seed uses (i+j) % 17 which gives 17 colours.
    # We want to minimize the number of colours used.
    
    # Strategy: Use vertex labels to derive edge colours.
    # Let's try a construction based on finite fields or modular arithmetic.
    
    # Approach: Assign each vertex v a label x_v in Z_p for some prime p.
    # Edge colour = (x_i * x_j) mod p or (x_i + x_j) mod p or similar.
    
    # The key constraint: in any 4 vertices, the 6 edges must have >= 5 distinct colours.
    # This means at most 1 pair of edges can share a colour within any K4.
    
    # Let's think about what structures allow this.
    # If we use a proper edge-colouring (all edges different), that's 120 colours - too many.
    # We need to reuse colours but carefully.
    
    # Idea: Use a 1-factorization-like approach but with fewer colours.
    # Alternatively, use a group-based construction.
    
    # Let's try: vertices 0..15, colour edge (i,j) = (i + j) mod m for some m.
    # The seed uses m=17. Can we do better with a different function?
    
    # Consider: colour (i,j) = (i * j) mod p for prime p >= 16.
    # But i*j mod p might have more collisions.
    
    # Better idea: Use two moduli or a mixed construction.
    
    # Let's try a systematic search over small parameterized families.
    
    # Family 1: (i + a*j) mod m for various a, m
    # Family 2: (i * j) mod m  
    # Family 3: (i + j) mod m1 combined with something
    
    # Actually, let's think more carefully. The constraint is that in any K4, 
    # among the 6 edges, at most 1 pair can have the same colour.
    
    # For the (i+j) mod 17 construction:
    # In a K4 with vertices a<b<c<d, edges are:
    # a+b, a+c, a+d, b+c, b+d, c+d (mod 17)
    # Two edges share a colour iff their sums are congruent mod 17.
    # e.g., a+d = b+c iff a-c = b-d iff (a,c) and (b,d) form a "cross" with equal sum.
    # In a 4-set, there are 3 ways to partition into two pairs:
    # (ab,cd), (ac,bd), (ad,bc)
    # For each partition, the two edges have the same colour iff the sums match.
    # With mod 17 and 4 distinct vertices from 0..15, can two different pairings have the same sum?
    # a+b = c+d and a+c = b+d would imply b=c and a=d, contradiction.
    # So at most one pairing can have equal sums. Hence at most 1 collision, giving >= 5 distinct colours.
    
    # The question is: can we use fewer than 17 colours?
    
    # Lower bound: f(16) >= 13 per the problem statement.
    
    # Let's try to find a construction with 13, 14, or 15 colours.
    
    # Try: partition vertices into groups and use a block design.
    
    # Alternative: use a 1-factorization of K16. K16 has a 1-factorization into 15 perfect matchings.
    # If we colour each matching with a distinct colour, that's 15 colours.
    # In any K4, how many edges from the same matching? At most 1 (since a matching has disjoint edges).
    # Wait, a K4 has 6 edges. In a 1-factorization, each perfect matching contributes at most 1 edge to any K4?
    # No, a perfect matching on 16 vertices has 8 edges. A K4 uses 4 vertices, so a perfect matching can have at most 2 edges within those 4 vertices (if it pairs them up).
    # Actually, a perfect matching on 16 vertices: for any 4 vertices, the matching either has 0, 1, or 2 edges within them.
    # If it has 2 edges, those 2 edges share the same colour (the matching's colour).
    # So in a K4, we could have up to 15 matchings each contributing edges. But each matching contributes at most 2 edges to the K4.
    # The 6 edges of K4 are partitioned among the 15 matchings. Each matching contributes 0, 1, or 2 edges.
    # If a matching contributes 2 edges, that's a collision (2 edges same colour).
    # How many matchings can contribute 2 edges to a given K4?
    # The 4 vertices can be paired in 3 ways. Each pairing corresponds to one perfect matching that contains both edges of that pairing (extended to a full perfect matching).
    # So at most 3 matchings can have 2 edges in the K4. That means at most 3 collisions.
    # With 15 colours and 3 collisions, we'd have 15 - 3 = 12 distinct colours? No wait.
    # We have 6 edges. If 3 matchings each have 2 edges (so 3 pairs), that's 3 distinct colours for those 6 edges.
    # The other 12 matchings have 0 edges in this K4.
    # So we'd have only 3 distinct colours! That's way too few.
    
    # So 1-factorization doesn't work directly.
    
    # Let's go back to the additive construction. (i+j) mod m.
    # We need m such that for any 4 distinct vertices from 0..15, the 6 sums have >= 5 distinct values mod m.
    # Equivalently, at most 1 pair of the 6 sums are equal mod m.
    
    # The sums are: a+b, a+c, a+d, b+c, b+d, c+d for a<b<c<d.
    # Collisions occur when:
    # a+b = c+d (mod m) -> (c-a) = (d-b) ... no, a+b = c+d means the two pairings have equal sum.
    # a+c = b+d (mod m)
    # a+d = b+c (mod m)
    # And also cross terms like a+b = a+c implies b=c which is impossible.
    # So only the 3 "cross" pairings can collide.
    
    # For mod m, we need that for any 4 distinct elements from {0,...,15}, at most one of the 3 cross-pairings has equal sums mod m.
    
    # With m=17 (prime > 16), all sums are distinct as integers in range [1, 29], so mod 17 they can collide.
    # The seed claims it works. Let's trust that and try smaller m.
    
    # Try m=16: sums range from 1 to 29. Mod 16, values 1..15,0.
    # Can we have two collisions? E.g., a+b = c+d mod 16 AND a+c = b+d mod 16?
    # a+b = c+d and a+c = b+d implies (a+b)-(a+c) = (c+d)-(b+d) => b-c = c-b => 2(b-c)=0 mod 16 => b-c = 0 or 8 mod 16.
    # Since b != c, we need b-c = 8 mod 16. So if c = b+8 and a+b = c+d = b+8+d => a = 8+d.
    # With a<b<c<d, a = d+8 and c = b+8. Since a < b, d+8 < b, so d < b-8. But c = b+8 > b > d, contradiction with c < d.
    # Wait, a < b < c < d. c = b+8, a = d+8. But a < b implies d+8 < b, so d < b-8. But d > c = b+8 > b, contradiction.
    # So no double collision for m=16? Let me re-examine.
    
    # Actually let's just try m=16 and check if it's valid.
    
    def check_validity(colours, n=16):
        """Check that every K4 has at least 5 distinct colours."""
        import itertools
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            edge_colours = set()
            for i, j in edges:
                idx = i * (n - 1 - i) + (j - i - 1)  # not quite right, let's use a dict
            # Better: build a lookup
            pass
        
        # Simpler: build a 2D array
        colour_matrix = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                colour_matrix[i][j] = colours[idx]
                colour_matrix[j][i] = colours[idx]
                idx += 1
        
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            cols = set()
            for i in range(4):
                for j in range(i+1, 4):
                    cols.add(colour_matrix[quad[i]][quad[j]])
            if len(cols) < 5:
                return False
        return True
    
    # Let's test m=16
    colours_16 = [(i + j) % 16 for i in range(16) for j in range(i + 1, 16)]
    
    # Quick check
    import itertools
    valid = True
    colour_matrix = [[-1]*16 for _ in range(16)]
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            colour_matrix[i][j] = colours_16[idx]
            colour_matrix[j][i] = colours_16[idx]
            idx += 1
    
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        cols = set()
        cols.add(colour_matrix[a][b])
        cols.add(colour_matrix[a][c])
        cols.add(colour_matrix[a][d])
        cols.add(colour_matrix[b][c])
        cols.add(colour_matrix[b][d])
        cols.add(colour_matrix[c][d])
        if len(cols) < 5:
            valid = False
            break
    
    if valid:
        # Count distinct colours
        distinct = len(set(colours_16))
        # Return colours, but they should be in range 0..119. 
        # The task says "120 integer colours in 0..119" - I think this means each colour value must be in 0..119,
        # and we minimize the number of distinct colours used.
        return colours_16
    
    # If m=16 doesn't work, try m=17 (the seed)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
