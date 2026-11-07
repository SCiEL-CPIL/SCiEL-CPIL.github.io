"""Construct mutually orthogonal Latin squares of order 10 using GF(25) structure."""

def construct():
    """
    Return a list of mutually orthogonal Latin squares of order 10.
    
    Uses the finite field GF(25) = GF(5)[x]/(x^2 + 2) to construct
    orthogonal Latin squares via linear maps over the field.
    
    The key idea: GF(25) has 25 elements, but we need order 10.
    Instead, we use a known construction for order 10 MOS.
    
    Actually, order 10 is special: it's the smallest order for which
    existence of a complete set (9 MOLS) is known to be impossible
    (Euler's 36 officers problem shows no 2 MOLS of order 6, but order 10
    does have MOLS). In fact, it's known that there exist exactly 2 MOLS
    of order 10, and no more than 2 can exist... wait, let me reconsider.
    
    Actually, for order 10, it is known that there exist at least 2 MOLS.
    The maximum number of MOLS of order 10 is 2 (this is a known result
    related to the fact that 10 = 2*5, and the Bruck-Ryser-Chowla type
    constraints plus specific results).
    
    Let me use a concrete construction. One standard approach:
    
    Represent elements of Z_10 in a structured way. Since 10 = 2*5,
    we can use the Chinese Remainder Theorem to identify Z_10 with Z_2 x Z_5.
    
    A Latin square of order 10 can be constructed as follows:
    L(a,b) = (f(a,b), g(a,b)) where (a,b) in Z_2 x Z_5 and the result
    maps back to Z_10.
    
    For orthogonality, we need the pairs (L1[i,j], L2[i,j]) to be all distinct.
    
    Let me use a simpler, verified construction. I'll hardcode a known pair
    of orthogonal Latin squares of order 10, or construct them algorithmically.
    
    Construction via Z_2 x Z_5:
    Let the cells be indexed by (r, c) where r in Z_2, c in Z_5.
    Map (r, c) to index: idx = r*5 + c (so row-major with r as "super-row").
    
    Actually, let's index rows and columns by elements of Z_2 x Z_5.
    Row i: i = (r_i, c_i) where r_i = i // 5, c_i = i % 5
    Col j: j = (r_j, c_j) where r_j = j // 5, c_j = j % 5
    
    Square 1: S1(i, j) = (r_i + r_j mod 2, c_i + c_j mod 5) mapped to Z_10
    Square 2: S2(i, j) = (r_i + r_j mod 2, c_i + 2*c_j mod 5) mapped to Z_10
    
    Wait, this won't work directly because the row and column indices need
    to produce each symbol exactly once in each row and column.
    
    Let me think more carefully. A Latin square L of order 10 satisfies:
    - For fixed row i, {L[i,j] : j in 0..9} = {0,1,...,9}
    - For fixed col j, {L[i,j] : i in 0..9} = {0,1,...,9}
    
    Using the CRT identification Z_10 ~= Z_2 x Z_5:
    Let row index i = (a, b) where a = i % 2, b = i // 2... 
    
    Hmm, let me just use a known explicit construction.
    
    Here's a clean approach:
    Identify Z_10 with Z_2 x Z_5 via: x -> (x mod 2, x mod 5)
    Inverse: (a, b) -> the unique x in Z_10 with x ≡ a (mod 2) and x ≡ b (mod 5)
    
    For a Latin square, define L: (Z_2 x Z_5) x (Z_2 x Z_5) -> Z_2 x Z_5
    L((a,b), (c,d)) = (a + c + f(b,d) mod 2, b + d + g(b,d) mod 5)
    
    This is getting complex. Let me just use a verified hardcoded pair.
    
    I'll generate two orthogonal Latin squares of order 10 using a known method.
    """
    
    n = 10
    
    # Use the structure Z_10 = Z_2 x Z_5
    # Map index i in Z_10 to (i % 2, i // 2) -- wait, that's not CRT.
    # CRT: i -> (i mod 2, i mod 5)
    # Inverse: given (a, b), find x such that x ≡ a (mod 2), x ≡ b (mod 5)
    
    def crt(a, b):
        """Find x in Z_10 such that x ≡ a (mod 2) and x ≡ b (mod 5)."""
        # x = 5*a + 2*b - 10*(a*b) ... let me just compute
        # We need x = 5*a*k1 + 2*b*k2 where 5*k1 ≡ 1 (mod 2), 2*k2 ≡ 1 (mod 5)
        # 5 ≡ 1 (mod 2), so k1 = 1. 2*3 = 6 ≡ 1 (mod 5), so k2 = 3.
        # x = 5*a*1 + 2*b*3 = 5a + 6b mod 10
        return (5 * a + 6 * b) % 10
    
    def decompose(x):
        """Decompose x in Z_10 into (x mod 2, x mod 5)."""
        return (x % 2, x % 5)
    
    # Construct Latin square 1:
    # L1((a,b), (c,d)) = (a + c mod 2, b + d mod 5)
    # This is a group table for Z_2 x Z_5, which is a Latin square.
    
    # Construct Latin square 2:
    # L2((a,b), (c,d)) = (a + c mod 2, b + 2*d mod 5)
    # Check: for fixed row (a,b), as (c,d) ranges over Z_2 x Z_5, 
    #   second component: b + 2*d mod 5 ranges over all of Z_5 (since 2 is invertible mod 5)
    #   first component: a + c mod 2 ranges over all of Z_2
    #   So the pair (a+c mod 2, b+2d mod 5) ranges over all of Z_2 x Z_5. ✓
    # For fixed col (c,d), as (a,b) ranges:
    #   first: a + c mod 2 ranges over Z_2 ✓
    #   second: b + 2*d mod 5 ranges over Z_5 ✓
    #   So it's a Latin square. ✓
    
    # Orthogonality: for fixed (a,b), (c,d), the pair 
    #   (L1, L2) = ((a+c, b+d), (a+c, b+2d))
    #   The first components are equal, so we need (b+d, b+2d) to be distinct 
    #   for each (c,d). But wait, the first components of L1 and L2 are both a+c mod 2,
    #   so they're always equal. That means the pair is (x, y) where x = a+c mod 2 
    #   and y1 = b+d mod 5, y2 = b+2d mod 5.
    #   Actually, the ordered pair of symbols is (L1_val, L2_val) where each is in Z_10.
    #   L1_val = crt((a+c)%2, (b+d)%5)
    #   L2_val = crt((a+c)%2, (b+2*d)%5)
    #   For these to be distinct for each (c,d), we need: if (b+d mod 5, b+2d mod 5) 
    #   is the same for two different (c,d) pairs, then L1_val = L2_val.
    #   But (a+c mod 2) also varies with c. Let me check more carefully.
    
    # Two cells (a,b,c,d) and (a',b',c',d') give the same pair if:
    #   crt((a+c)%2, (b+d)%5) = crt((a'+c')%2, (b'+d')%5) AND
    #   crt((a+c)%2, (b+2d)%5) = crt((a'+c')%2, (b'+2d')%5)
    # Since crt is injective, this means:
    #   (a+c)%2 = (a'+c')%2, (b+d)%5 = (b'+d')%5, AND (b+2d)%5 = (b'+2d')%5
    # From the last two: d ≡ d' (mod 5) [since 2 is invertible mod 5]
    # Then b+d ≡ b'+d' implies b ≡ b' (mod 5)
    # And (a+c)%2 = (a'+c')%2
    # So (a,b) = (a',b') and c ≡ c' (mod 2) and d = d' (mod 5), so (c,d) = (c',d')
    # Hence the mapping is injective, so orthogonality holds. ✓
    
    squares = []
    
    for scale in [1, 2]:
        sq = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a, b = decompose(i)
                c, d = decompose(j)
                val = crt((a + c) % 2, (b + scale * d) % 5)
                sq[i][j] = val
        squares.append(sq)
    
    # Verify they are valid Latin squares
    for idx, sq in enumerate(squares):
        for i in range(n):
            row = set(sq[i])
            assert row == set(range(n)), f"Row {i} of square {idx} invalid: {sorted(row)}"
        for j in range(n):
            col = set(sq[i][j] for i in range(n))
            assert col == set(range(n)), f"Col {j} of square {idx} invalid: {sorted(col)}"
    
    # Verify orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (squares[0][i][j], squares[1][i][j])
            assert p not in pairs, f"Duplicate pair {p} at ({i},{j})"
            pairs.add(p)
    assert len(pairs) == n*n, f"Expected {n*n} pairs, got {len(pairs)}"
    
    return squares
