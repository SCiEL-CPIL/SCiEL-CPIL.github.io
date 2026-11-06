def construct():
    n = 20
    m = 20
    
    # We will build a graph by partitioning vertices into two cliques of size 10.
    # Let A = {0, ..., 9} and B = {10, ..., 19}.
    # We will include all edges within A and all edges within B.
    # This gives 10*9/2 + 10*9/2 = 45 + 45 = 90 edges.
    
    # However, we must ensure that every 4-cycle is rainbow.
    # A 4-cycle can be:
    # 1. Four vertices all in A.
    # 2. Four vertices all in B.
    # 3. Two vertices in A, two in B. (No edges between A and B, so no such 4-cycles exist involving cross-edges).
    
    # So we only need to worry about 4-cycles entirely within A or entirely within B.
    # A 4-cycle in a clique K10 consists of 4 distinct vertices.
    # The edges of the 4-cycle are 4 distinct edges of the clique.
    # If we assign distinct colors to all edges in A, then any 4-cycle in A has 4 distinct colors.
    # Similarly for B.
    
    # But wait: we have only 20 colors.
    # The number of edges in K10 is 45.
    # We cannot assign distinct colors to all 45 edges of A because we only have 20 colors.
    # So we cannot make the entire K10 have distinct edge colors.
    
    # Let's reconsider.
    # If we use a complete graph K10, can we color its edges with 20 colors such that every 4-cycle is rainbow?
    # No, as reasoned before, if the graph is dense, constraints are tight.
    
    # Let's try a different structure.
    # What if we use a complete bipartite graph K10,10?
    # Edges = 100.
    # A 4-cycle in K10,10 involves 2 vertices from A and 2 from B.
    # Let the vertices be a1, a2 in A and b1, b2 in B.
    # The cycle is a1-b1-a2-b2-a1.
    # The edges are (a1,b1), (a1,b2), (a2,b1), (a2,b2).
    # We need c(a1,b1), c(a1,b2), c(a2,b1), c(a2,b2) to be distinct.
    
    # Can we color K10,10 with 20 colors such that for any a1 != a2 in A and b1 != b2 in B, the four colors c(a1,b1), c(a1,b2), c(a2,b1), c(a2,b2) are distinct?
    # This is equivalent to saying that the color matrix M (10x10) has the property that every 2x2 submatrix has 4 distinct entries.
    
    # This is a known problem: constructing a 10x10 array with 20 symbols such that every 2x2 submatrix is a Latin square (i.e., has 4 distinct symbols).
    # Actually, it's stronger: the 4 symbols must be distinct.
    
    # A simple construction:
    # Let M[i][j] = (i + j) % 20?
    # Check 2x2 submatrix for i, i+1 and j, j+1:
    # M[i][j] = i+j
    # M[i][j+1] = i+j+1
    # M[i+1][j] = i+j+1
    # M[i+1][j+1] = i+j+2
    # We have two equal entries: M[i][j+1] == M[i+1][j].
    # So this doesn't work.
    
    # We need a 2D array where no two entries in the same row or column are equal? No, that's a Latin square, which ensures distinct in rows/cols but not necessarily 2x2 distinctness.
    # Actually, if every row is a permutation of 0..9 and every column is a permutation of 0..9, does it ensure 2x2 distinctness?
    # No. Example:
    # 1 2
    # 2 1
    # Rows are permutations, cols are permutations, but 1 and 2 repeat.
    
    # We need a "2-rainbow" property.
    # It is known that such arrays exist if the number of symbols is large enough.
    # With 20 symbols and 10x10, it should be possible.
    
    # Let's try a construction based on orthogonal Latin squares.
    # If we have two orthogonal Latin squares L1 and L2 of order 10, we can form pairs (L1[i][j], L2[i][j]).
    # But we need single colors, not pairs.
    
    # Alternative:
    # Use colors 0..19.
    # Let M[i][j] = (i * 10 + j) % 20?
    # M[0][0]=0, M[0][1]=1, M[1][0]=10, M[1][1]=11.
    # 2x2 submatrix at (0,0): 0,1,10,11 -> distinct.
    # M[0][1]=1, M[0][2]=2, M[1][1]=11, M[1][2]=12 -> distinct.
    # Let's check if any 2x2 submatrix has duplicates.
    # M[i][j] = (10*i + j) % 20.
    # Since i is in 0..9, 10*i is 0,10,20,30...
    # 10*i % 20 is 0 if i even, 10 if i odd.
    # So M[i][j] = (10*(i%2) + j) % 20.
    # If i is even: M[i][j] = j.
    # If i is odd: M[i][j] = 10 + j.
    
    # Consider i=0, i+1=1, j=0, j+1=1.
    # M[0][0]=0, M[0][1]=1, M[1][0]=10, M[1][1]=11. Distinct.
    # Consider i=0, i+1=1, j=1, j+1=2.
    # M[0][1]=1, M[0][2]=2, M[1][1]=11, M[1][2]=12. Distinct.
    
    # What if i and i+1 have the same parity? Impossible, they are consecutive.
    # So one is even, one is odd.
    # Let i be even, i+1 be odd.
    # M[i][j] = j
    # M[i][j+1] = j+1
    # M[i+1][j] = 10+j
    # M[i+1][j+1] = 10+j+1
    # These are 4 distinct numbers modulo 20?
    # j, j+1, 10+j, 10+j+1.
    # Since j is in 0..8 (as j+1<=9), j+1 <= 9.
    # 10+j is in 10..18.
    # 10+j+1 is in 11..19.
    # The sets {j, j+1} and {10+j, 10+j+1} are disjoint because {0..9} and {10..19} are disjoint.
    # Within {j, j+1}, they are distinct.
    # Within {10+j, 10+j+1}, they are distinct.
    # So yes, all 4 are distinct.
    
    # Similarly if i is odd, i+1 is even.
    # M[i][j] = 10+j
    # M[i][j+1] = 10+j+1
    # M[i+1][j] = j
    # M[i+1][j+1] = j+1
    # Same argument.
    
    # So the coloring M[i][j] = (10*i + j) % 20 works for K10,10.
    # Wait, is it a simple graph? Yes.
    # Number of edges: 100.
    # This is much better than 55.
    
    # Let's double check if there are any other 4-cycles.
    # In K10,10, any 4-cycle must alternate between A and B.
    # So it must involve 2 vertices from A and 2 from B.
    # We proved that any such 4-cycle has distinct colors.
    # Are there 4-cycles with more than 2 vertices from A? No, because there are no edges within A.
    
    # So the construction is valid.
    # Edges: all (a, b) for a in 0..9, b in 10..19.
    # Color: c = (10 * a + (b - 10)) % 20.
    # Let's map b to j in 0..9: j = b - 10.
    # c = (10 * a + j) % 20.
    
    edges = []
    for a in range(10):
        for b in range(10, 20):
            j = b - 10
            c = (10 * a + j) % 20
            edges.append([a, b, c])
            
    return edges
