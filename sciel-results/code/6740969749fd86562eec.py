"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    if n != 10:
        # Fallback for other n, though task specifies 10
        return [[[(i+j)%n for j in range(n)] for i in range(n)]]
    
    # Construct 2 MOLS of order 10 using order 5 MOLS
    n5 = 5
    A5 = [[(i + j) % n5 for j in range(n5)] for i in range(n5)]
    B5 = [[(i + 2*j) % n5 for j in range(n5)] for i in range(n5)]
    
    S1 = [[0]*n for _ in range(n)]
    S2 = [[0]*n for _ in range(n)]
    
    for i in range(n5):
        for j in range(n5):
            a = A5[i][j]
            b = B5[i][j]
            
            # Block indices in 10x10
            r1 = 2*i
            r2 = 2*i + 1
            c1 = 2*j
            c2 = 2*j + 1
            
            # S1 mapping
            S1[r1][c1] = 2*a
            S1[r1][c2] = 2*a + 1
            S1[r2][c1] = 2*a + 1
            S1[r2][c2] = 2*a
            
            # S2 mapping (modified to ensure orthogonality)
            S2[r1][c1] = 2*b
            S2[r1][c2] = 2*b
            S2[r2][c1] = 2*b + 1
            S2[r2][c2] = 2*b + 1
            
    return [S1, S2]

# EVOLVE-BLOCK-END
