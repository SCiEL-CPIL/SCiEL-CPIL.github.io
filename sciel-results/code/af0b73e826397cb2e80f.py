"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Strategy: Decompose p=9 as 4+4+1
    # Part 1: 4x4 * 4x4 using Strassen (rank 49)
    # Part 2: 4x4 * 4x4 using Strassen (rank 49)  
    # Part 3: 4x4 * 4x1 using schoolbook (rank 16)
    # Total rank: 114
    
    total_rank = 114
    U = [[0] * total_rank for _ in range(n * m)]  # 16 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 36 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 36 rows
    
    # Define the 7 Strassen rank-1 tensors for 2x2x2
    # For 2x2 matrices A=[[a0,a1],[a2,a3]], B=[[b0,b1],[b2,b3]], C=A*B
    # p1 = (a0+a3)(b0+b3) -> c00 += 1, c11 += 1
    # p2 = (a2+a3)b0     -> c10 += 1, c11 += -1
    # p3 = a0(b1-b3)     -> c01 += 1, c11 += 1
    # p4 = a3(b2-b0)     -> c00 += 1, c10 += 1
    # p5 = (a0+a1)b3     -> c01 += 1, c00 += -1
    # p6 = (a2-a0)(b0+b1)-> c11 += 1
    # p7 = (a1-a3)(b2+b3)-> c00 += 1, c01 += 1
    
    def add_strassen_2x2(ar, ac, br, bc, cr, cc, offset):
        """Add 7 Strassen terms for 2x2 block at A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2] -> C[cr:cr+2, cc:cc+2]"""
        # p1 = (a0+a3)(b0+b3)
        c = offset
        U[ar*m + ac][c] = 1; U[(ar+1)*m + (ac+1)][c] = 1
        V[br*p + bc][c] = 1; V[(br+1)*p + (bc+1)][c] = 1
        W[cr*n + cc][c] = 1; W[(cr+1)*n + (cc+1)][c] = 1
        
        # p2 = (a2+a3)b0
        c = offset + 1
        U[(ar+1)*m + ac][c] = 1; U[(ar+1)*m + (ac+1)][c] = 1
        V[br*p + bc][c] = 1
        W[(cr+1)*n + cc][c] = 1; W[(cr+1)*n + (cc+1)][c] = -1
        
        # p3 = a0(b1-b3)
        c = offset + 2
        U[ar*m + ac][c] = 1
        V[br*p + (bc+1)][c] = 1; V[(br+1)*p + (bc+1)][c] = -1
        W[cr*n + (cc+1)][c] = 1; W[(cr+1)*n + (cc+1)][c] = 1
        
        # p4 = a3(b2-b0)
        c = offset + 3
        U[(ar+1)*m + (ac+1)][c] = 1
        V[(br+1)*p + bc][c] = 1; V[br*p + bc][c] = -1
        W[cr*n + cc][c] = 1; W[(cr+1)*n + cc][c] = 1
        
        # p5 = (a0+a1)b3
        c = offset + 4
        U[ar*m + ac][c] = 1; U[ar*m + (ac+1)][c] = 1
        V[(br+1)*p + (bc+1)][c] = 1
        W[cr*n + (cc+1)][c] = 1; W[cr*n + cc][c] = -1
        
        # p6 = (a2-a0)(b0+b1)
        c = offset + 5
        U[(ar+1)*m + ac][c] = 1; U[ar*m + ac][c] = -1
        V[br*p + bc][c] = 1; V[br*p + (bc+1)][c] = 1
        W[(cr+1)*n + (cc+1)][c] = 1
        
        # p7 = (a1-a3)(b2+b3)
        c = offset + 6
        U[ar*m + (ac+1)][c] = 1; U[(ar+1)*m + (ac+1)][c] = -1
        V[(br+1)*p + bc][c] = 1; V[(br+1)*p + (bc+1)][c] = 1
        W[cr*n + cc][c] = 1; W[cr*n + (cc+1)][c] = 1
        
        return offset + 7
    
    # Part 1: C[:, 0:4] = A * B[:, 0:4] (4x4 * 4x4)
    # Use recursive Strassen: split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # For 4x4x4, use the 7 Strassen multiplications on 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 * 2x2 multiplication, so each has rank 7 via Strassen
    # Total: 7 * 7 = 49 terms
    
    col = 0
    
    # M1 = (A11+A22)(B11+B22)
    # This is sum of 4 products: A11*B11 + A11*B22 + A22*B11 + A22*B22
    # We can't directly apply Strassen to sums. Instead, we compute each of the
    # 7 Strassen products for the 4x4 case by expanding.
    
    # Actually, let me just implement the 49-term decomposition directly.
    # For 4x4x4 with Strassen, we have 7 "multiplications" each of which is
    # a 2x2x2 multiplication (rank 7), giving 49 total rank-1 terms.
    
    # Let me define the 7 block-level Strassen products and expand each.
    
    # Block indices:
    # A11: ar=0,ac=0; A12: ar=0,ac=2; A21: ar=2,ac=0; A22: ar=2,ac=2
    # B11: br=0,bc=0; B12: br=0,bc=2; B21: br=2,bc=0; B22: br=2,bc=2
    # C11: cr=0,cc=0; C12: cr=0,cc=2; C21: cr=2,cc=0; C22: cr=2,cc=2
    
    # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # We need to add all 4 products, each with 7 Strassen terms = 28 terms
    # But wait, that's not right. M1 is ONE multiplication of two 2x2 matrices
    # where the matrices are sums. We can't split it into 4 separate Strassen
    # computations.
    
    # Let me reconsider. The correct approach:
    # For 4x4x4 Strassen, we have 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # The 7 multiplications are:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # For each Mi, we apply Strassen to the 2x2 matrices.
    # But the matrices in Mi may be sums/differences of blocks.
    # We need to express each 2x2 matrix in Mi as linear combinations of
    # the original 4x4 matrix entries.
    
    # Let me define helper functions to get the entries of block sums/diffs.
    
    # For M1: X = A11+A22, Y = B11+B22
    # X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3]
    # X[1,0] = A[1,0]+A[3,2], X[1,1] = A[1,1]+A[3,3]
    # Y[0,0] = B[0,0]+B[2,0], Y[0,1] = B[0,1]+B[2,1]
    # Y[1,0] = B[1,0]+B[3,0], Y[1,1] = B[1,1]+B[3,1]
    
    # This is getting complex. Let me just use the schoolbook for 4x4x4 (rank 64)
    # and schoolbook for 4x4x1 (rank 16), giving total rank 64+64+16 = 144.
    # That's the same as the original, so no improvement.
    
    # Let me try a different approach: use rank-28 for 4x4x4 by direct
    # block decomposition (not recursive Strassen).
    
    # Reset with rank 28+28+16 = 72
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: C[:, 0:4] = A * B[:, 0:4]
    # Use 2x2 block decomposition with Strassen for each block product
    # C11 = A11*B11 + A12*B21 (2 products, 14 terms)
    # C12 = A11*B12 + A12*B22 (2 products, 14 terms)
    # C21 = A21*B11 + A22*B21 (2 products, 14 terms)
    # C22 = A21*B12 + A22*B22 (2 products, 14 terms)
    # Total: 56 terms
    
    # Hmm, that's 56, not 28. Let me just use 56 terms.
    
    # Actually, I realize I should just implement a working solution first.
    # Let me use schoolbook for everything (rank 144) to ensure correctness,
    # then optimize.
    
    total_rank = n * m * p  # 144
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
