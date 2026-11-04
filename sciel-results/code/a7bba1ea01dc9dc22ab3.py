"""Matrix multiplication 2x3x6 decomposition using Strassen blocks."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 140.0
    n, m, p = 2, 3, 6
    
    # Try to build a rank-33 decomposition
    # Strategy: Use Strassen on 2x2x2 blocks
    # The 2x3 to 2x6 multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can split columns into pairs: (0,1), (2,3), (4,5)
    # For each pair (2k, 2k+1), we have a 2x3 -> 2x2 multiplication
    # Split each 2x2 block of A into four 1x1 blocks
    # Apply Strassen to each 2x3->2x2 as a composition
    
    # Actually, let's use a cleaner approach:
    # Split A into A1 (cols 0) and A2 (cols 1,2)
    # A1 is 2x1, A2 is 2x2
    # B columns 0,1,2 form 3x2, and cols 3,4,5 form 3x2, cols... 
    # No, B is 3x6 so it's 3x2 three times.
    
    # Let me use the direct Strassen block approach:
    # Rank 33 = use Strassen decomposition for 2x2x2 subproblems
    
    # Alternative clean approach:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) - A[i,1]*B[0,k] + A[i,1]*B[1,k] ... 
    # This is getting complex. Let me just use the verified rank-33 approach.
    
    # Simple verified approach: rank 27 decomposition
    # Split B into three 3x2 blocks: B1 (cols 0,1), B2 (cols 2,3), B3 (cols 4,5)
    # For each block, compute A(2x3) * B_block(3x2) 
    # Each 2x3*3x2 = 2x2 result can be done with rank at most 2*2*3=12 schoolbook
    # But we can do better.
    
    # Let's use the approach where we express:
    # For 2x3 * 3x2 -> 2x2:
    # C[i,j] = sum_{k=0}^{2} A[i,k] B[k,j]
    # = A[i,0]B[0,j] + A[i,1]B[1,j] + A[i,2]B[2,j]
    # This is rank 4 per block (for each j independently it's rank 3, but coupled)
    # Actually for a fixed j, C[:,j] = A * B[:,j] which is a 2-vector
    # C[:,j] = A0*B0j + A1*B1j + A2*B2j where Ai is column i of A
    # So each column of C uses 3 multiplications, total 6 for 2 columns.
    # Three blocks = 18. That's better than 36!
    
    # Wait, but that's still schoolbook. Let me count:
    # For B_block (3x2), compute A(2x3) * B_block(3x2):
    # Result[i,j] = A[i,0]*Bb[0,j] + A[i,1]*Bb[1,j] + A[i,2]*Bb[2,j] for i,j in {0,1}
    # That's 2*2*3 = 12 terms per block, 3 blocks = 36. Same as schoolbook.
    
    # To beat 36, we need to exploit structure. Let's use Strassen on the 2x2 part.
    # View A as [A_col0 | A_2x2] where A_2x2 is columns 1,2
    # View B_block as [B_row0; B_2x2] where B_row0 is row 0, B_2x2 is rows 1,2
    # Then A * B_block = A_col0 * B_row0 + A_2x2 * B_2x2 (outer product + 2x2 matmul)
    # Outer product: rank 4 (2*2 terms, but actually rank 2*1*1=2? No, outer product of 2-vec and 2-vec is rank 4 as a bilinear form? No, it's 2*2=4 scalar mults)
    # Actually outer product: (a0*b0, a0*b1; a1*b0, a1*b1) = 4 multiplications, rank 4
    # 2x2 * 2x2 = Strassen rank 7
    # So each block: rank 4 + 7 = 11? But the outer product and Strassen are independent.
    # Total: 3 blocks * 11 = 33. That matches rank 33!
    
    # Let's implement this:
    # For each output column pair (cols 2t, 2t+1 of C):
    #   Let Bb be B[:, 2t:2t+2] (3x2)
    #   A_col0 = A[:,0] (2x1)
    #   A_2x2 = A[:,1:3] (2x2)
    #   B_row0 = Bb[0,:] (1x2)
    #   B_2x2 = Bb[1:3,:] (2x2)
    #   C_block = A_col0 * B_row0 + A_2x2 * B_2x2
    #   First term: 4 scalar mults (outer product)
    #   Second term: 7 Strassen mults
    #   Total per block: 11
    #   Total: 33
    
    def verify(U, V, W):
        """Verify that sum_t U[:,t] outer V[:,t] outer W[:,t] equals the matmul tensor."""
        # Check a few random cases
        import random
        for _ in range(20):
            A = [[random.randint(-5,5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5,5) for _ in range(p)] for _ in range(m)]
            C_true = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            C_comp = [[0]*p for _ in range(n)]
            rank = len(U[0])
            for t in range(rank):
                Ucol = [U[i*m+j][t] for i in range(n) for j in range(m)]
                Vcol = [V[j*p+k][t] for j in range(m) for k in range(p)]
                Wcol = [W[k*n+i][t] for k in range(p) for i in range(n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_comp[i][k] += Ucol[i*m+j] * Vcol[j*p+k] * Wcol[k*n+i]
            if C_comp != C_true:
                return False
        return True
    
    # Build rank-33 decomposition
    rank = 33
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for t in range(3):  # three column blocks
        # Bb = B[:, 2t:2t+2]
        # Term 1: Outer product A_col0 * B_row0 (4 terms)
        for i in range(2):  # A[i,0]
            for kt in range(2):  # B[0, 2t+kt]
                U[i*m + 0][col] = 1
                V[0*p + (2*t+kt)][col] = 1
                W[(2*t+kt)*n + i][col] = 1
                col += 1
        
        # Term 2: Strassen on A_2x2 * B_2x2
        # A_2x2 = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
        # B_2x2 = [[B[1,2t], B[1,2t+1]], [B[2,2t], B[2,2t+1]]]
        # Strassen: 7 terms
        # Let a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
        # Let e=B[1,2t], f=B[1,2t+1], g=B[2,2t], h=B[2,2t+1]
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = (a-b)(f+h)
        # p4 = (c-a)(e+f)
        # p5 = (b-d)(g+h)
        # p6 = (a+c)g
        # p7 = (b+c)h
        # C00 = p1+p4+p5-p7
        # C01 = p1+p3-p5+p7
        # C10 = p2+p4+p7
        # C11 = p1-p2+p3+p6
        
        # p1 = (a+d)(e+h)
        U[0*m+1][col] = 1  # a
        U[1*m+2][col] = 1  # d
        V[1*p+(2*t)][col] = 1    # e
        V[1*p+(2*t+1)][col] = 1  # h
        W[(2*t)*n+0][col] = 1       # C00
        W[(2*t+1)*n+0][col] = 1     # C01
        W[(2*t)*n+1][col] = 1       # C10
        W[(2*t+1)*n+1][col] = -1    # C11
        col += 1
        
        # p2 = (c+d)e
        U[1*m+1][col] = 1  # c
        U[1*m+2][col] = 1  # d
        V[1*p+(2*t)][col] = 1  # e
        W[(2*t)*n+1][col] = 1   # C10
        W[(2*t+1)*n+1][col] = -1  # C11
        col += 1
        
        # p3 = (a-b)(f+h)
        U[0*m+1][col] = 1  # a
        U[0*m+2][col] = -1 # -b
        V[1*p+(2*t+1)][col] = 1  # f
        V[2*p+(2*t+1)][col] = 1  # h
        W[(2*t+1)*n+0][col] = 1   # C01
        W[(2*t+1)*n+1][col] = 1   # C11
        col += 1
        
        # p4 = (c-a)(e+f)
        U[1*m+1][col] = 1   # c
        U[0*m+1][col] = -1  # -a
        V[1*p+(2*t)][col] = 1    # e
        V[1*p+(2*t+1)][col] = 1  # f
        W[(2*t)*n+0][col] = 1    # C00
        W[(2*t)*n+1][col] = 1    # C10
        col += 1
        
        # p5 = (b-d)(g+h)
        U[0*m+2][col] = 1   # b
        U[1*m+2][col] = -1  # -d
        V[2*p+(2*t)][col] = 1    # g
        V[2*p+(2*t+1)][col] = 1  # h
        W[(2*t)*n+0][col] = 1       # C00
        W[(2*t+1)*n+0][col] = -1    # C01
        col += 1
        
        # p6 = (a+c)g
        U[0*m+1][col] = 1  # a
        U[1*m+1][col] = 1  # c
        V[2*p+(2*t)][col] = 1  # g
        W[(2*t+1)*n+1][col] = 1  # C11
        col += 1
        
        # p7 = (b+c)h
        U[0*m+2][col] = 1  # b
        U[1*m+1][col] = 1  # c
        V[2*p+(2*t+1)][col] = 1  # h
        W[(2*t+1)*n+0][col] = 1   # C01
        W[(2*t)*n+1][col] = 1     # C10
        W[(2*t)*n+0][col] = -1    # C00
        col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 36
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
