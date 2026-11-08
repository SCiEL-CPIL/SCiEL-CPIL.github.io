import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Attempt rank-22 via Kronecker-style block factorization
    # Treat multiplication as: C = [C_left, C_right] where C_left is 2x2 from cols 0,1 and C_right is 2x3 from cols 2,3,4
    # Use Strassen for the 2x2 part (rank 7) and handle the rest with efficient block handling
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: 2x2 multiplication (cols 0,1 of B) using Strassen
    # C[:,0:2] = A[:,0:2] * B[0:2,0:2]
    # Strassen for 2x2 matrices
    # Let A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # U indices: i*m + j where i is row of A, j is col of A (0,1)
    # V indices: j'*p + k where j' is row of B, k is col of B (0,1)
    # W indices: k*n + i where k is col of C, i is row of C
    
    # M1 = (a0+a1)(b0+b1)
    U[0*m+0][r] = 1
    U[1*m+1][r] = 1
    V[0*p+0][r] = 1
    V[1*p+1][r] = 1
    W[0*n+0][r] = 1
    W[1*n+1][r] = 1
    r += 1
    
    # M2 = (a2+a3)b0
    U[0*m+0][r] = 1
    U[1*m+1][r] = 1
    V[0*p+0][r] = 1
    W[0*n+1][r] = 1
    W[1*n+1][r] = -1
    r += 1
    
    # M3 = a0(b1-b3) -- wait, this doesn't match standard Strassen
    # Let me use the standard Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # I made errors above. Let me restart with correct Strassen.
    
    # Actually, let me just build this more carefully using the correct decomposition.
    # For now, let me use a simpler approach that's guaranteed correct.
    
    # Naive approach but try to optimize
    # Let me use a hybrid: Strassen for the 2x2 core, direct for the rest
    
    # Clear and restart
    for i in range(len(U)):
        for j in range(rank):
            U[i][j] = 0
    for i in range(len(V)):
        for j in range(rank):
            V[i][j] = 0
    for i in range(len(W)):
        for j in range(rank):
            W[i][j] = 0
    
    r = 0
    
    # Direct computation for all elements, but group them
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For 2x3x5, naive is 2*3*5 = 30
    # Can we do better?
    
    # Let's try: use the fact that we can compute some products once and reuse
    # Precompute: A[:,0]*B[0,k], A[:,1]*B[1,k], A[:,2]*B[2,k] for each k
    # That's still 3*5 = 15 products of 2x1 * 1x1 = 2 outputs each = 30 outputs
    # No gain from that perspective.
    
    # Let me try a different grouping:
    # For fixed k, C[:,k] = A * B[:,k] where B[:,k] is a 3x1 vector
    # This is a 2x3 times 3x1 = 2x1 matrix-vector product, which has rank 3 naively
    # So 5 such products = 15 rank-1 contributions to each column, but they're independent
    
    # Actually for a 2x3 x 3x1 multiplication, the tensor rank is exactly 3 (since the 
    # multiplication matrix has rank min(2,3,3)=... let me think)
    # The tensor for matvec (2x3) @ (3x1) is sum_j sum_i A[i,j] B[j] e_i e_k... no
    # T = sum_i sum_j sum_k A[i,j] B[j] C[i,k] but C is output, so
    # The tensor T_{i,j,k} for matvec A(2x3) * b(3x1) -> c(2x1) has T_{i,j} = A[i,j] and output c[i]
    # As a bilinear map, rank is at most 2*3 = 6, but actually it's min(n*p, m) or something
    # For 2x3 @ 3x1, the tensor is in U(2) * V(3) * W(2), and its rank is 3 (since we need 
    # to sum over the 3 middle indices, and the structure allows rank 3)
    
    # Hmm, this is getting complicated. Let me just provide a correct rank-30 solution and hope 
    # for the best, or find a known decomposition.
    
    # Known: rank of 2x3x5 multiplication tensor is at most 20 (from various bounds)
    # Let me try to construct a rank-20 or lower solution.
    
    # Actually, let me just use the naive decomposition which is guaranteed correct,
    # but I'll try rank 30 first to ensure validity, then optimize.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][r] = 1
                V[j*p + k][r] = 1
                W[k*n + i][r] = 1
                r += 1
    
    # We used rank 30
    rank = r
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
