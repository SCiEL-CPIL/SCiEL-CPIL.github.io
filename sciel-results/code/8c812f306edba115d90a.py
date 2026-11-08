"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Test with multiple random matrices
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Compute using decomposition
            C_comp = np.zeros((n, p))
            for r in range(rank):
                # X_r[i] = sum_j U[i*m+j, r] * A[i,j]
                X_r = np.array([sum(U[i*m+j, r] * A[i, j] for j in range(m)) for i in range(n)])
                # Y_r[k] = sum_j V[j*p+k, r] * B[j,k]
                Y_r = np.array([sum(V[j*p+k, r] * B[j, k] for j in range(m)) for k in range(p)])
                # C[i,k] += W[k*n+i, r] * X_r[i] * Y_r[k]
                for i in range(n):
                    for k in range(p):
                        C_comp[i, k] += W[k*n+i, r] * X_r[i] * Y_r[k]
            
            if not np.allclose(C_comp, C_true, atol=1e-10):
                return False
        
        return True
    
    # Strassen-based decomposition for 2x2x2 blocks
    # Each 2x2x2 multiplication has rank 7 (Strassen)
    # We decompose 2x4x8 into 8 independent 2x2x2 multiplications
    # Total rank: 8 * 7 = 56
    
    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2x2 matrix multiplication.
        C = A * B where A, B are 2x2, C is 2x2.
        Returns U, V, W matrices of shape (4, 7) each.
        """
        # Strassen's algorithm for 2x2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # Strassen's 7 products:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C[0,0] = M1 - M4 + M5 - M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # U[i*m+j, r] where i in {0,1}, j in {0,1}
        # V[j*p+k, r] where j in {0,1}, k in {0,1}
        # W[k*n+i, r] where k in {0,1}, i in {0,1}
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # U: a and d contribute, so U[0,0]=1 (a), U[1,0]=1 (d) [row index: i*m+j]
        # V: e and h contribute, so V[0,0]=1 (e), V[1,0]=1 (h) [row index: j*p+k]
        # W: contributes to C[0,0] with +1, C[1,1] with +1
        U[0, 0] = 1  # a
        U[1, 0] = 1  # d
        V[0, 0] = 1  # e
        V[1, 0] = 1  # h
        W[0, 0] = 1  # C[0,0] += M1
        W[3, 0] = 1  # C[1,1] += M1
        
        # M2 = (c+d)e
        U[2, 1] = 1  # c
        U[3, 1] = 1  # d
        V[0, 1] = 1  # e
        W[2, 1] = 1  # C[1,0] += M2
        W[3, 1] = -1  # C[1,1] -= M2
        
        # M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f
        V[0, 2] = -1  # -h... wait, h is at V[1,0]... 
        # Let me reconsider the indexing
        # V[j*p+k, r]: j is row of B, k is column of B
        # B = [[e, f], [g, h]]
        # So B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # V[0,0]=e, V[1,0]=f, V[2,0]=g, V[3,0]=h
        
        # Let me redo this more carefully
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = A*B
        # C[i,k] = sum_j A[i,j]*B[j,k]
        
        # U[i*m+j, r]: i=row of A, j=col of A
        # V[j*p+k, r]: j=row of B, k=col of B  
        # W[k*n+i, r]: k=col of C, i=row of C
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        # C00 += M1, C11 += M1
        W[0, 0] = 1  # C[0,0]
        W[3, 0] = 1  # C[1,1]
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        # C10 += M2, C11 -= M2
        W[2, 1] = 1  # C[1,0]
        W[3, 1] = -1  # C[1,1]
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1  # A00
        V[1, 2] = 1  # B01
        V[3, 2] = -1  # B11
        # C01 += M3, C11 += M3
        W[1, 2] = 1  # C[0,1]
        W[3, 2] = 1  # C[1,1]
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1  # A11
        V[2, 3] = 1  # B10
        V[0, 3] = -1  # B00
        # C00 -= M4, C10 += M4
        W[0, 3] = -1  # C[0,0]
        W[2, 3] = 1  # C[1,0]
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        # C00 += M5, C01 += M5
        W[0, 4] = 1  # C[0,0]
        W[1, 4] = 1  # C[0,1]
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1  # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        # C11 += M6
        W[3, 5] = 1  # C[1,1]
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1  # A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        # C00 -= M7
        W[0, 6] = -1  # C[0,0]
        
        return U, V, W
    
    # Verify Strassen decomposition
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Build full 2x4x8 decomposition using 8 Strassen blocks
    # For each k-pair (k1, k2), we have a 2x2x2 block
    # Actually, we need to decompose 2x4x8 into blocks
    
    # The matrix multiplication 2x4 * 4x8 = 2x8
    # We can split the 4 columns of A and 4 rows of B into pairs
    # A = [A_left | A_right] where A_left, A_right are 2x2
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x4
    
    # C = A_left * B_top + A_right * B_bottom
    
    # Each of these is a 2x2 * 2x4 multiplication
    # 2x2 * 2x4 can be done with 2 Strassen 2x2x2 blocks (one for each column of B)
    # So total: 4 Strassen blocks, each rank 7, total rank 28? That seems too good.
    
    # Wait, let me reconsider. The problem is 2x4 * 4x8.
    # We can view this as: for each output column k (0..7), C[:,k] = A * B[:,k]
    # where B[:,k] is a 4x1 vector.
    
    # Alternatively, split into 4 independent 2x2 * 2x2 problems?
    # A = [[A00, A01], [A10, A11]] (2x2) and [[A02, A03], [A12, A13]] (2x2)
    # B has 8 columns, so B = [B0 | B1 | B2 | B3 | B4 | B5 | B6 | B7]
    # Each Bk is 4x1, which can be split into [Bk_top; Bk_bottom] (2x1)
    
    # C[:,k] = A_left * Bk_top + A_right * Bk_bottom
    # This is 2x2 * 2x1 + 2x2 * 2x1 = 2x2 * 2x2 (concatenated)
    
    # So for each k, we have a 2x2x2 problem (treating Bk as 2x2 by duplicating)
    # No, that's not right either.
    
    # Let me think differently. The tensor is T_{i,j,k} = delta_{jk} for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    # Actually T_{i,j,k} = 1 if the (i,k) entry of A*B involves A[i,j]*B[j,k]
    
    # The standard approach: decompose into independent sub-problems.
    # 2x4 * 4x8 can be seen as 8 independent 2x4 * 4x1 multiplications.
    # Each 2x4 * 4x1 has rank at most 8 (schoolbook).
    
    # But we can do better by grouping. Let's use the fact that:
    # C = A * B where A is 2x4, B is 4x8.
    # Split A into two 2x2 blocks: A = [A1 | A2]
    # Split B into two 2x8 blocks: B = [B1; B2]
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is 2x2 * 2x8, which can be done with 4 Strassen 2x2x2 blocks.
    # Total: 8 Strassen blocks, rank 8*7 = 56.
    
    # Let's verify this construction
    def build_strassen_decomposition():
        """Build 2x4x8 decomposition using 8 Strassen 2x2x2 blocks."""
        rank = 56  # 8 blocks * 7 per block
        U = np.zeros((8, rank))  # n*m = 2*4 = 8
        V = np.zeros((32, rank))  # m*p = 4*8 = 32
        W = np.zeros((16, rank))  # p*n = 8*2 = 16
        
        # For each of the 8 blocks:
        # Block b (b=0..7) handles one column k=b of the output
        # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
        # B1 = B[0:2, b] (2x1), B2 = B[2:4, b] (2x1)
        # C[:,b] = A1*B1 + A2*B2
        
        # We can combine A1*B1 and A2*B2 into a single 2x2x2 problem:
        # Let A' = [A1 | A2] (2x4), B' = [B1; B2] (4x1)
        # C[:,b] = A' * B'
        
        # But this is still 2x4 * 4x1, not 2x2x2.
        
        # Alternative: Use 4 Strassen blocks per output column:
        # C[:,b] = A1*B1 + A2*B2
        # A1*B1 is 2x2 * 2x1, which is like half of a 2x2x2
        # We can pad to 2x2x2 by duplicating B1
        
        # Actually, let's just use the direct approach:
        # For each output column k, compute C[:,k] = A * B[:,k]
        # This is a 2x4 * 4x1 multiplication.
        # We can do this with rank 8 (schoolbook) or potentially less.
        
        # For now, let's use 7 Strassen-like blocks per 2x2x2 subproblem.
        # Split the 4x8 B matrix into 4 groups of 2 columns each.
        # For each group, we have a 2x4 * 4x2 multiplication.
        # 2x4 * 4x2 = (2x2*2x2 + 2x2*2x2 + 2x2*2x2 + 2x2*2x2) using Strassen
        # Each 2x2x2 has rank 7, so 4*7 = 28 per group, total 4*28 = 112. Too much.
        
        # Let me try a different approach. Just use schoolbook for now and return rank 64.
        # The Strassen approach needs more careful construction.
        
        return None
    
    # For now, return schoolbook as the safe fallback
    best_result = schoolbook()
    
    # Verify the schoolbook result
    if verify(best_result["U"], best_result["V"], best_result["W"]):
        return best_result
    
    # If verification fails (shouldn't happen), still return it
    return best_result
# EVOLVE-BLOCK-END
