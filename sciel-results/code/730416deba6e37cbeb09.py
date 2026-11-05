import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 115  # Safety margin before 120s limit
    
    # Index mapping:
    # A[i,j] -> index i*m + j (0..8)
    # B[j,k] -> index j*p + k (0..11)
    # C[i,k] -> index k*n + i (0..11)
    
    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Check valid indices (where T should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    if not np.isclose(T_approx[alpha, beta, gamma], 1.0, atol=atol):
                        return False
        
        # Check invalid indices (where T should be 0)
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    valid_set.add((alpha, beta, gamma))
        
        for alpha in range(n * m):
            for beta in range(m * p):
                for gamma in range(p * n):
                    if (alpha, beta, gamma) not in valid_set:
                        if not np.isclose(T_approx[alpha, beta, gamma], 0.0, atol=atol):
                            return False
        return True
    
    # Try to construct a rank-21 decomposition using block structure
    def try_rank_21():
        rank = 21
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Use Strassen-like decomposition for 2x2 blocks
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (7 terms)
        # A indices: 0,1,3,4 (A00,A01,A10,A11)
        # B indices: 0,1,4,5 (B00,B01,B10,B11)
        # C indices: 0,3,1,4 (C00,C01,C10,C11)
        
        # T1: (A00+A11)(B00+B10) -> C00, C11
        U[0, 0] += 1; U[4, 0] += 1
        V[0, 0] += 1; V[4, 0] += 1
        W[0, 0] += 1; W[4, 0] += 1
        
        # T2: (A10+A11)B00 -> C10, -C11
        U[3, 1] += 1; U[4, 1] += 1
        V[0, 1] += 1
        W[1, 1] += 1; W[4, 1] += -1
        
        # T3: A00(B01-B11) -> C01, C11
        U[0, 2] += 1
        V[1, 2] += 1; V[5, 2] += -1
        W[3, 2] += 1; W[4, 2] += 1
        
        # T4: A11(B10-B00) -> C00, C10
        U[4, 3] += 1
        V[4, 3] += 1; V[0, 3] += -1
        W[0, 3] += 1; W[1, 3] += 1
        
        # T5: (A00+A01)B11 -> C01, -C00
        U[0, 4] += 1; U[1, 4] += 1
        V[5, 4] += 1
        W[3, 4] += 1; W[0, 4] += -1
        
        # T6: (A10-A00)(B00+B01) -> C11
        U[3, 5] += 1; U[0, 5] += -1
        V[0, 5] += 1; V[1, 5] += 1
        W[4, 5] += 1
        
        # T7: (A01-A11)(B10+B11) -> C00
        U[1, 6] += 1; U[4, 6] += -1
        V[4, 6] += 1; V[5, 6] += 1
        W[0, 6] += 1
        
        # Now handle the remaining elements involving index 2 in A and B
        # A[2,:] = A20,A21,A22 (indices 6,7,8)
        # A[:,2] = A02,A12,A22 (indices 2,5,8)
        # B[2,:] = B20,B21,B22,B23 (indices 8,9,10,11)
        # B[:,2] = B02,B12,B22 (indices 2,6,10)
        # B[:,3] = B03,B13,B23 (indices 3,7,11)
        # C[2,:] = C20,C21,C22,C23 (indices 2,5,8,11)
        # C[:,2] = C02,C12,C22 (indices 6,7,8)
        # C[:,3] = C03,C13,C23 (indices 9,10,11)
        
        # We need 14 more terms to cover the rest
        # Use direct multiplication for the remaining 14 elements
        
        # T8: A02*B00 -> C02
        U[2, 7] += 1; V[0, 7] += 1; W[6, 7] += 1
        
        # T9: A02*B01 -> C03? No, C01
        # Wait, C[i,k] = sum_j A[i,j]*B[j,k]
        # A02 * B01 -> C0,1? No, A[0,2]*B[2,1] but B[2,1] doesn't exist in our block
        # Actually B has 4 columns, so B[j,k] for j in 0..2, k in 0..3
        
        # Let me reconsider. The remaining terms should be:
        # For i in 0..2, k in 0..3, we need to cover all A[i,j]*B[j,k] terms
        # that aren't already covered by the 7 Strassen terms above.
        
        # The Strassen terms above cover:
        # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        
        # Remaining:
        # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        # A[0:2,2] * B[0:2,0:4] -> C[0:2,0:4]
        # A[2,0:2] * B[0:2,0:4] -> C[2,0:4]
        # A[2,2] * B[2,0:4] -> C[2,0:4]
        # A[2,2] * B[0:2,0:4] -> C[2,0:4]
        
        # This is getting complex. Let me just use direct terms for the rest.
        
        # T8: A02*B00 -> C00
        U[2, 7] += 1; V[0, 7] += 1; W[0, 7] += 1
        
        # T9: A02*B01 -> C01
        U[2, 8] += 1; V[1, 8] += 1; W[3, 8] += 1
        
        # T10: A02*B02 -> C02
        U[2, 9] += 1; V[2, 9] += 1; W[6, 9] += 1
        
        # T11: A02*B03 -> C03
        U[2, 10] += 1; V[3, 10] += 1; W[9, 10] += 1
        
        # T12: A12*B00 -> C10
        U[5, 11] += 1; V[0, 11] += 1; W[1, 11] += 1
        
        # T13: A12*B01 -> C11
        U[5, 12] += 1; V[1, 12] += 1; W[4, 12] += 1
        
        # T14: A12*B02 -> C12
        U[5, 13] += 1; V[2, 13] += 1; W[7, 13] += 1
        
        # T15: A12*B03 -> C13
        U[5, 14] += 1; V[3, 14] += 1; W[10, 14] += 1
        
        # T16: A20*B00 -> C20
        U[6, 15] += 1; V[0, 15] += 1; W[2, 15] += 1
        
        # T17: A20*B01 -> C21
        U[6, 16] += 1; V[1, 16] += 1; W[5, 16] += 1
        
        # T18: A20*B02 -> C22
        U[6, 17] += 1; V[2, 17] += 1; W[8, 17] += 1
        
        # T19: A20*B03 -> C23
        U[6, 18] += 1; V[3, 18] += 1; W[11, 18] += 1
        
        # T20: A21*B00 -> C20 (duplicate with T16? No, A21 is different from A20)
        # A21 is index 7
        U[7, 19] += 1; V[0, 19] += 1; W[2, 19] += 1
        
        # T21: A21*B01 -> C21
        U[7, 20] += 1; V[1, 20] += 1; W[5, 20] += 1
        
        # Check if this works
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # Try rank-21 first
    result = try_rank_21()
    if result is not None:
        return result
    
    # Fallback: use the known rank-34 solution from the current program
    def try_rank_34():
        rank = 34
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        # Part 1: A_TL (2x2) * B_top_L (2x2) -> C_top_L (2x2) - Strassen
        U[0, col] += 1; U[4, col] += 1
        V[0, col] += 1; V[4, col] += 1
        W[0, col] += 1; W[4, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[0, col] += 1
        W[1, col] += 1; W[4, col] += -1
        col += 1
        
        U[0, col] += 1
        V[1, col] += 1; V[5, col] += -1
        W[3, col] += 1; W[4, col] += 1
        col += 1
        
        U[4, col] += 1
        V[4, col] += 1; V[0, col] += -1
        W[0, col] += 1; W[1, col] += 1
        col += 1
        
        U[0, col] += 1; U[1, col] += 1
        V[5, col] += 1
        W[3, col] += 1; W[0, col] += -1
        col += 1
        
        U[3, col] += 1; U[0, col] += -1
        V[0, col] += 1; V[1, col] += 1
        W[4, col] += 1
        col += 1
        
        U[1, col] += 1; U[4, col] += -1
        V[4, col] += 1; V[5, col] += 1
        W[0, col] += 1
        col += 1
        
        # Part 2: A_TL (2x2) * B_top_R (2x2) -> C_top_R (2x2) - Strassen
        U[0, col] += 1; U[4, col] += 1
        V[2, col] += 1; V[6, col] += 1
        W[6, col] += 1; W[9, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[2, col] += 1
        W[7, col] += 1; W[10, col] += -1
        col += 1
        
        U[0, col] += 1
        V[3, col] += 1; V[6, col] += -1
        W[9, col] += 1; W[10, col] += 1
        col += 1
        
        U[4, col] += 1
        V[6, col] += 1; V[2, col] += -1
        W[6, col] += 1; W[7, col] += 1
        col += 1
        
        U[0, col] += 1; U[1, col] += 1
        V[6, col] += 1
        W[9, col] += 1; W[6, col] += -1
        col += 1
        
        U[3, col] += 1; U[0, col] += -1
        V[2, col] += 1; V[3, col] += 1
        W[10, col] += 1
        col += 1
        
        U[1, col] += 1; U[4, col] += -1
        V[6, col] += 1; V[7, col] += 1
        W[6, col] += 1
        col += 1
        
        # Part 3: A_TR (2x1) * B_bot (1x4)
        U[2, col] += 1; V[8, col] += 1; W[0, col] += 1; col += 1
        U[2, col] += 1; V[9, col] += 1; W[3, col] += 1; col += 1
        U[2, col] += 1; V[10, col] += 1; W[6, col] += 1; col += 1
        U[2, col] += 1; V[11, col] += 1; W[9, col] += 1; col += 1
        
        U[5, col] += 1; V[8, col] += 1; W[1, col] += 1; col += 1
        U[5, col] += 1; V[9, col] += 1; W[4, col] += 1; col += 1
        U[5, col] += 1; V[10, col] += 1; W[7, col] += 1; col += 1
        U[5, col] += 1; V[11, col] += 1; W[10, col] += 1; col += 1
        
        # Part 4: A_BL (1x2) * B_top (2x4)
        U[6, col] += 1; V[0, col] += 1; W[2, col] += 1; col += 1
        U[7, col] += 1; V[4, col] += 1; W[2, col] += 1; col += 1
        
        U[6, col] += 1; V[1, col] += 1; W[5, col] += 1; col += 1
        U[7, col] += 1; V[5, col] += 1; W[5, col] += 1; col += 1
        
        U[6, col] += 1; V[2, col] += 1; W[8, col] += 1; col += 1
        U[7, col] += 1; V[6, col] += 1; W[8, col] += 1; col += 1
        
        U[6, col] += 1; V[3, col] += 1; W[11, col] += 1; col += 1
        U[7, col] += 1; V[7, col] += 1; W[11, col] += 1; col += 1
        
        # Part 5: A_BR (1x1) * B_bot (1x4)
        U[8, col] += 1; V[8, col] += 1; W[2, col] += 1; col += 1
        U[8, col] += 1; V[9, col] += 1; W[5, col] += 1; col += 1
        U[8, col] += 1; V[10, col] += 1; W[8, col] += 1; col += 1
        U[8, col] += 1; V[11, col] += 1; W[11, col] += 1; col += 1
        
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    result = try_rank_34()
    if result is not None:
        return result
    
    # Final fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
