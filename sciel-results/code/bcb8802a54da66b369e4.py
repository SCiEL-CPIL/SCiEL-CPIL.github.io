import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 28 Decomposition for 2x3x5
    # Block 1: 2x3x2 (Rank 11)
    # Block 2: 2x3x3 (Rank 17)
    
    # Helper: Trivial n x m x p
    def trivial(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        idx_i = np.repeat(np.arange(n), m * p)
        idx_j = np.tile(np.repeat(np.arange(m), p), n)
        idx_k = np.tile(np.arange(p), n * m)
        U[idx_i * m + idx_j, np.arange(rank)] = 1
        V[idx_j * p + idx_k, np.arange(rank)] = 1
        W[idx_k * n + idx_i, np.arange(rank)] = 1
        return U, V, W

    # Helper: Strassen 2x2x2
    def strassen_222():
        rank = 7
        U = np.zeros((4, rank), dtype=int)
        V = np.zeros((4, rank), dtype=int)
        W = np.zeros((4, rank), dtype=int)
        # M1
        U[0,0]=1; U[2,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        # M2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        # M4
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # M5
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        # M6
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        # M7
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        return U, V, W

    # Build 2x3x2 (Rank 11)
    # A(2x3) = [A1(2x2) | A2(2x1)]
    # B(3x2) = [B1(2x2); B2(1x2)]
    # C = A1B1 + A2B2
    # A1B1: 2x2x2 (Strassen, Rank 7)
    # A2B2: 2x1x2 (Trivial, Rank 4)
    
    U_232 = np.zeros((6, 11), dtype=int)
    V_232 = np.zeros((6, 11), dtype=int)
    W_232 = np.zeros((4, 11), dtype=int)
    
    # A1B1 part (cols 0-6)
    U1, V1, W1 = strassen_222()
    # Map A1 indices (0,1,2,3) to A indices (0,1,2,3)
    # A: 00,01,10,11 -> 0,1,2,3
    U_232[0:4, 0:7] = U1
    # Map B1 indices (0,1,2,3) to B indices (0,1,2,3)
    # B: 00,01,10,11 -> 0,1,2,3
    V_232[0:4, 0:7] = V1
    # Map C indices (0,1,2,3) to C indices (0,1,2,3)
    W_232[0:4, 0:7] = W1
    
    # A2B2 part (cols 7-10)
    # A2 is 2x1 (indices 4,5 in A? No, A is 2x3. A1 is 2x2 (0,1,2,3), A2 is 2x1 (4,5))
    # B2 is 1x2 (indices 4,5 in B? No, B is 3x2. B1 is 2x2 (0,1,2,3), B2 is 1x2 (4,5))
    # C is 2x2.
    # Trivial 2x1x2:
    # A2: 0,1. B2: 0,1.
    # Products: A2[0]B2[0], A2[0]B2[1], A2[1]B2[0], A2[1]B2[1]
    # C[0,0] = A2[0]B2[0]
    # C[0,1] = A2[0]B2[1]
    # C[1,0] = A2[1]B2[0]
    # C[1,1] = A2[1]B2[1]
    
    U_232[4, 7] = 1
    V_232[4, 7] = 1
    W_232[0, 7] = 1
    
    U_232[4, 8] = 1
    V_232[5, 8] = 1
    W_232[1, 8] = 1
    
    U_232[5, 9] = 1
    V_232[4, 9] = 1
    W_232[2, 9] = 1
    
    U_232[5, 10] = 1
    V_232[5, 10] = 1
    W_232[3, 10] = 1
    
    # Build 2x3x3 (Rank 17)
    # A(2x3) = [A1(2x2) | A2(2x1)]
    # B(3x3) = [B1(2x3); B2(1x3)]
    # C = A1B1 + A2B2
    # A1B1: 2x2x3 (Rank 11)
    # A2B2: 2x1x3 (Rank 6)
    
    # Build 2x2x3 (Rank 11)
    # B1(2x3) = [B1a(2x2) | B1b(2x1)]
    # A1B1 = A1B1a + A1B1b
    # A1B1a: 2x2x2 (Strassen, Rank 7)
    # A1B1b: 2x2x1 (Trivial, Rank 4)
    
    U_223 = np.zeros((4, 11), dtype=int)
    V_223 = np.zeros((6, 11), dtype=int)
    W_223 = np.zeros((6, 11), dtype=int)
    
    # A1B1a (cols 0-6)
    U1, V1, W1 = strassen_222()
    U_223[0:4, 0:7] = U1
    V_223[0:4, 0:7] = V1
    W_223[0:4, 0:7] = W1 # C1a is 2x2, maps to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # A1B1b (cols 7-10)
    # A1 (2x2), B1b (2x1), C1b (2x1)
    # Trivial 2x2x1:
    # A[0]B[0], A[0]B[1], A[1]B[0], A[1]B[1]
    # C[0] = A[0]B[0] + A[0]B[1] ? No.
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # Rank 4.
    
    U_223[0, 7] = 1
    V_223[0, 7] = 1
    W_223[4, 7] = 1 # C[0,2]
    
    U_223[1, 8] = 1
    V_223[2, 8] = 1
    W_223[4, 8] = 1
    
    U_223[2, 9] = 1
    V_223[0, 9] = 1
    W_223[5, 9] = 1
    
    U_223[3, 10] = 1
    V_223[2, 10] = 1
    W_223[5, 10] = 1
    
    # Build 2x1x3 (Rank 6)
    # A2 (2x1), B2 (1x3), C2 (2x3)
    # Trivial.
    
    U_213 = np.zeros((2, 6), dtype=int)
    V_213 = np.zeros((3, 6), dtype=int)
    W_213 = np.zeros((6, 6), dtype=int)
    
    # A[0]B[0], A[0]B[1], A[0]B[2]
    # A[1]B[0], A[1]B[1], A[1]B[2]
    
    U_213[0, 0] = 1; V_213[0, 0] = 1; W_213[0, 0] = 1 # C[0,0]
    U_213[0, 1] = 1; V_213[1, 1] = 1; W_213[1, 1] = 1 # C[0,1]
    U_213[0, 2] = 1; V_213[2, 2] = 1; W_213[2, 2] = 1 # C[0,2]
    
    U_213[1, 3] = 1; V_213[0, 3] = 1; W_213[3, 3] = 1 # C[1,0]
    U_213[1, 4] = 1; V_213[1, 4] = 1; W_213[4, 4] = 1 # C[1,1]
    U_213[1, 5] = 1; V_213[2, 5] = 1; W_213[5, 5] = 1 # C[1,2]
    
    # Assemble 2x3x3
    U_233 = np.zeros((6, 17), dtype=int)
    V_233 = np.zeros((9, 17), dtype=int)
    W_233 = np.zeros((6, 17), dtype=int)
    
    # A1B1 part (cols 0-10)
    # A1 indices 0-3. B1 indices 0-5. C1 indices 0-5.
    U_233[0:4, 0:11] = U_223
    V_233[0:6, 0:11] = V_223
    W_233[0:6, 0:11] = W_223
    
    # A2B2 part (cols 11-16)
    # A2 indices 4,5. B2 indices 6,7,8. C2 indices 0-5.
    U_233[4:6, 11:17] = U_213
    V_233[6:9, 11:17] = V_213
    W_233[0:6, 11:17] = W_213
    
    # Assemble 2x3x5
    # B(3x5) = [B_left(3x2) | B_right(3x3)]
    # C = A B_left + A B_right
    
    U_full = np.zeros((6, 28), dtype=int)
    V_full = np.zeros((15, 28), dtype=int)
    W_full = np.zeros((10, 28), dtype=int)
    
    # Left part (cols 0-10)
    U_full[0:6, 0:11] = U_232
    V_full[0:6, 0:11] = V_232
    W_full[0:4, 0:11] = W_232 # C_left is 2x2, maps to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Right part (cols 11-27)
    U_full[0:6, 11:28] = U_233
    V_full[6:15, 11:28] = V_233
    W_full[4:10, 11:28] = W_233 # C_right is 2x3, maps to C[0,2], C[0,3], C[0,4], C[1,2], C[1,3], C[1,4]
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
