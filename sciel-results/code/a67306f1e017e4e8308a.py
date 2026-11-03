import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Construct M_{1,4,4} (Schoolbook, Rank 16)
    # n=1, m=4, p=4
    U_144, V_144, W_144 = _schoolbook(1, 4, 4)
    
    # 2. Construct M_{2,2,4} (Strassen-based, Rank 14)
    # This is M_{2,2,2} + M_{2,2,2}
    U_224, V_224, W_224 = _strassen_224()
    
    # 3. Construct M_{2,4,4} (Two M_{2,2,4})
    # n=2, m=4, p=4
    # Split m=4 into two m=2 blocks.
    # A is 2x4. B is 4x4.
    # C = A_left B_top + A_right B_bot
    # This corresponds to two M_{2,2,4} computations.
    U_244, V_244, W_244 = _combine_m_224()
    
    # 4. Combine M_{2,4,4} and M_{1,4,4} for M_{3,4,4}
    # n=3, m=4, p=4
    # A is 3x4. B is 4x4.
    # C_top (2x4) = A_top (2x4) B
    # C_bot (1x4) = A_bot (1x4) B
    # Total rank = Rank(M_{2,4,4}) + Rank(M_{1,4,4}) = 28 + 16 = 44.
    
    rank_total = U_244.shape[1] + U_144.shape[1]
    U = np.zeros((3 * 4, rank_total))
    V = np.zeros((4 * 4, rank_total))
    W = np.zeros((4 * 3, rank_total))
    
    # Fill M_{2,4,4} part (rows 0-1 of A, all rows of B, cols 0-1 of C)
    # U_244 has shape (8, 28). Indices i in 0..1, j in 0..3.
    # U_144 has shape (4, 16). Indices i=0, j in 0..3.
    
    # M_{2,4,4} part
    U[:8, :28] = U_244
    V[:16, :28] = V_244
    W[:8, :28] = W_244 # W is p*n = 4*2 = 8 for this part? 
    # Wait, W for M_{2,4,4} has shape (p*n, R) = (4*2, 28) = (8, 28).
    # Indices k in 0..3, i in 0..1. Flattened k*2+i.
    
    # M_{1,4,4} part
    # This corresponds to i=2 (the 3rd row of A).
    # U_144 has shape (4, 16). Indices j in 0..3. (i is fixed 0 in local, 2 in global)
    # Global U index: i*m+j = 2*4+j = 8+j.
    U[8:12, 28:44] = U_144
    
    # V_144 has shape (16, 16). Indices j in 0..3, k in 0..3.
    # Global V index: j*p+k = j*4+k. Same as local.
    V[:, 28:44] = V_144
    
    # W_144 has shape (4, 16). Indices k in 0..3, i=0 local.
    # Global W index: k*n+i = k*3+2.
    # W_144 rows are k.
    W[2:12:3, 28:44] = W_144 # k=0->2, k=1->5, k=2->8, k=3->11
    
    # Verify
    if not _check_exact(U, V, W, 3, 4, 4):
        # Fallback to schoolbook if something went wrong
        U, V, W = _schoolbook(3, 4, 4)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    # Returns U, V, W for M_{2,2,2} with rank 7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _strassen_224():
    # M_{2,2,4} using two M_{2,2,2}
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    
    # First block (k=0,1)
    U[:, 0:7] = U1
    V[0:4, 0:7] = V1 # j*2+k, k in 0,1 -> indices 0,1,2,3
    W[0:4, 0:7] = W1 # k*2+i, k in 0,1 -> indices 0,1,2,3
    
    # Second block (k=2,3)
    U[:, 7:14] = U2
    V[4:8, 7:14] = V2 # j*2+k, k in 2,3 -> indices 4,5,6,7
    W[4:8, 7:14] = W2 # k*2+i, k in 2,3 -> indices 4,5,6,7
    
    return U, V, W

def _combine_m_224():
    # M_{2,4,4} using two M_{2,2,4}
    U1, V1, W1 = _strassen_224()
    U2, V2, W2 = _strassen_224()
    
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((8, 28))
    
    # Block 1: j in 0,1 (A_left), B_top (k in 0,1? No, B is 4x4)
    # A is 2x4. Split A into A_left (2x2) and A_right (2x2).
    # B is 4x4. Split B into B_top (2x4) and B_bot (2x4).
    # C = A_left B_top + A_right B_bot.
    
    # Term 1: A_left (j=0,1) * B_top (j=0,1, k=0..3)
    # This is M_{2,2,4}.
    # U1 corresponds to A_left.
    U[0:4, 0:14] = U1
    # V1 corresponds to B_top. B_top rows are j=0,1.
    # V index is j*p+k = j*4+k.
    # For j=0: k=0..3 -> 0..3
    # For j=1: k=0..3 -> 4..7
    # V1 has shape (8, 14). Rows 0..7.
    # V1 rows 0..3 correspond to j=0? No, V1 was for M_{2,2,4} where p=4.
    # In M_{2,2,4}, V index is j*4+k. j in 0,1.
    # So V1 rows 0..7 cover j=0,1 and k=0..3.
    V[0:8, 0:14] = V1
    # W1 corresponds to C_top.
    # W index is k*n+i = k*2+i.
    # W1 has shape (8, 14).
    W[0:8, 0:14] = W1
    
    # Term 2: A_right (j=2,3) * B_bot (j=2,3, k=0..3)
    # This is M_{2,2,4}.
    U[4:8, 14:28] = U2
    # V2 corresponds to B_bot.
    # B_bot rows are j=2,3.
    # V index is j*4+k.
    # For j=2: k=0..3 -> 8..11
    # For j=3: k=0..3 -> 12..15
    # V2 has shape (8, 14). Rows 0..7 correspond to local j=0,1.
    # We need to map local j=0 -> global j=2, local j=1 -> global j=3.
    # So V2 rows 0..3 (local j=0) -> V rows 8..11.
    # V2 rows 4..7 (local j=1) -> V rows 12..15.
    V[8:12, 14:28] = V2[0:4, :]
    V[12:16, 14:28] = V2[4:8, :]
    
    # W2 corresponds to C_top.
    # W index is k*2+i.
    # W2 has shape (8, 14).
    W[0:8, 14:28] = W2
    
    return U, V, W

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
