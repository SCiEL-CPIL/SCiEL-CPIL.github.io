import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses recursive block decomposition:
    1. Split A (2x4) into two 2x2 blocks.
    2. Split B (4x5) into two 2x5 blocks.
    3. Each 2x2x5 block is split into 2x2x2 (Strassen, rank 7) and 2x2x3 (Rank 11).
    4. 2x2x3 is split into 2x2x2 (Strassen, rank 7) and 2x2x1 (Schoolbook, rank 4).
    Total rank = 2 * (7 + 7 + 4) = 36.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Helper to create Strassen 2x2x2 decomposition
    # Returns U (4x7), V (4x7), W (4x7)
    # Note: W index is k*n+i. n=2. So W is 4x7.
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # A indices: 00, 11. B indices: 00, 11.
        # U[0,0]=1, U[3,0]=1 (a, d)
        # V[0,0]=1, V[3,0]=1 (e, h) -- Wait, B is 2x2. B[0,0]=e, B[1,1]=h?
        # B indices: j*p+k. p=2.
        # B[0,0] is index 0. B[1,1] is index 3.
        # W indices: k*n+i.
        # M1 contributes to C00, C01, C10, C11?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        
        # We need to map the 7 products to the tensor.
        # Product r is $M_r = \sum U_{ij,r} A_{ij} \sum V_{jk,r} B_{jk}$.
        # Then $C_{ik} = \sum W_{ki,r} M_r$.
        
        # M1 = (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        # M1 adds to C00 (W[0,0]) and C11 (W[3,0])
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (c+d)(e)
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        # M2 adds to C10 (W[2,0]? No, C10 is index 2 in W? W[k*n+i]. C10 -> k=1, i=0 -> idx 2. C11 -> idx 3.)
        # C10 = M2 + M4. So W[2,1] = 1.
        # C11 = M1 - M2 + ... So W[3,1] = -1.
        W[2,1] = 1; W[3,1] = -1
        
        # M3 = (a-b)(f+h)
        U[0,2] = 1; U[1,2] = -1
        V[1,2] = 1; V[3,2] = 1
        # C01 = M3 + M5. C01 is k=1, i=0 -> idx 2? No.
        # W index k*n+i.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        # So C01 is W[2, :].
        W[2,2] = 1
        
        # M4 = (a+d)(i-g)
        U[0,3] = 1; U[3,3] = 1
        V[2,3] = 1; V[3,3] = -1 # i is B[1,0] (idx 2), g is B[0,1] (idx 1)?
        # B indices: 00(0), 01(1), 10(2), 11(3).
        # i is B[1,0] -> idx 2. g is B[0,1] -> idx 1.
        # So V[2,3]=1, V[1,3]=-1.
        V[1,3] = -1
        # C00 = ... + M4. C10 = M2 + M4.
        W[0,3] = 1; W[1,3] = 1
        
        # M5 = (a+c)(e-h)
        U[0,4] = 1; U[2,4] = 1
        V[0,4] = 1; V[3,4] = -1
        # C00 = ... - M5. C01 = M3 + M5.
        W[0,4] = -1; W[2,4] = 1
        
        # M6 = (b+d)(g-h)
        U[1,5] = 1; U[3,5] = 1
        V[1,5] = 1; V[3,5] = -1
        # C11 = ... + M6.
        W[3,5] = 1
        
        # M7 = (a-c)(f+i)
        U[0,6] = 1; U[2,6] = -1
        V[1,6] = 1; V[2,6] = 1
        # C00 = ... + M7. C11 = ... + M7.
        W[0,6] = 1; W[3,6] = 1
        
        return U, V, W

    # Helper to create Schoolbook 2x2x1 decomposition (Rank 4)
    # A (2x2), B (2x1). C (2x1).
    # U (4x4), V (2x4), W (2x4)
    def schoolbook_221():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # r=0: A00 B0 -> C0
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # r=1: A01 B1 -> C0
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        # r=2: A10 B0 -> C1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        # r=3: A11 B1 -> C1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    # Assemble 2x2x3 (Rank 11)
    # U (4x11), V (6x11), W (6x11)
    def block_223():
        U1, V1, W1 = strassen_222() # 2x2x2 part (cols 0,1 of B)
        U2, V2, W2 = schoolbook_221() # 2x2x1 part (col 2 of B)
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Part 1: 2x2x2 (B cols 0,1)
        U[:, 0:7] = U1
        V[0:4, 0:7] = V1 # B indices 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
        # Wait, V for 2x2x2 has shape 4x7. Indices 0..3.
        # In 2x2x3, B has 6 indices: 0..5.
        # B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3.
        # So V1 maps to V[0:4, :].
        # W for 2x2x2 has shape 4x7. Indices 0..3.
        # In 2x2x3, W has 6 indices: 0..5.
        # W indices: k*n+i.
        # k=0,1. i=0,1.
        # C00->0, C01->2, C10->1, C11->3.
        # So W1 maps to W[0:4, :].
        W[0:4, 0:7] = W1
        
        # Part 2: 2x2x1 (B col 2)
        # B index for col 2: B[0,2]->4, B[1,2]->5.
        U[:, 7:11] = U2
        V[4:6, 7:11] = V2 # V2 is 2x4. Maps to V[4:6, :].
        # W for 2x2x1: C0->0, C1->1.
        # In 2x2x3, C0 is idx 0, C1 is idx 1.
        # So W2 maps to W[0:2, :].
        W[0:2, 7:11] = W2
        
        return U, V, W

    # Assemble 2x2x5 (Rank 18)
    # U (4x18), V (10x18), W (10x18)
    def block_225():
        U1, V1, W1 = strassen_222() # 2x2x2 (B cols 0,1)
        U2, V2, W2 = block_223() # 2x2x3 (B cols 2,3,4)
        
        U = np.zeros((4, 18))
        V = np.zeros((10, 18))
        W = np.zeros((10, 18))
        
        # Part 1: 2x2x2
        U[:, 0:7] = U1
        V[0:4, 0:7] = V1
        W[0:4, 0:7] = W1
        
        # Part 2: 2x2x3
        U[:, 7:18] = U2
        # V2 is 6x11. Maps to V[4:10, :].
        V[4:10, 7:18] = V2
        # W2 is 6x11. Maps to W[4:10, :].
        W[4:10, 7:18] = W2
        
        return U, V, W

    # Assemble 2x4x5 (Rank 36)
    # U (8x36), V (20x36), W (10x36)
    def construct_final():
        U1, V1, W1 = block_225() # A[0:2, :], B[0:2, :]
        U2, V2, W2 = block_225() # A[2:4, :], B[2:4, :]
        
        U = np.zeros((8, 36))
        V = np.zeros((20, 36))
        W = np.zeros((10, 36))
        
        # Part 1: A[0:2, :] (U indices 0..3), B[0:2, :] (V indices 0..9)
        U[0:4, 0:18] = U1
        V[0:10, 0:18] = V1
        W[:, 0:18] = W1
        
        # Part 2: A[2:4, :] (U indices 4..7), B[2:4, :] (V indices 10..19)
        U[4:8, 18:36] = U2
        V[10:20, 18:36] = V2
        W[:, 18:36] = W2
        
        return U, V, W

    U, V, W = construct_final()
    
    # Verify
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_s = np.zeros((n*m, rank_schoolbook))
        V_s = np.zeros((m*p, rank_schoolbook))
        W_s = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
