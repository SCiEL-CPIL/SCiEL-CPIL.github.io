"""
Fresh design: Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7)
to get rank 189 for 6x6x6 matrix multiplication.

Key insight: If T_A is a decomposition of A (size a×a) and T_B is a decomposition of B (size b×b),
then kron(T_A, T_B) is a decomposition of the (a*b)×(a*b) matrix multiplication.

Specifically:
- 3×3 schoolbook: rank 27
- 2×2 Strassen: rank 7
- 6×6 via kron: rank 27 * 7 = 189

This is better than the naive 3×3 block approach with 2×2 Strassen (which also gives 189
but through a different construction). The Kronecker approach is cleaner and well-known.

The construction:
- Partition 6×6 matrices into 3×3 blocks of 2×2
- The 3×3 block multiplication uses schoolbook (27 terms)
- Each 2×2 block product uses Strassen (7 terms)
- Total: 27 * 7 = 189 rank-1 terms

Implementation via Kronecker product of the base decompositions with appropriate row/column permutations.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7).
    """
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Define 2x2 Strassen decomposition
    # U22: shape (4, 7) where row = i*2+j for A[i,j], col = rank term
    # V22: shape (4, 7) where row = j*2+k for B[j,k], col = rank term  
    # W22: shape (4, 7) where row = k*2+i for C[i,k], col = rank term

    # Strassen terms for 2x2:
    # P1 = (x11+x22)(y11+y22) -> c11+=1, c22+=1
    # P2 = (x21+x22)(y11)     -> c21+=1, c22+=-1
    # P3 = (x11)(y12-y22)     -> c12+=1, c22+=1
    # P4 = (x22)(y21-y11)     -> c11+=1, c21+=1
    # P5 = (x11+x12)(y22)     -> c11+=-1, c12+=1
    # P6 = (x21-x11)(y11+y12) -> c22+=1
    # P7 = (x12-x22)(y21+y22) -> c11+=1

    U22 = np.zeros((4, 7), dtype=float)
    V22 = np.zeros((4, 7), dtype=float)
    W22 = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    # x11=A[0,0]=row0, x22=A[1,1]=row3
    # y11=B[0,0]=row0, y22=B[1,1]=row3
    # c11=C[0,0]: W index = k*n+i = 0*2+0=0, c22=C[1,1]: W index = 1*2+1=3
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    # x21=A[1,0]=row2, x22=A[1,1]=row3
    # y11=B[0,0]=row0
    # c21=C[1,0]: W index = 0*2+1=1, c22=C[1,1]: W index = 1*2+1=3
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[1, 1] = 1; W22[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    # x11=A[0,0]=row0
    # y12=B[0,1]=row1, y22=B[1,1]=row3
    # c12=C[0,1]: W index = 1*2+0=2, c22=C[1,1]: W index = 1*2+1=3
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[2, 2] = 1; W22[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    # x22=A[1,1]=row3
    # y21=B[1,0]=row2, y11=B[0,0]=row0
    # c11=C[0,0]: W index = 0, c21=C[1,0]: W index = 1
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    # x11=A[0,0]=row0, x12=A[0,1]=row1
    # y22=B[1,1]=row3
    # c11=C[0,0]: W index = 0, c12=C[0,1]: W index = 2
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1; W22[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    # x21=A[1,0]=row2, x11=A[0,0]=row0
    # y11=B[0,0]=row0, y12=B[0,1]=row1
    # c22=C[1,1]: W index = 3
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    # x12=A[0,1]=row1, x22=A[1,1]=row3
    # y21=B[1,0]=row2, y22=B[1,1]=row3
    # c11=C[0,0]: W index = 0
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1

    # Define 3x3 schoolbook decomposition
    # U33: shape (9, 27) where row = i*3+j for A[i,j], col = rank term
    # V33: shape (9, 27) where row = j*3+k for B[j,k], col = rank term
    # W33: shape (9, 27) where row = k*3+i for C[i,k], col = rank term

    U33 = np.zeros((9, 27), dtype=float)
    V33 = np.zeros((9, 27), dtype=float)
    W33 = np.zeros((9, 27), dtype=float)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i * 3 + j, col] = 1
                V33[j * 3 + k, col] = 1
                W33[k * 3 + i, col] = 1
                col += 1

    # Now form the Kronecker product decomposition for 6x6
    # The Kronecker product of decompositions:
    # If (U1, V1, W1) decomposes size-a matmul and (U2, V2, W2) decomposes size-b matmul,
    # then (kron(U1,U2), kron(V1,V2), kron(W1,W2)) decomposes size-(a*b) matmul,
    # with appropriate row permutations.
    
    # For U: row index in kron = (i1*a1 + j1) * (a2*b2) + (i2*a2 + j2)
    #        where a1=3, b1=3, a2=2, b2=2
    #        Target row index for 6x6: (i1*2 + i2) * 6 + (j1*2 + j2)
    
    # Let's build the final matrices directly
    rank = 27 * 7  # 189
    
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)
    
    # Column in final: c = c33 * 7 + c22
    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22
            
            # U33[i3*3 + j3, c33] * U22[i2*2 + j2, c22] -> U66 row (i3*2+i2)*6 + (j3*2+j2)
            for i3 in range(3):
                for j3 in range(3):
                    u33_val = U33[i3 * 3 + j3, c33]
                    if u33_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, c22]
                            if u22_val == 0:
                                continue
                            target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U66[target_row, c_final] = u33_val * u22_val
            
            # V33[j3*3 + k3, c33] * V22[j2*2 + k2, c22] -> V66 row (j3*2+j2)*6 + (k3*2+k2)
            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, c33]
                    if v33_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0:
                                continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V66[target_row, c_final] = v33_val * v22_val
            
            # W33[k3*3 + i3, c33] * W22[k2*2 + i2, c22] -> W66 row (k3*2+k2)*6 + (i3*2+i2)
            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, c33]
                    if w33_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0:
                                continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W66[target_row, c_final] = w33_val * w22_val
    
    # Verify correctness
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m))
        B_test = np.random.randint(-5, 6, size=(m, p))
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A_test[i, j] * B_test[j, k] * U66[i*m + j, r] * V66[j*p + k, r] * W66[k*n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_fb = np.zeros((n * m, rank_school), dtype=float)
            V_fb = np.zeros((m * p, rank_school), dtype=float)
            W_fb = np.zeros((p * n, rank_school), dtype=float)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, column] = 1
                        V_fb[j * p + k, column] = 1
                        W_fb[k * n + i, column] = 1
                        column += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}
# EVOLVE-BLOCK-END
