"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Schoolbook decomposition: rank = n*m*p = 40
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    
    # Strassen-like decomposition: split inner dim 4 into 2+2, output dim 5 into 2+3
    # Block 1: 2x2x2 (Strassen rank 7)
    # Block 2: 2x2x3 (rank 9)
    # Total rank: 16
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For multiplying (2x2) * (2x2) -> (2x2)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # W index: k*n + i where n=2 for this block
    
    # Standard Strassen factors for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U factors (input A, shape 2x2): rows indexed by (i,j) where i=row of A, j=col of A
    # V factors (input B, shape 2x2): rows indexed by (j,k) where j=row of B, k=col of B
    # W factors (output C, shape 2x2): rows indexed by (k,i) where k=col of C, i=row of C
    # W[k*n+i] with n=2
    
    strassen_U = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0) = A11
        [0, 0, 1, 0, 1, -1, 0], # (0,1) = A12
        [0, 1, 0, 0, 0, 1, 0],  # (1,0) = A21
        [1, 1, 0, 1, 0, 0, -1], # (1,1) = A22
    ]
    
    strassen_V = [
        [1, 1, 0, 1, 0, 1, 0],  # (0,0) = B11
        [0, 0, 1, 0, 1, 0, 1],  # (0,1) = B12
        [1, 0, 0, -1, 0, 0, 1], # (1,0) = B21
        [1, 0, -1, 0, 1, 0, 0], # (1,1) = B22
    ]
    
    strassen_W = [
        [0, 0, 1, 0, 1, 0, 0],  # (0,0) = C11: k=0,i=0
        [1, 1, 0, 0, 0, 1, 0],  # (0,1) = C12: k=1,i=0
        [0, 0, 0, 1, 0, 0, 1],  # (1,0) = C21: k=0,i=1
        [0, 1, 0, 0, 1, 0, 0],  # (1,1) = C22: k=1,i=1
    ]
    
    # Verify Strassen block
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # W[k*2+i] should equal C[i][k]
    
    def verify_strassen_block():
        for i in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s = 0
                    for j in range(2):
                        s += strassen_U[i*2+j][r] * strassen_V[j*2+k][r] * strassen_W[k*2+i][r]
                if s != 1:
                    return False
        return True
    
    # For 2x2x3 block (rank 9): multiply (2x2) * (2x3) -> (2x3)
    # This is like doing two 2x2x2 multiplications but sharing some factors
    # Actually, let's use a different approach: treat 2x2x3 as rank 6 (schoolbook) 
    # or find a better decomposition
    
    # For 2x2x3: input A is 2x2, input B is 2x3, output C is 2x3
    # Schoolbook rank = 2*2*3 = 6
    # Let's check if we can do better with rank 6
    
    # Actually, the tensor rank of 2x2x3 is 6 (it's a full rank tensor)
    # So we use rank 6 for this block
    
    # Total rank: 7 + 6 = 13
    
    # Let me reconsider the decomposition
    # Inner dimension 4 split into 2+2
    # Output dimension 5 split into 2+3
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, :]  (2x2 times 2x2 gives 2x2)
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[0:2, :]  (2x2 times 2x2 gives 2x2)  
    # Wait, this doesn't work cleanly
    
    # Let me think more carefully
    # A is 2x4, B is 4x5, C is 2x5
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split j into 0,1 and 2,3
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # For k=0,1: first term is (2x2)*(2x2)->(2x2), second term is (2x2)*(2x2)->(2x2)
    # For k=2,3,4: first term is (2x2)*(2x3)->(2x3), second term is (2x2)*(2x3)->(2x3)
    
    # This is getting complex. Let me just use the direct sum approach:
    # Total rank = 7 + 7 + 6 + 6 = 26? No, that's not right either.
    
    # Let me just try rank 16 with a systematic approach
    # Actually, let me try to construct a rank-16 solution directly
    
    # Alternative: use the fact that 2x4x5 can be decomposed as:
    # (2x2x2) + (2x2x2) + (2x2x3) + (2x2x3)
    # But we need to share factors where possible
    
    # Let me just return the schoolbook for now and try random search for improvement
    for rank in range(rank_school - 1, 12, -1):
        if time.time() > deadline - 5:
            break
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
            if verify(U, V, W, rank):
                best = {"U": U, "V": V, "W": W}
                break
    
    return best
# EVOLVE-BLOCK-END
