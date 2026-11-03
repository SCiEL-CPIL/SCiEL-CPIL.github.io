import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all (i,j,k)."""
    rank = len(U[0])
    # For each c, the contribution is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Wait, the tensor decomposition is T = sum U_c \otimes V_c \otimes W_c.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor T[i,j,k] is 1 if i,j,k are valid indices for the product, else 0?
    # No, the tensor T is defined such that T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'.
    # In the flattened form:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The condition for the term to be non-zero in the sum is that the indices match.
    # Specifically, the "tensor" we are decomposing is the identity mapping of the multiplication.
    # T[i, j, k] = 1 for all i,j,k.
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check if the product matches the tensor value (which is 1)
                    # Wait, the sum over c must equal 1.
                    pass
    
    # Correct verification:
    # Compute the full tensor from U, V, W and compare to the target tensor.
    # Target tensor T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # To be efficient, we can just check a few random points or all points if rank is small.
    # Given the small dimensions (4,4,7), we can check all points.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U: (i,j) -> i*2+j
    V: (j,k) -> j*2+k
    W: (k,i) -> k*2+i
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen's 7 multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U vectors (for A):
    # M1: A00 + A11 -> U[0][0]=1, U[3][0]=1
    # M2: A10 + A11 -> U[2][1]=1, U[3][1]=1
    # M3: A00       -> U[0][2]=1
    # M4: A11       -> U[3][3]=1
    # M5: A00 + A01 -> U[0][4]=1, U[1][4]=1
    # M6: A10 - A00 -> U[2][5]=1, U[0][5]=-1
    # M7: A01 - A11 -> U[1][6]=1, U[3][6]=-1
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # V vectors (for B):
    # M1: B00 + B11 -> V[0][0]=1, V[3][0]=1
    # M2: B00       -> V[0][1]=1
    # M3: B01 - B11 -> V[1][2]=1, V[3][2]=-1
    # M4: B10 - B00 -> V[2][3]=1, V[0][3]=-1
    # M5: B11       -> V[3][4]=1
    # M6: B00 + B01 -> V[0][5]=1, V[1][5]=1
    # M7: B10 + B11 -> V[2][6]=1, V[3][6]=1
    
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    # W vectors (for C):
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # W index: k*2+i
    # W[0] is C00 (k=0, i=0)
    # W[1] is C01 (k=0, i=1)
    # W[2] is C10 (k=1, i=0)
    # W[3] is C11 (k=1, i=1)
    
    # M1: +1 to C00, +1 to C11
    # M2: +1 to C10
    # M3: +1 to C01, -1 to C11
    # M4: -1 to C00, +1 to C10
    # M5: +1 to C01, +1 to C11
    # M6: +1 to C00, -1 to C11
    # M7: +1 to C00
    
    W[0][0] = 1; W[3][0] = 1
    W[2][1] = 1
    W[1][2] = 1; W[3][2] = -1
    W[0][3] = -1; W[2][3] = 1
    W[1][4] = 1; W[3][4] = 1
    W[0][5] = 1; W[3][5] = -1
    W[0][6] = 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    Build 4x4x4 decomposition using Kronecker product of Strassen.
    Rank 49.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Kronecker product
    # U4 size 16x49
    # V4 size 16x49
    # W4 size 16x49
    
    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    # Index in 4x4
                    I = i * 2 + k
                    J = j * 2 + l
                    K = k * 2 + i  # Wait, W index is k*n+i. Here n=4.
                    # In 2x2, W index is k*2+i.
                    # In 4x4, W index is k*4+i.
                    
                    # The Kronecker product structure:
                    # U4[I, J, c1, c2] = U2[i, j, c1] * U2[k, l, c2]
                    # This is not quite right. The indices for U are (i,j).
                    # In 4x4, i = i1*2 + i2, j = j1*2 + j2.
                    # U4[i1*2+i2, j1*2+j2, c1*7+c2] = U2[i1, j1, c1] * U2[i2, j2, c2]
                    
                    pass
    
    # Let's do it properly with loops
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    I = i1 * 2 + i2
                    J = j1 * 2 + j2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            U4[I][J] # No, U4 is 2D: index (I, J) is wrong. U4 is (n*m, rank).
                            # U4 index is (I*4+J)? No, U is (n*m, rank).
                            # n=4, m=4. Size 16.
                            # Index is i*4+j.
                            # U4[i*4+j, c] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                            
                            val = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                            if val != 0:
                                U4[i * 4 + j][c] = val # Typo: I, J
                                U4[I * 4 + J][c] = val
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    J = j1 * 2 + j2
                    K = k1 * 2 + k2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            val = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                            if val != 0:
                                V4[J * 4 + K][c] = val
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    K = k1 * 2 + k2
                    I = i1 * 2 + i2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            val = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                            if val != 0:
                                W4[K * 4 + I][c] = val
    
    return U4, V4, W4


def _schoolbook_4x4x3():
    """
    Build 4x4x3 schoolbook decomposition.
    Rank 48.
    """
    n, m, p = 4, 4, 3
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
                
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Build 4x4x4 part (Rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Build 4x4x3 part (Rank 48)
    U3, V3, W3 = _schoolbook_4x4x3()
    
    # Combine them
    # Total Rank = 49 + 48 = 97
    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part (k=0..3)
    for i in range(n * m):
        for c in range(49):
            U[i][c] = U4[i][c]
            
    for j in range(m):
        for k in range(4): # p=4 for the first part
            for c in range(49):
                V[j * p + k][c] = V4[j * 4 + k][c]
                
    for k in range(4):
        for i in range(n):
            for c in range(49):
                W[k * n + i][c] = W4[k * 4 + i][c]
                
    # Copy 4x4x3 part (k=4..6)
    # The V and W indices for the second part are offset.
    # V index for second part: j*p + k where k in 4..6.
    # In U3, V3, W3, k is 0..2.
    # So V3[j*3+k] corresponds to V[j*p + (k+4)]
    # W3[k*4+i] corresponds to W[(k+4)*n + i]
    
    offset = 49
    for i in range(n * m):
        for c in range(48):
            U[i][offset + c] = U3[i][c]
            
    for j in range(m):
        for k in range(3):
            for c in range(48):
                V[j * p + (k + 4)][offset + c] = V3[j * 3 + k][c]
                
    for k in range(3):
        for i in range(n):
            for c in range(48):
                W[(k + 4) * n + i][offset + c] = W3[k * 4 + i][c]
                
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        result = {"U": U_f, "V": V_f, "W": W_f}
        
    return result

# EVOLVE-BLOCK-END
