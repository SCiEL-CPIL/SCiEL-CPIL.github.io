import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n=2, m=3, p=5):
    """Verify the tensor decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T from the decomposition
    # T[i*m+j, j*p+k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually, the tensor for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    # Rank-24 decomposition:
    # j=2: A[:,2]*B[2,:] -> rank 10
    # j=0,1, k=0,1: Strassen 2x2x2x2 -> rank 7
    # j=0,1, k=2,3: Strassen 2x2x2x2 -> rank 7
    # j=0,1, k=4: direct 2x2x2x1 -> rank 4
    # Total: 10+7+7+4 = 28... let me try harder for 24.
    
    # Alternative: use a unified Strassen-like for 2x2x2x5
    # Split k into {0,1,2,3,4}
    # Use 7 terms covering k=0,1 and 5 more for k=2,3,4 with sharing
    
    # Let me try rank 24:
    # j=2: 10 terms
    # j=0,1: 14 terms using a clever 2x2x2x5 in rank 14
    
    # For 2x2x2x5 in rank 14:
    # Use Strassen for k=0,1 (7 terms)
    # Use 7 more terms for k=2,3,4 by reusing A-combinations
    
    # Actually, let me just do 10+7+7 = 24 by handling k=4 within the Strassen blocks
    # No, that doesn't work.
    
    # Let me carefully construct rank 24:
    # 10 terms for j=2
    # 7 terms for j=0,1 k=0,1 (Strassen)
    # 7 terms for j=0,1 k=2,3 (Strassen)
    # But that's 24 and we still need k=4!
    
    # So rank 24 = 10 + 7 + 7 only works if k=4 is handled within those.
    # It's not, so we need 28.
    
    # Let me try a different approach: merge k=4 into the second Strassen block.
    # Instead of doing k=2,3 as 2x2x2x2, do k=2,3,4 as 2x2x2x3.
    # Rank of 2x2x2x3: can we do it in 7? No, minimum is at least 6.
    # Actually 2x2x2x3 has tensor rank at most 2*2*3=12, and at least... 
    # I think it's 8 or 9.
    
    # Let me just go with rank 24 by a different split:
    # Split j into {0} and {1,2}:
    # j=0: A[:,0]*B[0,:] -> rank 2*5=10
    # j=1,2: A[:,1:3]*B[1:3,:] -> 2x2x2x5
    
    # Same problem.
    
    # I'll construct the rank-28 solution correctly and verify it.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    # Part 1: j=2, all k (rank 10)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 10
    
    # Part 2: j=0,1, k=4 (rank 4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # r = 14
    
    # Part 3: j=0,1, k=0,1 Strassen (rank 7)
    # Using standard Strassen for 2x2x2:
    # M1 = (a00+a10)(b00+b10)
    U[0*m+0][r] = 1; U[1*m+0][r] = 1
    V[0*p+0][r] = 1; V[1*p+0][r] = 1
    W[0*n+0][r] = 1; W[1*n+1][r] = 1
    r += 1
    # M2 = (a10+a11)(b00+b01)
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+0][r] = 1; V[0*p+1][r] = 1
    W[0*n+1][r] = 1; W[1*n+0][r] = 1
    r += 1
    # M3 = (a00-a10)(b11-b01)
    U[0*m+0][r] = 1; U[1*m+0][r] = -1
    V[1*p+1][r] = 1; V[0*p+1][r] = -1
    W[1*n+0][r] = 1; W[1*n+1][r] = 1
    r += 1
    # M4 = (a01-a11)(b00+b10)
    U[0*m+1][r] = 1; U[1*m+1][r] = -1
    V[0*p+0][r] = 1; V[1*p+0][r] = 1
    W[0*n+0][r] = 1; W[1*n+1][r] = 1
    r += 1
    # M5 = (a00+a01)(b11-b01)
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[1*p+1][r] = 1; V[0*p+1][r] = -1
    W[0*n+1][r] = 1; W[1*n+0][r] = 1
    r += 1
    # M6 = (a10+a11)(b01-b11)
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+1][r] = 1; V[1*p+1][r] = -1
    W[1*n+0][r] = 1; W[1*n+1][r] = 1
    r += 1
    # M7 = (a00-a10)(b00+b10)
    U[0*m+0][r] = 1; U[1*m+0][r] = -1
    V[0*p+0][r] = 1; V[1*p+0][r] = 1
    W[0*n+0][r] = 1; W[1*n+1][r] = 1
    r += 1
    # r = 21
    
    # Part 4: j=0,1, k=2,3 Strassen (rank 7)
    # M8 = (a00+a10)(b02+b12)
    U[0*m+0][r] = 1; U[1*m+0][r] = 1
    V[0*p+2][r] = 1; V[1*p+2][r] = 1
    W[2*n+0][r] = 1; W[3*n+1][r] = 1
    r += 1
    # M9 = (a10+a11)(b02+b03)
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+2][r] = 1; V[0*p+3][r] = 1
    W[2*n+1][r] = 1; W[3*n+0][r] = 1
    r += 1
    # M10 = (a00-a10)(b13-b03)
    U[0*m+0][r] = 1; U[1*m+0][r] = -1
    V[1*p+3][r] = 1; V[0*p+3][r] = -1
    W[3*n+0][r] = 1; W[3*n+1][r] = 1
    r += 1
    # M11 = (a01-a11)(b02+b12)
    U[0*m+1][r] = 1; U[1*m+1][r] = -1
    V[0*p+2][r] = 1; V[1*p+2][r] = 1
    W[2*n+0][r] = 1; W[3*n+1][r] = 1
    r += 1
    # M12 = (a00+a01)(b13-b03)
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[1*p+3][r] = 1; V[0*p+3][r] = -1
    W[2*n+1][r] = 1; W[3*n+0][r] = 1
    r += 1
    # M13 = (a10+a11)(b03-b13)
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+3][r] = 1; V[1*p+3][r] = -1
    W[3*n+0][r] = 1; W[3*n+1][r] = 1
    r += 1
    # M14 = (a00-a10)(b02+b12)
    U[0*m+0][r] = 1; U[1*m+0][r] = -1
    V[0*p+2][r] = 1; V[1*p+2][r] = 1
    W[2*n+0][r] = 1; W[3*n+1][r] = 1
    r += 1
    # r = 28
    
    result = {"U": U, "V": V, "W": W}
    # Verify correctness
    if not _verify(U, V, W):
        # Fallback: return naive rank-30 solution
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END
