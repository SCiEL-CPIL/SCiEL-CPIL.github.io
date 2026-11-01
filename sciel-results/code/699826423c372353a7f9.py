"""Matrix multiplication 2,2,13 - rank-4 decomposition via row/column separation."""
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 13
    
    # The matrix multiplication tensor for (2x2) * (2x13) -> (2x13) has rank 4.
    # 
    # Decomposition:
    # out[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We can write this as a sum of 4 rank-1 terms:
    # Term r: U_r[i,j] * V_r[j,k] * W_r[k,i]
    #
    # Term 0: U[0,0]=1, V[0,k]=1 for all k, W[k,0]=1 for all k
    #   -> contributes A[0,0] * B[0,k] to out[0,k]
    #
    # Term 1: U[1,1]=1, V[1,k]=1 for all k, W[k,0]=1 for all k  
    #   -> contributes A[1,1] * B[1,k] to out[1,k]
    #
    # Wait, that's not right. Let me redo.
    #
    # out[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # out[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # 4 rank-1 terms:
    # Term 0: U[0,0]=1, V[0,k]=1 ∀k, W[k,0]=1 ∀k  -> A[0,0]*B[0,k] added to out[0,k]
    # Term 1: U[0,1]=1, V[1,k]=1 ∀k, W[k,0]=1 ∀k  -> A[0,1]*B[1,k] added to out[0,k]
    # Term 2: U[1,0]=1, V[0,k]=1 ∀k, W[k,1]=1 ∀k  -> A[1,0]*B[0,k] added to out[1,k]
    # Term 3: U[1,1]=1, V[1,k]=1 ∀k, W[k,1]=1 ∀k  -> A[1,1]*B[1,k] added to out[1,k]
    #
    # Check: for out[i,k], sum over r of U_r[i,j]*V_r[j,k]*W_r[k,i]
    # For i=0, k arbitrary:
    #   r=0: U_0[0,j]=1 if j=0 else 0, V_0[j,k]=1 if j=0, W_0[k,0]=1 -> 1*1*1 = 1 (contributes A[0,0]*B[0,k])
    #   r=1: U_1[0,j]=1 if j=1 else 0, V_1[j,k]=1 if j=1, W_1[k,0]=1 -> 1*1*1 = 1 (contributes A[0,1]*B[1,k])
    #   r=2: U_2[0,j]=0 for all j -> 0
    #   r=3: U_3[0,j]=0 for all j -> 0
    #   Total: A[0,0]*B[0,k] + A[0,1]*B[1,k] ✓
    #
    # For i=1, k arbitrary:
    #   r=0: U_0[1,j]=0 -> 0
    #   r=1: U_1[1,j]=0 -> 0
    #   r=2: U_2[1,j]=1 if j=0, V_2[j,k]=1 if j=0, W_2[k,1]=1 -> 1 (contributes A[1,0]*B[0,k])
    #   r=3: U_3[1,j]=1 if j=1, V_3[j,k]=1 if j=1, W_3[k,1]=1 -> 1 (contributes A[1,1]*B[1,k])
    #   Total: A[1,0]*B[0,k] + A[1,1]*B[1,k] ✓
    
    rank = 4
    U = [[0] * rank for _ in range(n * m)]  # (4, 4)
    V = [[0] * rank for _ in range(m * p)]  # (26, 4)
    W = [[0] * rank for _ in range(p * n)]  # (26, 4)
    
    # Term 0: i=0, j=0
    U[0 * m + 0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
    for k in range(p):
        W[k * n + 0][0] = 1
    
    # Term 1: i=0, j=1
    U[0 * m + 1][1] = 1
    for k in range(p):
        V[1 * p + k][1] = 1
    for k in range(p):
        W[k * n + 0][1] = 1
    
    # Term 2: i=1, j=0
    U[1 * m + 0][2] = 1
    for k in range(p):
        V[0 * p + k][2] = 1
    for k in range(p):
        W[k * n + 1][2] = 1
    
    # Term 3: i=1, j=1
    U[1 * m + 1][3] = 1
    for k in range(p):
        V[1 * p + k][3] = 1
    for k in range(p):
        W[k * n + 1][3] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
