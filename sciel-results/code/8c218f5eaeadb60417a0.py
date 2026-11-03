"""Tensor decomposition for 2x3 matrix times 3x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    We use a structured construction that exploits the linearity in the
    output dimension (n=2) to reduce rank from the schoolbook 36.
    
    The key insight: for each k, we can compute both C[0,k] and C[1,k]
    using a shared "sum" term plus two correction terms, giving 3 terms
    per k value instead of 6. This yields rank 18.
    
    However, we can do better by grouping k values. Let's try a direct
    construction with rank 20 using a more clever sharing pattern.
    
    Actually, let me construct a valid rank-18 decomposition first, then
    see if we can improve to rank 20 with a different structure.
    
    For rank 18:
    For each k in 0..5:
      Term 3k:   Sum term - contributes to both rows
      Term 3k+1: Correction for row 0
      Term 3k+2: Correction for row 1
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    Let me verify the rank-18 construction:
    For term 3k (sum term):
      U[i*3+j, 3k] = 1 for all i in {0,1}, j in {0,1,2}
      V[j*6+k, 3k] = 1 for all j in {0,1,2}
      W[k*2+0, 3k] = 1, W[k*2+1, 3k] = 1
    
    For term 3k+1 (correction for row 0):
      U[0*3+j, 3k+1] = 1 for all j, U[1*3+j, 3k+1] = 0 for all j
      V[j*6+k, 3k+1] = 1 for all j
      W[k*2+0, 3k+1] = -1, W[k*2+1, 3k+1] = 0
    
    For term 3k+2 (correction for row 1):
      U[0*3+j, 3k+2] = 0 for all j, U[1*3+j, 3k+2] = 1 for all j
      V[j*6+k, 3k+2] = 1 for all j
      W[k*2+0, 3k+2] = 0, W[k*2+1, 3k+2] = -1
    
    Verification for T[i,j,k] = sum_c U[i*3+j,c] * V[j*6+k,c] * W[k*2+i,c]:
    
    For i=0, k=k:
    Sum over c:
      c=3k: U[0*3+j,3k]*V[j*6+k,3k]*W[k*2+0,3k] = 1*1*1 = 1
      c=3k+1: U[0*3+j,3k+1]*V[j*6+k,3k+1]*W[k*2+0,3k+1] = 1*1*(-1) = -1
      c=3k+2: U[0*3+j,3k+2]*V[j*6+k,3k+2]*W[k*2+0,3k+2] = 0*1*0 = 0
      Other c: 0
    Sum = 1 - 1 + 0 = 0. WRONG! Should be 1.
    
    Let me fix the signs. We want:
    T[0,j,k] = 1
    T[1,j,k] = 1
    
    Let me try:
    Term 3k: U all 1s, V all 1s (for j), W[k*2+0]=1, W[k*2+1]=1
    Term 3k+1: U[0*3+j]=1, U[1*3+j]=0, V=1, W[k*2+0]=-1, W[k*2+1]=0
    Term 3k+2: U[0*3+j]=0, U[1*3+j]=1, V=1, W[k*2+0]=0, W[k*2+1]=-1
    
    T[0,j,k] = 1*1*1 + 1*1*(-1) + 0*1*0 = 1 - 1 = 0. Still wrong.
    
    I need to rethink. Let me use:
    Term 3k: U all 1s, V all 1s, W[k*2+0]=1, W[k*2+1]=1
    Term 3k+1: U[0*3+j]=1, U[1*3+j]=0, V=1, W[k*2+0]=0, W[k*2+1]=1
    Term 3k+2: U[0*3+j]=0, U[1*3+j]=1, V=1, W[k*2+0]=1, W[k*2+1]=0
    
    T[0,j,k] = 1*1*1 + 1*1*0 + 0*1*1 = 1 + 0 + 0 = 1. CORRECT!
    T[1,j,k] = 1*1*1 + 0*1*1 + 1*1*0 = 1 + 0 + 0 = 1. CORRECT!
    
    So rank 18 works! But can we do better?
    
    Let me try to find a rank-20 construction that might be more efficient,
    or see if rank 18 can be reduced further.
    
    Actually, rank 18 is already an improvement over 36. Let me try to see
    if we can get to rank 20 with a different structure, but rank 18 seems
    valid. Let me implement it.
    
    Wait, I should double-check. The tensor equation is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    
    With my construction:
    For a fixed (i,j,k), only terms c=3k, 3k+1, 3k+2 can contribute
    (since V[j*6+k, c] is non-zero only when c is in {3k, 3k+1, 3k+2}
    for the given k... wait, no. V[j*6+k, c] depends on k, so for a
    fixed k, only terms with that k in V can contribute.
    
    Actually, V[j*6+k, c] is non-zero for term c only if the k-index in
    V matches. In my construction, term 3k has V[j*6+k, 3k] = 1, and
    V[j*6+k', 3k] = 0 for k' != k. So for a fixed (i,j,k), only terms
    3k, 3k+1, 3k+2 contribute to V[j*6+k, c].
    
    So the sum is over c=3k, 3k+1, 3k+2 only:
    T[i,j,k] = U[i*3+j,3k]*V[j*6+k,3k]*W[k*2+i,3k] 
             + U[i*3+j,3k+1]*V[j*6+k,3k+1]*W[k*2+i,3k+1]
             + U[i*3+j,3k+2]*V[j*6+k,3k+2]*W[k*2+i,3k+2]
    
    For i=0:
    = 1*1*1 + 1*1*0 + 0*1*1 = 1
    For i=1:
    = 1*1*1 + 0*1*1 + 1*1*0 = 1
    
    Both equal 1. So rank 18 is valid!
    
    Can we do better? The theoretical lower bound for 2x3x6 is at least
    2*3 = 6 (from the slice rank bound) and at most 36 (schoolbook).
    The true rank is believed to be around 20, but 18 is already better.
    
    Let me implement rank 18.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    
    for k in range(p):
        # Term 3k: sum term
        c = 3 * k
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = 1
        for j in range(m):
            V[j * p + k][c] = 1
        W[k * n + 0][c] = 1
        W[k * n + 1][c] = 1
        
        # Term 3k+1: correction for row 0
        c = 3 * k + 1
        for j in range(m):
            U[0 * m + j][c] = 1
            # U[1 * m + j][c] = 0 (already 0)
            V[j * p + k][c] = 1
        W[k * n + 0][c] = 0
        W[k * n + 1][c] = 1
        
        # Term 3k+2: correction for row 1
        c = 3 * k + 2
        for j in range(m):
            # U[0 * m + j][c] = 0 (already 0)
            U[1 * m + j][c] = 1
            V[j * p + k][c] = 1
        W[k * n + 0][c] = 1
        W[k * n + 1][c] = 0
    
    # Verify the decomposition
    # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    raise ValueError(f"Verification failed at i={i}, j={j}, k={k}: got {total}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
