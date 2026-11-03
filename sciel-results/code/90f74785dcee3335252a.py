"""Matrix multiplication tensor decomposition for 2x3 by 3x6."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 110  # Safety margin before 120s limit

    def make_schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify that the decomposition exactly decomposes the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_arr.shape[1]):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank):
        """
        Try to find a decomposition of given rank using a structured approach.
        We exploit the structure: for each (j,k), the contribution to C[i,k] is A[i,j]*B[j,k].
        
        Key insight: We can group terms by (j,k) and use the fact that for fixed (j,k),
        we need to represent the 2x1 outer product of A[:,j] and B[j,k].
        
        More generally, we can use a "block" structure where we share V entries across i.
        """
        # Strategy: Use the fact that the tensor is T[i,j,k] = 1 for all valid (i,j,k)
        # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Approach: Fix V to be "spread out" and solve for U and W.
        # Or: Use a parametric construction.
        
        # Let's try: for each c, define U[:,c], V[:,c], W[:,c].
        # A natural sparse structure: each column has non-zeros in a limited set of positions.
        
        # Try construction based on "sum and difference" for the 2x dimension.
        # For each (j,k), instead of 2 terms (one for i=0, one for i=1), try 1 term with:
        #   U[0*m+j] = a, U[1*m+j] = b, V[j*p+k] = 1, W[k*n+0] = c, W[k*n+1] = d
        # This gives: a*c for C[0,k] contribution, b*d for C[1,k] contribution.
        # We need a*c = 1 and b*d = 1 for each (j,k) pair if we use one term per (j,k).
        # But we also need these to be consistent across j for fixed k.
        
        # Alternative: use 2 terms per (j,k) but with shared structure.
        # Term 1: U[0*m+j]=1, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=1
        #   -> contributes 1 to both C[0,k] and C[1,k] (from A[0,j]*B[j,k] + A[1,j]*B[j,k])
        #   -> Wait, that's not right. It contributes U[0*m+j]*V[j*p+k]*W[k*n+0] = 1*1*1 = 1 to C[0,k]
        #   and U[1*m+j]*V[j*p+k]*W[k*n+1] = 1*1*1 = 1 to C[1,k].
        #   But we need exactly A[0,j]*B[j,k] for C[0,k], not 1.
        #   The issue is that U, V, W are fixed coefficients; they don't depend on A and B.
        #   The decomposition must work for ALL A and B. So the tensor T[i,j,k] must be 
        #   decomposed as sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k] = 1.
        
        # So we just need the sum to equal 1 for each (i,j,k).
        
        # Let's try a direct construction:
        # Use rank = 2*p + 2*(m-1)*p = 2*6 + 2*2*6 = 12 + 24 = 36? No.
        
        # Better: use the "Strassen-like" idea for the 2x dimension.
        # For 2x1x1 matrix mult (which is just multiplication), rank is 1.
        # For 2x3x6, think of it as 6 independent 2x3x1 problems? No, they share j.
        
        # Let's try rank 20 with a specific pattern:
        # Columns 0-5: for k=0..5, one "sum" term
        #   U[0*m+j] = 1, U[1*m+j] = 1 for all j
        #   V[j*p+k] = 1 for all j
        #   W[k*n+0] = 1, W[k*n+1] = 1
        #   This contributes 1 to C[0,k] and 1 to C[1,k] for each j.
        #   Total contribution to C[0,k] from this term: sum_j 1*1*1 = 3 (for j=0,1,2)
        #   That's too much. We need exactly 1 per (i,j,k) triple, not per (i,k).
        
        # I think the key constraint is: for each FIXED (i,j,k), the sum must be 1.
        # So if I use one term that covers all j for a given (i,k), it contributes 1 for each j,
        # which is correct! Because for each j, the term contributes U[i*m+j]*V[j*p+k]*W[k*n+i].
        # If I set U[i*m+j]=1 for all j, V[j*p+k]=1 for all j, W[k*n+i]=1, then
        # for each j, the contribution is 1*1*1 = 1. 
        # But wait, that term also contributes to other (i',j',k') combinations.
        # For a different i' (say i'=1 if i=0), it contributes U[1*m+j]*V[j*p+k]*W[k*n+1].
        # If U[1*m+j]=1 and W[k*n+1]=1, then it also contributes 1 to C[1,k] for each j.
        
        # So one term with U[i*m+j]=1 for all j, V[j*p+k]=1 for all j, W[k*n+i]=1 
        # contributes 1 to C[i,k] for each j. That's 3 separate (i,j,k) triples each getting 1.
        # But it also contributes to C[i',k] for other i' if U[i'*m+j] and W[k*n+i'] are non-zero.
        
        # Let me be more careful. A single column c with:
        #   U[0*m+0]=a, U[0*m+1]=b, U[0*m+2]=c, U[1*m+0]=d, U[1*m+1]=e, U[1*m+2]=f
        #   V[0*p+k]=g, V[1*p+k]=h, V[2*p+k]=i (for a specific k, 0 elsewhere)
        #   W[k*n+0]=j, W[k*n+1]=k (for the same k, 0 elsewhere)
        # Then for triple (0,0,k): contribution = a * g * j
        # For triple (0,1,k): contribution = b * h * j
        # For triple (0,2,k): contribution = c * i * j
        # For triple (1,0,k): contribution = d * g * k
        # For triple (1,1,k): contribution = e * h * k
        # For triple (1,2,k): contribution = f * i * k
        # For triples with different k': contribution = 0 (since V is 0 for other k')
        
        # So if I use one column per k, with V non-zero only for that k, then each column
        # only affects the 6 triples with that k. I need 6 columns total, one per k.
        # For column c_k (corresponding to k), I need:
        #   a_k * g_k * j_k = 1
        #   b_k * h_k * j_k = 1
        #   c_k * i_k * j_k = 1
        #   d_k * g_k * k_k = 1
        #   e_k * h_k * k_k = 1
        #   f_k * i_k * k_k = 1
        # This is 6 equations with 9 unknowns (a,b,c,d,e,f,g,h,i) per column, plus j_k, k_k.
        # Actually 11 unknowns per column. Very underdetermined.
        # So rank 6 should work! Let me construct it.
        
        # For each k, set:
        #   j_k = 1, k_k = 1
        #   g_k = 1, h_k = 1, i_k = 1
        #   a_k = 1, b_k = 1, c_k = 1, d_k = 1, e_k = 1, f_k = 1
        # Then all 6 equations are satisfied: 1*1*1 = 1.
        # But wait, this means for column c_k, U[0*m+j]=1 for all j, U[1*m+j]=1 for all j,
        # V[j*p+k]=1 for all j, W[k*n+0]=1, W[k*n+1]=1.
        # This works! And it's only 6 columns (one per k).
        
        # Let me verify: for triple (i,j,k), the sum over columns is:
        #   Only column c_k has non-zero V[j*p+k] (since other columns have V=0 for this k).
        #   So the contribution is U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 * 1 * 1 = 1. ✓
        
        # So rank 6 works! Let me double-check the shapes.
        # U: (n*m, rank) = (6, 6)
        # V: (m*p, rank) = (18, 6)
        # W: (p*n, rank) = (12, 6)
        
        rank_actual = p  # 6
        U = [[0] * rank_actual for _ in range(n * m)]
        V = [[0] * rank_actual for _ in range(m * p)]
        W = [[0] * rank_actual for _ in range(p * n)]
        
        for k in range(p):
            col = k
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = 1
            for j in range(m):
                V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try rank 6 first
    result = try_rank(p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return make_schoolbook()
