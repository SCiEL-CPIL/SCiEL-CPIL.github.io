"""Matrix multiplication tensor decomposition for 2x3 by 3x6."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # The multiplication tensor: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # U is 6xR, V is 18xR, W is 12xR
    
    # Construction: We'll build a rank-20 decomposition.
    # For each (j,k) pair (18 pairs), we need to cover i=0 and i=1.
    # Naively 2 terms per (j,k) = 36 terms.
    # We can reduce by having some columns serve multiple (j,k) pairs.
    
    # Key idea: For a fixed j, the terms for different k can share structure in U.
    # U[i*m+j, c] only depends on i and j, not k.
    # So if we have a column where V has non-zeros at multiple k positions for the same j,
    # then U[i*m+j,c] is shared across those k values.
    
    # Let's try: for each j (3 values), use 2 columns that cover all 6 k values.
    # Column A_j: V[j*p+0..5, c] = [1,1,1,1,1,1], U[0*m+j,c]=1, U[1*m+j,c]=0, W[k*n+0,c]=1, W[k*n+1,c]=0
    #   This contributes 1 to C[0,k] for all k from A[0,j]*B[j,k]
    # Column B_j: V[j*p+0..5, c] = [1,1,1,1,1,1], U[0*m+j,c]=0, U[1*m+j,c]=1, W[k*n+0,c]=0, W[k*n+1,c]=1
    #   This contributes 1 to C[1,k] for all k from A[1,j]*B[j,k]
    # That's 3*2 = 6 columns, but each V entry is 1, so we get sum over j of A[i,j]*B[j,k] = C[i,k]. 
    # Wait, that's exactly what we want! But that's only 6 terms, which seems too good.
    
    # Let me check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For column A_j (j=0,1,2): U[0*m+j,c]=1, U[1*m+j,c]=0, V[j*p+k,c]=1 for all k, W[k*n+0,c]=1, W[k*n+1,c]=0
    # Contribution to (i,j',k'): U[i*m+j',c] * V[j'*p+k',c] * W[k'*n+i,c]
    # For i=0, j'=j, k'=k: 1 * 1 * 1 = 1
    # For i=1, j'=j, k'=k: 0 * 1 * 0 = 0
    # For i=0, j'!=j: U[0*m+j',c]=0 (since only U[0*m+j,c]=1), so 0
    # Similarly for other cases.
    # So column A_j contributes 1 to (i=0, j=j, k=k) for all k. That's 6 contributions per j.
    # But we need sum over ALL j for each (i,k). So we need sum_j A[i,j]*B[j,k].
    
    # Actually, this construction gives: for each (i,j,k), the sum over columns is:
    # From column A_j: contributes 1 if i=0 and j'=j and k'=k, else 0.
    # From column B_j: contributes 1 if i=1 and j'=j and k'=k, else 0.
    # So for (i=0, j, k): only column A_j contributes, giving 1. ✓
    # For (i=1, j, k): only column B_j contributes, giving 1. ✓
    # For (i=0, j', k) where j'≠j: no column contributes. ✓ (should be 0)
    # Wait, but the tensor equation requires sum_c ... = 1 for ALL i,j,k in range, not just diagonal.
    
    # I think I'm confusing myself. The tensor T[i,j,k] = 1 for all valid (i,j,k) means
    # that the bilinear map (A,B) -> C has C[i,k] = sum_j A[i,j]B[j,k].
    # The tensor decomposition represents this multilinear map.
    # The condition is: for the standard basis vectors e_i in R^n, f_j in R^m, g_k in R^p,
    # T(e_i, f_j, g_k) = 1 if and only if... no, T is the structure tensor.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for (n,m,p) has entries T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'}? No.
    # Actually, the tensor is defined such that <T, A⊗B⊗C> = trace(C^T (AB))
    # So T[i,j,k] = 1 if the (i,k) entry of AB uses A[i,j] and B[j,k], which is always true.
    # So T[i,j,k] = 1 for all i∈[0,n), j∈[0,m), k∈[0,p).
    
    # The decomposition: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = T[i,j,k] = 1 for all i,j,k.
    
    # So my construction above works! Let me verify more carefully.
    # 6 columns: for j=0,1,2 and type in {A,B}:
    # A_j: U[0*m+j]=1, others 0 in U; V[j*p+0..5]=1; W[k*n+0]=1 for all k, W[k*n+1]=0
    # B_j: U[1*m+j]=1, others 0 in U; V[j*p+0..5]=1; W[k*n+1]=1 for all k, W[k*n+0]=0
    
    # Sum over c for (i,j',k'):
    # If i=0: only A_j columns contribute. For A_j: U[0*m+j',c]=1 if j'=j, else 0.
    #   So sum = sum_j delta_{j,j'} * 1 * 1 = 1 if j'=j, but we sum over all j... 
    #   Wait, for fixed (i=0, j', k'), the sum is over c.
    #   Column A_j contributes: U[0*m+j', A_j] * V[j'*p+k', A_j] * W[k'*n+0, A_j]
    #   = (1 if j'=j else 0) * (1 if j'=j else 0... no, V[j'*p+k', A_j] is 1 only if j'=j)
    #   Actually V[A_j] has 1s at positions (j,0),(j,1),...,(j,5) where j is the specific j for that column.
    #   So V[j'*p+k', A_j] = 1 if j'=j, else 0.
    #   And W[k'*n+0, A_j] = 1 for all k'.
    #   So contribution from A_j is: (1 if j'=j) * (1 if j'=j) * 1 = 1 if j'=j, else 0.
    #   Sum over j=0,1,2: exactly one j equals j', so sum = 1. ✓
    
    # Similarly for i=1 with B_j columns.
    
    # So rank 6 works?! That seems surprisingly low. Let me double-check.
    
    # Actually wait - I need to be more careful. The tensor equation is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # With my 6-column construction:
    # For (i=0, j, k): 
    #   Sum over c in {A_0, A_1, A_2, B_0, B_1, B_2}
    #   A_j' contributes: U[0*m+j, A_j'] * V[j*p+k, A_j'] * W[k*n+0, A_j']
    #   = (1 if j=j') * (1 if j=j') * 1 = 1 if j=j', else 0
    #   B_j' contributes: U[0*m+j, B_j'] * V[j*p+k, B_j'] * W[k*n+0, B_j']
    #   = 0 * (1 if j=j') * 0 = 0
    #   Total: 1 ✓
    
    # This is correct! Rank 6 should work. But this seems too simple. Let me verify with a concrete example.
    
    # n=2, m=3, p=6. Rank=6.
    # U is 6x6, V is 18x6, W is 12x6.
    
    # Let me just build it and verify numerically.
    
    rank = 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        # Column A_j
        U[0 * m + j][col] = 1
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
        col += 1
        # Column B_j
        U[1 * m + j][col] = 1
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
        col += 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1) > 1e-10:
                    max_err = max(max_err, abs(s - 1))
    
    if max_err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # If verification failed (shouldn't happen), fall back to schoolbook
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}

