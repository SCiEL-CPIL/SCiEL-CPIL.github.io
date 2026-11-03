"""Matrix multiplication 2x4x7 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 7]
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    # Tensor T[i, j, k] = 1 for the multiplication (i-th row of A, j-th col of B -> k-th col of result)
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # So tensor entry T[i,j,k] = 1 (it's a 0-1 tensor for standard matrix mult)
    
    # Build the full tensor as a 3D array
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    
    # Schoolbook decomposition: rank = n*m*p = 56
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # We want to find a lower rank decomposition.
    
    # Strategy: Try to find a decomposition of rank r < 56
    # The tensor can be viewed as a linear map from (U ⊗ V) to W
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # For exact decomposition with integer/half-integer entries,
    # we can use the fact that the tensor has structure.
    
    # Let's try a direct approach: use the known result that for 2x4x7,
    # we can potentially achieve rank less than 56.
    
    # First, let's implement the schoolbook as baseline
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=object)
    V_school = np.zeros((m*p, rank_school), dtype=object)
    W_school = np.zeros((p*n, rank_school), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # We can try to see if some columns are linear combinations of others
    
    # Alternative: Use a more clever decomposition
    # For 2x4x7, let's try to exploit that n=2 is small
    
    # The tensor T has the property that for fixed i, the slice T[i,:,:] is a 4x7 matrix
    # T[i,j,k] = 1 for all j,k, so each slice is all-ones
    
    # This means: sum_j A[i,j] * B[j,k] for each i,k
    # For i=0: sum_j A[0,j] * B[j,k]
    # For i=1: sum_j A[1,j] * B[j,k]
    
    # We can write this as:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # This is just two independent 1x4x7 matrix multiplications
    # Each 1x4x7 has rank 4*7 = 28 in schoolbook
    # Total schoolbook rank = 2*28 = 56
    
    # Can we do better? For 1xm x pxq, the rank is m*p in general (no known improvement)
    # But for 2x4x7, maybe we can share some terms between the two rows?
    
    # Let's try: Can we find a decomposition where some rank-1 terms serve both i=0 and i=1?
    # A rank-1 term u ⊗ v ⊗ w contributes to T[i,j,k] as u[i]*v[j]*w[k]
    # For it to contribute to both i=0 and i=1, we need u[0] and u[1] both nonzero
    # But then it would add to both C[0,k] and C[1,k] with the same v[j]*w[k] pattern
    # This is only useful if the same pattern appears in both rows
    
    # Since all entries are 1, every (j,k) pair appears in both rows
    # So we might be able to use terms where u = [1,1] or similar
    
    # Let's try to construct a lower rank decomposition
    
    # Idea: Use terms with u = [1, 1] to cover both rows simultaneously
    # A term with u=[1,1], v=[e_j], w=[e_k] contributes 1 to T[0,j,k] and T[1,j,k]
    # This is exactly what we need for each (j,k) pair!
    
    # So we can use 4*7 = 28 terms with u=[1,1], v=e_j, w=e_k
    # This gives rank 28 instead of 56!
    
    # Let's verify: 
    # For term (j,k): u=[1,1], v has 1 at position j, w has 1 at position k
    # Contribution to T[i,j',k'] = u[i] * v[j'] * w[k']
    # = 1 * (1 if j'=j else 0) * (1 if k'=k else 0)
    # = 1 if (j',k') = (j,k) else 0
    # This is correct for all i!
    
    # So rank 28 works!
    rank = m * p  # 28
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # u = [1, 1] for the two i values
            U[0*m + j, col] = 1  # i=0, j
            U[1*m + j, col] = 1  # i=1, j
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1  # k, i=0
            W[k*n + 1, col] = 1  # k, i=1
            col += 1
    
    # Verify the decomposition
    # T_reconstructed[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_rec = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                T_rec[i, j, k] = s
    
    # Check if it matches
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T_rec[i, j, k] != T[i, j, k]:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        # Convert to list format
        U_list = [[U[i, j] for j in range(rank)] for i in range(n*m)]
        V_list = [[V[i, j] for j in range(rank)] for i in range(m*p)]
        W_list = [[W[i, j] for j in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = [[U_school[i, j] for j in range(rank_school)] for i in range(n*m)]
        V_list = [[V_school[i, j] for j in range(rank_school)] for i in range(m*p)]
        W_list = [[W_school[i, j] for j in range(rank_school)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
