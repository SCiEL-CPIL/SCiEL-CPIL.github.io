"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2(U_rows, V_cols, W_rows, U_start, V_start, W_start, U, V, W, rank_offset):
    """Apply Strassen's 2x2 decomposition to a sub-block.
    
    U_rows: list of 2 U-row indices (i*m+j values)
    V_cols: list of 2 V-col indices (j*p+k values)  
    W_rows: list of 2 W-row indices (k*n+i values)
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = DIMENSIONS
    u0, u1 = U_rows
    v0, v1 = V_cols
    w0, w1 = W_rows
    
    # We need to map (i,j) -> u_index, (j,k) -> v_index, (k,i) -> w_index
    # For a 2x2 block, we have 4 (i,j,k) combinations
    # Let's figure out the actual indices
    
    # u0 = i0*m + j0, u1 = i1*m + j1
    # v0 = j0*p + k0, v1 = j1*p + k1  -- but this doesn't work for 2x2
    # 
    # Actually for a 2x2 sub-multiplication, we need:
    # A is 2x2: indices (i,j) for i in {i0,i1}, j in {j0,j1}
    # B is 2x2: indices (j,k) for j in {j0,j1}, k in {k0,k1}
    # C is 2x2: indices (i,k) for i in {i0,i1}, k in {k0,k1}
    #
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    i0, j0 = divmod(u0, m)
    i1, j1 = divmod(u1, m)
    j0v, k0 = divmod(v0, p)
    j1v, k1 = divmod(v1, p)
    k0w, i0w = divmod(w0, n)
    k1w, i1w = divmod(w1, n)
    
    # For a proper 2x2 block, we need:
    # i0 == i0w, i1 == i1w, j0 == j0v, j1 == j1v, k0 == k0w, k1 == k1w
    # And the four (i,j,k) combos are: (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1), etc.
    # Wait, no. For 2x2: i in {i0,i1}, j in {j0,j1}, k in {k0,k1}
    # The tensor entries are T[i,j,k] for all 8 combinations? No, T[i,j,k] = 1 for all.
    # 
    # Actually the tensor is T[i,j,k] = 1 for all valid (i,j,k). The decomposition
    # must satisfy: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    #
    # For a 2x2 block with i in {i0,i1}, j in {j0,j1}, k in {k0,k1}:
    # We have 8 tensor entries, all = 1.
    # U indices: i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1 (4 values)
    # V indices: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1 (4 values)
    # W indices: k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1 (4 values)
    
    u_indices = [i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1]
    v_indices = [j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1]
    w_indices = [k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1]
    
    # Strassen: 7 terms instead of 8
    # Term 1: M1 = (A11+A22)*(B11+B22)
    #   A11 = A[i0,j0], A22 = A[i1,j1]
    #   B11 = B[j0,k0], B22 = B[j1,k1]
    #   Contribution to C: C11 += M1, C22 += M1
    #   U: u[i0,j0]=1, u[i1,j1]=1, others 0
    #   V: v[j0,k0]=1, v[j1,k1]=1, others 0
    #   W: w[k0,i0]=1, w[k1,i1]=1, others 0
    #   But this gives T[i0,j0,k0] += 1, T[i0,j0,k1] += 0 (V[j0,k1]=0), etc.
    #   Actually T[i,j,k] = U[i*m+j]*V[j*p+k]*W[k*n+i]
    #   For (i0,j0,k0): U[i0*m+j0]*V[j0*p+k0]*W[k0*n+i0] = 1*1*1 = 1
    #   For (i0,j0,k1): U[i0*m+j0]*V[j0*p+k1]*W[k1*n+i0] = 1*0*0 = 0
    #   For (i0,j1,k0): U[i0*m+j1]*V[j1*p+k0]*W[k0*n+i0] = 0*0*1 = 0
    #   For (i0,j1,k1): U[i0*m+j1]*V[j1*p+k1]*W[k1*n+i0] = 0*1*0 = 0
    #   For (i1,j0,k0): U[i1*m+j0]*V[j0*p+k0]*W[k0*n+i1] = 0*1*0 = 0
    #   For (i1,j0,k1): U[i1*m+j0]*V[j0*p+k1]*W[k1*n+i1] = 0*0*1 = 0
    #   For (i1,j1,k0): U[i1*m+j1]*V[j1*p+k0]*W[k0*n+i1] = 1*0*0 = 0
    #   For (i1,j1,k1): U[i1*m+j1]*V[j1*p+k1]*W[k1*n+i1] = 1*1*1 = 1
    #   So this term contributes 1 to T[i0,j0,k0] and T[i1,j1,k1], 0 elsewhere.
    #   But we need all 8 entries to sum to 1 across all terms.
    #
    # This is the standard Strassen decomposition. Let me just implement the 7 terms.
    
    terms = []
    # M1 = (A11+A22)*(B11+B22): U: {u00,u11}, V: {v00,v11}, W: {w00,w11}
    terms.append(([u_indices[0], u_indices[3]], [v_indices[0], v_indices[3]], [w_indices[0], w_indices[3]], 1))
    # M2 = (A21+A22)*B11: U: {u10,u11}, V: {v00}, W: {w00,w01}
    #   A21 = A[i1,j0], A22 = A[i1,j1], B11 = B[j0,k0]
    #   C21 += M2, C22 += M2... 
    #   Actually: C21 = M2+M4, C22 = M1-M2+M3+M6
    #   M2 contributes to C21 and C22? Let me re-derive.
    #   Standard Strassen:
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #   M2 = (A21+A22)*B11, so it contributes to C21 (+1) and C22 (-1)
    #   As a tensor term: U[i1,j0]=1, U[i1,j1]=1, V[j0,k0]=1, W[k0,i1]=1, W[k1,i1]=-1
    #   T[i1,j0,k0] += 1*1*1 = 1
    #   T[i1,j1,k0] += 1*0*1 = 0 (V[j1,k0]=0)
    #   T[i1,j0,k1] += 1*0*(-1) = 0
    #   T[i1,j1,k1] += 1*0*(-1) = 0
    #   Hmm, that only gives 1 to T[i1,j0,k0]. But M2 should contribute to C21 and C22.
    #   C21 = C[i1,k0], C22 = C[i1,k1]
    #   T[i1,j,k] for j in {j0,j1}, k in {k0,k1}
    #   The term contributes to T[i1,j0,k0] = 1, and we need it to also affect C22.
    #   I think I'm confusing myself. Let me just use the standard formulation.
    
    # Let me use a cleaner approach. The 7 Strassen terms for 2x2:
    # Define A11=A[i0,j0], A12=A[i0,j1], A21=A[i1,j0], A22=A[i1,j1]
    # B11=B[j0,k0], B12=B[j0,k1], B21=B[j1,k0], B22=B[j1,k1]
    
    # M1 = (A11+A22)*(B11+B22)
    #   U: A11, A22 -> U[i0*m+j0]=1, U[i1*m+j1]=1
    #   V: B11, B22 -> V[j0*p+k0]=1, V[j1*p+k1]=1
    #   W: C11, C22 -> W[k0*n+i0]=1, W[k1*n+i1]=1
    #   T[i0,j0,k0] += 1, T[i0,j0,k1] += 0, T[i0,j1,k0] += 0, T[i0,j1,k1] += 0
    #   T[i1,j0,k0] += 0, T[i1,j0,k1] += 0, T[i1,j1,k0] += 0, T[i1,j1,k1] += 1
    
    # M2 = (A21+A22)*B11
    #   U: A21, A22 -> U[i1*m+j0]=1, U[i1*m+j1]=1
    #   V: B11 -> V[j0*p+k0]=1
    #   W: C21, C22 -> W[k0*n+i1]=1, W[k1*n+i1]=-1
    #   T[i1,j0,k0] += 1*1*1 = 1
    #   T[i1,j0,k1] += 1*1*(-1) = -1... but V[j0*p+k1] = 0, so T[i1,j0,k1] += 0
    #   T[i1,j1,k0] += 1*0*1 = 0
    #   T[i1,j1,k1] += 1*0*(-1) = 0
    
    # This is getting error-prone. Let me just hardcode a known good 3x5x7 decomposition
    # or use a verified Strassen-based construction.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
