"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Build the tensor decomposition using block Strassen
    # A is n x m = 3 x 5, B is m x p = 5 x 5, C is n x p = 3 x 5
    # Split: rows of A/C: [2, 1], cols of A/rows of B: [2, 2, 1], cols of B/C: [2, 2, 1]
    
    # Row splits for A and C: 3 = 2 + 1
    row_splits_A = [0, 2, 3]  # block boundaries
    # Col splits for A and row splits for B: 5 = 2 + 2 + 1
    col_splits_A = [0, 2, 4, 5]
    # Col splits for B and C: 5 = 2 + 2 + 1
    col_splits_B = [0, 2, 4, 5]
    
    # Number of row blocks for A/C: 2
    # Number of col blocks for A/row blocks for B: 3
    # Number of col blocks for B/C: 3
    
    U_terms = []  # list of (i, j, value) for U
    V_terms = []  # list of (j, k, value) for V
    W_terms = []  # list of (k, i, value) for W
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict: {(i,j): val}, v_dict: {(j,k): val}, w_dict: {(k,i): val}"""
        U_terms.append(u_dict)
        V_terms.append(v_dict)
        W_terms.append(w_dict)
    
    # Process each block combination
    for ri in range(len(row_splits_A) - 1):
        for cj in range(len(col_splits_A) - 1):
            for ck in range(len(col_splits_B) - 1):
                # Block sizes
                nr = row_splits_A[ri+1] - row_splits_A[ri]
                nc = col_splits_A[cj+1] - col_splits_A[cj]
                nk = col_splits_B[ck+1] - col_splits_B[ck]
                
                # Offsets
                i0 = row_splits_A[ri]
                j0 = col_splits_A[cj]
                k0 = col_splits_B[ck]
                
                # Block A[i0:i0+nr, j0:j0+nc] @ B[j0:j0+nc, k0:k0+nk] -> C[i0:i0+nr, k0:k0+nk]
                # Only valid if nc == nk (inner dimension match)
                if nc != nk:
                    continue
                
                # For 2x2x2 blocks, use Strassen (rank 7)
                # For other sizes, use schoolbook
                if nr == 2 and nc == 2 and nk == 2:
                    # Strassen decomposition for 2x2x2
                    # The 7 terms of Strassen's algorithm
                    # Each term: u[i,j] * v[j,k] * w[i,k] where i,j,k in {0,1}
                    # (local indices)
                    
                    # Term 1: (A00+A11)(B00+B11) -> C00, C11
                    u1 = {(0,0): 1, (1,1): 1}
                    v1 = {(0,0): 1, (1,1): 1}
                    w1 = {(0,0): 1, (1,1): 1}
                    add_term(
                        {(i0+ii, j0+jj): val for (ii,jj), val in u1.items()},
                        {(j0+jj, k0+kk): val for (jj,kk), val in v1.items()},
                        {(k0+kk, i0+ii): val for (kk,ii), val in w1.items()}
                    )
                    
                    # Term 2: (A10+A11)B00 -> C10
                    u2 = {(1,0): 1, (1,1): 1}
                    v2 = {(0,0): 1}
                    w2 = {(0,1): 1}
                    add_term(
                        {(i0+ii, j0+jj): val for (ii,jj), val in u2.items()},
                        {(j0+jj, k0+kk): val for (jj,kk), val in v2.items()},
                        {(k0+kk, i0+ii): val for (kk,ii), val in w2.items()}
                    )
                    
                    # Term 3: A00(B10-B11) -> C01
                    u3 = {(0,0): 1}
                    v3 = {(1,0): 1, (1,1): -1}
                    w3 = {(1,0): 1}
                    add_term(
                        {(i0+ii, j0+jj): val for (ii,jj), val in u3.items()},
                        {(j0+jj, k0+kk): val for (jj,kk), val in v3.items()},
                        {(k0+kk, i0+ii): val for (kk,ii), val in w3.items()}
                    )
                    
                    # Term 4: A11(B01-B00) -> C01
                    u4 = {(1,1): 1}
                    v4 = {(0,1): 1, (0,0): -1}
                    w4 = {(1,0): 1}
                    add_term(
                        {(i0+ii, j0+jj): val for (ii,jj), val in u4.items()},
                        {(j0+jj, k0+kk): val for (jj,kk), val in v4.items()},
                        {(k0+kk, i0+ii): val for (kk,ii), val in w4.items()}
                    )
                    
                    # Term 5: (A00+A01)B11 -> C00
                    u5 = {(0,0): 1, (0,1): 1}
                    v5 = {(1,1): 1}
                    w5 = {(0,0): 1}
                    add_term(
                        {(i0+ii, j0+jj): val for (ii,jj), val in u5.items()},
                        {(j0+jj, k0+kk): val for (jj,kk), val in v5.items()},
                        {(k0+kk, i0+ii): val for (kk,ii), val in w5.items()}
                    )
                    
                    # Term 6: (A10-A00)(B00+B01) -> C10
                    u6 = {(1,0): 1, (0,0): -1}
                    v6 = {(0,0): 1, (0,1): 1}
                    w6 = {(0,1): 1}
                    add_term(
                        {(i0+ii, j0+jj): val for (ii,jj), val in u6.items()},
                        {(j0+jj, k0+kk): val for (jj,kk), val in v6.items()},
                        {(k0+kk, i0+ii): val for (kk,ii), val in w6.items()}
                    )
                    
                    # Term 7: (A01-A11)(B10+B11) -> C11
                    u7 = {(0,1): 1, (1,1): -1}
                    v7 = {(1,0): 1, (1,1): 1}
                    w7 = {(1,1): 1}
                    add_term(
                        {(i0+ii, j0+jj): val for (ii,jj), val in u7.items()},
                        {(j0+jj, k0+kk): val for (jj,kk), val in v7.items()},
                        {(k0+kk, i0+ii): val for (kk,ii), val in w7.items()}
                    )
                else:
                    # Schoolbook for this block
                    for ii in range(nr):
                        for jj in range(nc):
                            for kk in range(nk):
                                add_term(
                                    {(i0+ii, j0+jj): 1},
                                    {(j0+jj, k0+kk): 1},
                                    {(k0+kk, i0+ii): 1}
                                )
    
    rank = len(U_terms)
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for (i, j), val in U_terms[r].items():
            U[i * m + j, r] = val
        for (j, k), val in V_terms[r].items():
            V[j * p + k, r] = val
        for (k, i), val in W_terms[r].items():
            W[k * n + i, r] = val
    
    # Verify the decomposition
    # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-10:
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
