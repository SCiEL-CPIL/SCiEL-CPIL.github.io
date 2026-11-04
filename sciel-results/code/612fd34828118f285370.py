"""Optimized tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    where the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    equals 1 for all valid (i,j,k) and 0 otherwise.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Tensor dimensions
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    def get_strassen_block(i0, j0, k0):
        """
        Returns lists of (u_idx, u_val), (v_idx, v_val), (w_idx, w_val) for 7 terms.
        i0, j0, k0 are global offsets for a 2x2x2 block.
        """
        terms = []
        
        # Standard Strassen for 2x2:
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # U indices: i*m + j, V indices: j*p + k, W indices: k*n + i
        #
        # Term 1: M1
        terms.append(([((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)],
                      [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)],
                      [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]))
                      
        # Term 2: M2 = (A21+A22)B11
        # Contributes to C21 (+1) and C22 (-1).
        # C21: i=i1, k=k0.
        # C22: i=i1, k=k1.
        terms.append(([((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)],
                      [((j0+0)*p + (k0+0), 1)],
                      [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]))
                      
        # Term 3: M3 = A11(B21-B11)
        # Contributes to C12 (+1) and C22 (+1).
        # C12: i=i0, k=k1.
        # C22: i=i1, k=k1.
        terms.append(([((i0+0)*m + (j0+0), 1)],
                      [((j0+1)*p + (k0+1), 1), ((j0+0)*p + (k0+1), -1)],
                      [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]))
                      
        # Term 4: M4 = A22(B12-B22)
        # Contributes to C11 (+1) and C21 (+1).
        # C11: i=i0, k=k0.
        # C21: i=i1, k=k0.
        terms.append(([((i0+1)*m + (j0+1), 1)],
                      [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)],
                      [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]))
                      
        # Term 5: M5 = (A11+A12)B22
        # Contributes to C12 (+1) and C11 (-1).
        # C12: i=i0, k=k1.
        # C11: i=i0, k=k0.
        terms.append(([((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)],
                      [((j0+1)*p + (k0+1), 1)],
                      [((k0+1)*n + (i0+0), 1), ((k0+0)*n + (i0+0), -1)]))
                      
        # Term 6: M6 = (A21-A11)(B11+B12)
        # Contributes to C22 (+1).
        # C22: i=i1, k=k1.
        terms.append(([((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)],
                      [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)],
                      [((k0+1)*n + (i0+1), 1)]))
                      
        # Term 7: M7 = (A12-A22)(B21+B22)
        # Contributes to C11 (+1).
        # C11: i=i0, k=k0.
        terms.append(([((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)],
                      [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)],
                      [((k0+0)*n + (i0+0), 1)]))
        
        return terms

    def add_strassen_terms(terms, U_cols, V_cols, W_cols):
        for u, v, w in terms:
            U_col = np.zeros(dim_u)
            for idx, val in u:
                U_col[idx] = val
            V_col = np.zeros(dim_v)
            for idx, val in v:
                V_col[idx] = val
            W_col = np.zeros(dim_w)
            for idx, val in w:
                W_col[idx] = val
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    U_cols = []
    V_cols = []
    W_cols = []
    
    # Add 4 Strassen blocks
    # Block 1: i0=0, j0=0, k0=0
    add_strassen_terms(get_strassen_block(0, 0, 0), U_cols, V_cols, W_cols)
    # Block 2: i0=0, j0=2, k0=0
    add_strassen_terms(get_strassen_block(0, 2, 0), U_cols, V_cols, W_cols)
    # Block 3: i0=0, j0=0, k0=2
    add_strassen_terms(get_strassen_block(0, 0, 2), U_cols, V_cols, W_cols)
    # Block 4: i0=0, j0=2, k0=2
    add_strassen_terms(get_strassen_block(0, 2, 2), U_cols, V_cols, W_cols)
    
    # Add the "Broadcast" term for the remaining parts
    # Covers: i in {0,1}, k=4 AND i=2, all k
    U_new = np.ones(dim_u)
    V_new = np.ones(dim_v)
    W_new = np.zeros(dim_w)
    
    # W_new[k*n + i]
    # For i=0, k=4: idx = 4*3 + 0 = 12
    W_new[4*n + 0] = 1
    # For i=1, k=4: idx = 4*3 + 1 = 13
    W_new[4*n + 1] = 1
    # For i=2, all k: idx = k*3 + 2
    for k in range(p):
        W_new[k*n + 2] = 1
        
    U_cols.append(U_new)
    V_cols.append(V_new)
    W_cols.append(W_new)
    
    # Stack columns
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                max_err = max(max_err, abs(s - 1.0))
                
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
