import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Dimensions for the factor matrices
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        """
        Add a single rank-1 term to the decomposition.
        u_sparse, v_sparse, w_sparse are dicts: {index: value}
        """
        U_col = np.zeros(dim_u, dtype=int)
        V_col = np.zeros(dim_v, dtype=int)
        W_col = np.zeros(dim_w, dtype=int)
        
        for idx, val in u_sparse.items():
            U_col[idx] = val
        for idx, val in v_sparse.items():
            V_col[idx] = val
        for idx, val in w_sparse.items():
            W_col[idx] = val
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def add_strassen_block(i0, j0, k0):
        """
        Adds 7 terms corresponding to Strassen's algorithm for a 2x2x2 block
        starting at global indices i0, j0, k0.
        This covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
        """
        # Indices
        i1, i2 = i0, i0 + 1
        j1, j2 = j0, j0 + 1
        k1, k2 = k0, k0 + 1
        
        # Map global (i,j) to U index, (j,k) to V index, (k,i) to W index
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i

        # Term 1: M1 = (A11+A22)(B11+B22)
        # U: A11, A22 -> 1
        # V: B11, B22 -> 1
        # W: C11, C22 -> 1
        u1 = {u_idx(i1, j1): 1, u_idx(i2, j2): 1}
        v1 = {v_idx(j1, k1): 1, v_idx(j2, k2): 1}
        w1 = {w_idx(k1, i1): 1, w_idx(k2, i2): 1}
        add_term(u1, v1, w1)

        # Term 2: M2 = (A21+A22)B11
        # U: A21, A22 -> 1
        # V: B11 -> 1
        # W: C21 -> 1, C22 -> -1
        u2 = {u_idx(i2, j1): 1, u_idx(i2, j2): 1}
        v2 = {v_idx(j1, k1): 1}
        w2 = {w_idx(k1, i2): 1, w_idx(k2, i2): -1}
        add_term(u2, v2, w2)

        # Term 3: M3 = A11(B21-B11)
        # U: A11 -> 1
        # V: B21 -> 1, B11 -> -1
        # W: C12 -> 1, C22 -> 1
        u3 = {u_idx(i1, j1): 1}
        v3 = {v_idx(j2, k1): 1, v_idx(j1, k1): -1}
        w3 = {w_idx(k2, i1): 1, w_idx(k2, i2): 1}
        add_term(u3, v3, w3)

        # Term 4: M4 = A22(B11-B22)
        # U: A22 -> 1
        # V: B11 -> 1, B22 -> -1
        # W: C11 -> 1, C21 -> 1
        u4 = {u_idx(i2, j2): 1}
        v4 = {v_idx(j1, k1): 1, v_idx(j2, k2): -1}
        w4 = {w_idx(k1, i1): 1, w_idx(k1, i2): 1}
        add_term(u4, v4, w4)

        # Term 5: M5 = (A11+A12)B22
        # U: A11, A12 -> 1
        # V: B22 -> 1
        # W: C11 -> -1, C12 -> 1
        u5 = {u_idx(i1, j1): 1, u_idx(i1, j2): 1}
        v5 = {v_idx(j2, k2): 1}
        w5 = {w_idx(k1, i1): -1, w_idx(k2, i1): 1}
        add_term(u5, v5, w5)

        # Term 6: M6 = (A21-A11)(B11+B21)
        # U: A21 -> 1, A11 -> -1
        # V: B11, B21 -> 1
        # W: C22 -> 1
        u6 = {u_idx(i2, j1): 1, u_idx(i1, j1): -1}
        v6 = {v_idx(j1, k1): 1, v_idx(j2, k1): 1}
        w6 = {w_idx(k2, i2): 1}
        add_term(u6, v6, w6)

        # Term 7: M7 = (A12-A22)(B21+B22)
        # U: A12 -> 1, A22 -> -1
        # V: B21, B22 -> 1
        # W: C11 -> 1
        u7 = {u_idx(i1, j2): 1, u_idx(i2, j2): -1}
        v7 = {v_idx(j2, k1): 1, v_idx(j2, k2): 1}
        w7 = {w_idx(k1, i1): 1}
        add_term(u7, v7, w7)

    def add_schoolbook_term(i, j, k):
        """
        Adds a single schoolbook term for the specific (i,j,k) entry.
        """
        u_sb = {i * m + j: 1}
        v_sb = {j * p + k: 1}
        w_sb = {k * n + i: 1}
        add_term(u_sb, v_sb, w_sb)

    # 1. Apply Strassen blocks to the 2x4x4 sub-tensor (i in {0,1}, j in {0..3}, k in {0..3})
    # Split j into {0,1}, {2,3}
    # Split k into {0,1}, {2,3}
    
    # Block (i:0-1, j:0-1, k:0-1)
    add_strassen_block(0, 0, 0)
    # Block (i:0-1, j:2-3, k:0-1)
    add_strassen_block(0, 2, 0)
    # Block (i:0-1, j:0-1, k:2-3)
    add_strassen_block(0, 0, 2)
    # Block (i:0-1, j:2-3, k:2-3)
    add_strassen_block(0, 2, 2)
    
    covered = set()
    for i in [0, 1]:
        for j in range(4):
            for k in range(4):
                covered.add((i, j, k))

    # 2. Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook_term(i, j, k)
                    # This covers:
                    # i=2, all j, all k (20 terms)
                    # i in {0,1}, k=4, all j (8 terms)
                    # Total 28 terms.
    
    # Stack columns
    if not U_cols:
        # Fallback if something went wrong (should not happen)
        U = np.zeros((dim_u, 1))
        V = np.zeros((dim_v, 1))
        W = np.zeros((dim_w, 1))
    else:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
    rank = U.shape[1]
    
    # Verify exactness
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Must be 1 for all valid i,j,k
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Vectorized dot product
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if val != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback to schoolbook if verification fails
        U_sb = np.zeros((dim_u, n*m*p), dtype=int)
        V_sb = np.zeros((dim_v, n*m*p), dtype=int)
        W_sb = np.zeros((dim_w, n*m*p), dtype=int)
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
# EVOLVE-BLOCK-END
