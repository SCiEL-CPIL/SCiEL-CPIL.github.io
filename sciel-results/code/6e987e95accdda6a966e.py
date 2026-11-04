import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U rows: (i,j) -> 2i+j
    U22 = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    
    # V rows: (j,k) -> 2j+k
    V22 = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 0, 0, 1],  # j=0, k=1
        [0, 0, 0, 1, 0, 1, 0],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    
    # W rows: (k,i) -> 2k+i
    W22 = [
        [1, 0, 0, -1, 0, -1, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],   # k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],   # k=1, i=0
        [1, 0, -1, 0, 1, 0, -1]  # k=1, i=1
    ]
    
    U22 = np.array(U22, dtype=float)
    V22 = np.array(V22, dtype=float)
    W22 = np.array(W22, dtype=float)
    
    # Kronecker product for 4x4x4 part
    # Rank 49
    U44 = np.kron(U22, U22)
    V44 = np.kron(V22, V22)
    W44 = np.kron(W22, W22)
    
    # The Kronecker product np.kron(A, B) creates a matrix of size (A.rows*B.rows) x (A.cols*B.cols)
    # Row index r = r_A * B.rows + r_B
    # We need to check if the indexing matches the problem statement.
    # Problem: U rows are i*m + j. Here m=4. So row = 4*i + j.
    # In U22, row = 2*i + j.
    # In U44 = kron(U22, U22), row = (2*i1 + j1)*4 + (2*i0 + j0) = 8*i1 + 4*j1 + 2*i0 + j0.
    # We want row = 4*i + j = 4*(2*i1 + i0) + (2*j1 + j0) = 8*i1 + 4*i0 + 2*j1 + j0.
    # The order of terms is different: 4*j1 + 2*i0 vs 4*i0 + 2*j1.
    # So we need to permute the rows of U44.
    
    # Let's create a permutation map for rows of size 16.
    # Current row index in kron: r_kron = 8*i1 + 4*j1 + 2*i0 + j0
    # Desired row index: r_des = 8*i1 + 4*i0 + 2*j1 + j0
    
    perm_U = []
    for i in range(4):
        for j in range(4):
            i1 = i // 2
            i0 = i % 2
            j1 = j // 2
            j0 = j % 2
            r_des = i * 4 + j
            r_kron = (i1 * 2 + j1) * 4 + (i0 * 2 + j0)
            perm_U.append(r_kron)
            
    # Apply permutation to rows of U44
    U44_perm = U44[perm_U, :]
    
    # For V: rows are j*p + k. Here p=4 for the 4x4x4 part.
    # Row = 4*j + k.
    # In V22, row = 2*j + k.
    # In V44 = kron(V22, V22), row = (2*j1 + k1)*4 + (2*j0 + k0) = 8*j1 + 4*k1 + 2*j0 + k0.
    # Desired: 4*j + k = 4*(2*j1 + j0) + (2*k1 + k0) = 8*j1 + 4*j0 + 2*k1 + k0.
    # Mismatch: 4*k1 + 2*j0 vs 4*j0 + 2*k1.
    
    perm_V = []
    for j in range(4):
        for k in range(4):
            j1 = j // 2
            j0 = j % 2
            k1 = k // 2
            k0 = k % 2
            r_des = j * 4 + k
            r_kron = (j1 * 2 + k1) * 4 + (j0 * 2 + k0)
            perm_V.append(r_kron)
            
    V44_perm = V44[perm_V, :]
    
    # For W: rows are k*n + i. Here n=4.
    # Row = 4*k + i.
    # In W22, row = 2*k + i.
    # In W44 = kron(W22, W22), row = (2*k1 + i1)*4 + (2*k0 + i0) = 8*k1 + 4*i1 + 2*k0 + i0.
    # Desired: 4*k + i = 4*(2*k1 + k0) + (2*i1 + i0) = 8*k1 + 4*k0 + 2*i1 + i0.
    # Mismatch: 4*i1 + 2*k0 vs 4*k0 + 2*i1.
    
    perm_W = []
    for k in range(4):
        for i in range(4):
            k1 = k // 2
            k0 = k % 2
            i1 = i // 2
            i0 = i % 2
            r_des = k * 4 + i
            r_kron = (k1 * 2 + i1) * 4 + (k0 * 2 + i0)
            perm_W.append(r_kron)
            
    W44_perm = W44[perm_W, :]
    
    # Now handle the remaining 3 columns (k=4,5,6) with schoolbook
    # We need to add 48 columns.
    # Initialize full matrices
    rank_total = 49 + 48
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((28, rank_total)) # 4*7
    W_full = np.zeros((28, rank_total)) # 7*4
    
    # Copy Strassen part
    U_full[:, :49] = U44_perm
    V_full[:16, :49] = V44_perm # V rows for k<4 are 0..15
    W_full[:16, :49] = W44_perm # W rows for k<4 are 0..15
    
    # Add schoolbook part for k=4,5,6
    col_offset = 49
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                U_full[u_idx, col_offset] = 1
                V_full[v_idx, col_offset] = 1
                W_full[w_idx, col_offset] = 1
                col_offset += 1
                
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
