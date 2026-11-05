import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block sizes: 2, 2, 1
    row_sizes = [2, 2, 1]
    mid_sizes = [2, 2, 1]
    col_sizes = [2, 2, 1]
    
    # Start indices
    row_starts = [0, 2, 4]
    mid_starts = [0, 2, 4]
    col_starts = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block_factors(i, j, k):
        """Add schoolbook factors for block (i,j,k) to U_cols, V_cols, W_cols."""
        rs = row_sizes[i]
        ms = mid_sizes[j]
        cs = col_sizes[k]
        
        r0 = row_starts[i]
        m0 = mid_starts[j]
        c0 = col_starts[k]
        
        for a_row in range(rs):
            for b_col in range(cs):
                for mid in range(ms):
                    # U: A[r0+a_row, m0+mid]
                    u = np.zeros(n*m)
                    u[(r0 + a_row)*m + (m0 + mid)] = 1
                    
                    # V: B[m0+mid, c0+b_col]
                    v = np.zeros(m*p)
                    v[(m0 + mid)*p + (c0 + b_col)] = 1
                    
                    # W: C[r0+a_row, c0+b_col] -> index (r0+a_row)*n + (c0+b_col)
                    w = np.zeros(p*n)
                    w[(r0 + a_row)*n + (c0 + b_col)] = 1
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
    
    def add_strassen_factors(i, j, k):
        """Add Strassen factors for 2x2x2 block (i,j,k)."""
        r0 = row_starts[i]
        m0 = mid_starts[j]
        c0 = col_starts[k]
        
        # Strassen's 7 products
        strassen_terms = [
            # (U_A_coeffs for [A00, A01, A10, A11], V_B_coeffs for [B00, B01, B10, B11], W_C_coeffs for [C00, C01, C10, C11])
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
        ]
        
        for u_a, v_b, w_c in strassen_terms:
            # Build U vector
            u = np.zeros(n*m)
            for a_idx, coeff in enumerate(u_a):
                if coeff != 0:
                    a_row = a_idx // 2
                    a_col = a_idx % 2
                    u[(r0 + a_row)*m + (m0 + a_col)] = coeff
            
            # Build V vector
            v = np.zeros(m*p)
            for b_idx, coeff in enumerate(v_b):
                if coeff != 0:
                    b_row = b_idx // 2
                    b_col = b_idx % 2
                    v[(m0 + b_row)*p + (c0 + b_col)] = coeff
            
            # Build W vector
            w = np.zeros(p*n)
            for c_idx, coeff in enumerate(w_c):
                if coeff != 0:
                    c_row = c_idx // 2
                    c_col = c_idx % 2
                    w[(r0 + c_row)*n + (c0 + c_col)] = coeff
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    # Process all block triplets
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if row_sizes[i] == 2 and mid_sizes[j] == 2 and col_sizes[k] == 2:
                    add_strassen_factors(i, j, k)
                else:
                    add_block_factors(i, j, k)
    
    # Convert to matrices
    U = np.column_stack(U_cols) if U_cols else np.zeros((25, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((25, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((25, 0))
    
    rank = U.shape[1]
    
    # Verify
    def compute_error(U, V, W):
        err = 0.0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    val = np.sum(U[i*5+j, :] * V[j*5+k, :] * W[k*5+i, :])
                    err += (val - 1.0)**2
        return err
    
    error = compute_error(U, V, W)
    if error > 1e-9:
        # Fallback to schoolbook
        U_sb = np.zeros((25, 125))
        V_sb = np.zeros((25, 125))
        W_sb = np.zeros((25, 125))
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U_sb[i*5+j, col] = 1
                    V_sb[j*5+k, col] = 1
                    W_sb[k*5+i, col] = 1
                    col += 1
        return {"U": [[float(x) for x in row] for row in U_sb],
                "V": [[float(x) for x in row] for row in V_sb],
                "W": [[float(x) for x in row] for row in W_sb]}
    
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
# EVOLVE-BLOCK-END
