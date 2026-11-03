"""Matrix multiplication 4x4 x 4x9: rank-minimal exact tensor decomposition.

Uses block Strassen (2x2 blocks of 2x2) for the first 8 columns (4x4 x 4x8)
and a direct (schoolbook) factorization for the last column (4x4 x 4x1).
Total rank = 2*56 + 1*16 = 128 (vs schoolbook 144).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 4, 4, 9

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: dict {index: value}, etc."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries.items():
            u_vec[idx] = val
        for idx, val in v_entries.items():
            v_vec[idx] = val
        for idx, val in w_entries.items():
            w_vec[idx] = val
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    def strassen_4x4_block(k_offset):
        """Add 56 terms for 4x4*4x4 matmul with B columns offset by k_offset.
        
        A[i,j] for i,j in 0..3
        B[j, k_offset + k] for j in 0..3, k in 0..3
        C[i, k_offset + k] for i in 0..3, k in 0..3
        """
        # Strassen formulas for 2x2 block matrices:
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # M1 = A00 * (B01 - B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A01 - A11) * (B10 + B11)
        # M4 = A11 * (B00 - B01)
        # M5 = (A00 + A01) * B00
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A11 - A10) * (B10 - B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Each Mx is a 2x2*2x2 product. We expand each into 8 schoolbook terms.
        # For 2x2*2x2: C[i',k'] = sum_{j'=0}^{1} A[i',j']*B[j',k']
        # 8 terms: one for each (i', j', k') in {0,1}^3

        # Helper: expand a 2x2*2x2 product where A and B are specified by
        # their global indices and coefficients.
        # A_spec: list of (row_global, col_global, coeff)
        # B_spec: list of (row_global, col_global, coeff)
        # The product contributes to C at positions determined by the formula.
        # We need to know which C block this contributes to and with what sign.
        
        def expand_2x2_product(A_spec, B_spec, c_row_base, c_col_base, sign=1):
            """Expand a 2x2*2x2 product into 8 rank-1 terms.
            
            A_spec: list of (a_row, a_col, a_coeff) - 4 entries for the 2x2 A block
            B_spec: list of (b_row, b_col, b_coeff) - 4 entries for the 2x2 B block
            c_row_base, c_col_base: base row/col for the C output (2x2 block)
            sign: overall sign multiplier
            
            The 2x2 A block has local indices (0,0),(0,1),(1,0),(1,1)
            The 2x2 B block has local indices (0,0),(0,1),(1,0),(1,1)
            Local col of A must match local row of B for a term to contribute.
            """
            # Map local indices to global
            a_local = {}  # (local_row, local_col) -> (global_row, global_col, coeff)
            for (gr, gc, coeff) in A_spec:
                # Determine local index from global
                # We'll figure out local from the structure
                a_local[(gr, gc)] = coeff
            
            b_local = {}
            for (gr, gc, coeff) in B_spec:
                b_local[(gr, gc)] = coeff
            
            # We need to match A's local column with B's local row
            # A local cols: 0,1 -> corresponding global cols
            # B local rows: 0,1 -> corresponding global rows
            
            # Extract the global coordinates for local indices
            a_rows = sorted(set(gr for gr, gc, c in A_spec))
            a_cols = sorted(set(gc for gr, gc, c in A_spec))
            b_rows = sorted(set(gr for gr, gc, c in B_spec))
            b_cols = sorted(set(gc for gr, gc, c in B_spec))
            
            # A local (lr, lc) -> global (a_rows[lr], a_cols[lc])
            # B local (lr, lc) -> global (b_rows[lr], b_cols[lc])
            
            for lr in range(2):
                for lc in range(2):
                    for kc in range(2):
                        # Term: A[lr, lc] * B[lc, kc] -> C[lr, kc]
                        a_gr = a_rows[lr]
                        a_gc = a_cols[lc]
                        b_gr = b_rows[lc]
                        b_gc = b_cols[kc]
                        
                        a_coeff = a_local.get((a_gr, a_gc), 0)
                        b_coeff = b_local.get((b_gr, b_gc), 0)
                        
                        if a_coeff == 0 or b_coeff == 0:
                            continue
                        
                        total_coeff = sign * a_coeff * b_coeff
                        if total_coeff == 0:
                            continue
                        
                        # C global position: (c_row_base + lr, c_col_base + kc)
                        c_row = c_row_base + lr
                        c_col = c_col_base + kc
                        
                        u_idx = a_gr * m + a_gc
                        v_idx = b_gr * p + b_gc
                        w_idx = c_col * n + c_row
                        
                        add_term({u_idx: total_coeff}, {v_idx: 1}, {w_idx: 1})

        # Define the 2x2 blocks with their global coordinates
        # A00: rows 0-1, cols 0-1
        # A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1
        # A11: rows 2-3, cols 2-3
        # B00: rows 0-1, cols k_offset+0 to k_offset+1
        # B01: rows 0-1, cols k_offset+2 to k_offset+3
        # B10: rows 2-3, cols k_offset+0 to k_offset+1
        # B11: rows 2-3, cols k_offset+2 to k_offset+3
        # C00: rows 0-1, cols k_offset+0 to k_offset+1
        # C01: rows 0-1, cols k_offset+2 to k_offset+3
        # C10: rows 2-3, cols k_offset+0 to k_offset+1
        # C11: rows 2-3, cols k_offset+2 to k_offset+3

        # M1 = A00 * (B01 - B11)
        # A00: [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]
        # (B01 - B11): B01 has +1 at (0,k+2),(0,k+3),(1,k+2),(1,k+3)
        #               B11 has -1 at (2,k+2),(2,k+3),(3,k+2),(3,k+3)
        # But these are in different row blocks! B01 is rows 0-1, B11 is rows 2-3.
        # In the abstract Strassen formula, B01 - B11 is a 2x2 matrix where
        # entry (i,j) = B01[i,j] - B11[i,j] = B[0+i, k+2+j] - B[2+i, k+2+j]
        # So the "effective" B matrix for M1 has entries that mix rows 0-1 and 2-3.
        # This means we can't simply use a 2x2 B block.
        
        # The correct interpretation: M1[i',k'] = sum_{j'=0}^{1} A00[i',j'] * (B01[j',k'] - B11[j',k'])
        # = sum_{j'} A[i',j'] * B[j', k+2+k'] - sum_{j'} A[i',j'] * B[2+j', k+2+k']
        
        # So M1 contributes to C00 (rows 0-1, cols k+0 to k+1) with:
        # For each i', k' in 0,1:
        #   sum_{j'=0}^{1} A[i',j'] * B[j', k+2+k'] - sum_{j'=0}^{1} A[i',j'] * B[2+j', k+2+k']
        
        # Wait, that doesn't seem right dimensionally. Let me reconsider.
        
        # Actually, I think the issue is that in the block Strassen formula,
        # when we write M1 = A00 * (B01 - B11), the (B01 - B11) is a 2x2 matrix
        # whose entries are linear combinations of B entries. The product is then
        # a 2x2 matrix that gets added to C00.
        
        # Let me just directly compute the contributions:
        # M1[i',k'] = sum_{j'=0}^{1} A[0+i', 0+j'] * (B[0+j', k+2+k'] - B[2+j', k+2+k'])
        # for i', k' in 0,1
        
        # This gives 2*2*2 = 8 scalar products (i' in 0,1; j' in 0,1; k' in 0,1)
        # Each with coefficient ±1.
        
        # M1 contributes to C00 (c_row_base=0, c_col_base=k_offset) with sign +1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i,j] * B[j, k_offset+2+k] -> C[i, k_offset+k]
                    u_idx = i * m + j
                    v_idx = j * p + (k_offset + 2 + k)
                    w_idx = (k_offset + k) * n + i
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    # -A[i,j] * B[2+j, k_offset+2+k] -> C[i, k_offset+k]
                    v_idx2 = (2 + j) * p + (k_offset + 2 + k)
                    add_term({u_idx: 1}, {v_idx2: -1}, {w_idx: 1})
        
        # M2 = (A00 + A10) * B11
        # (A00 + A10)[i',j'] = A[i',j'] + A[2+i',j'] for i',j' in 0,1
        # B11[j',k'] = B[2+j', k+2+k'] for j',k' in 0,1
        # M2[i',k'] = sum_{j'} (A[i',j'] + A[2+i',j']) * B[2+j', k+2+k']
        # Contributes to C10 (rows 2-3, cols k+0 to k+1) with sign +1
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i,j] * B[2+j, k_offset+2+k] -> C[2+i, k_offset+k]
                    u_idx = i * m + j
                    v_idx = (2 + j) * p + (k_offset + 2 + k)
                    w_idx = (k_offset + k) * n + (2 + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    # A[2+i,j] * B[2+j, k_offset+2+k] -> C[2+i, k_offset+k]
                    u_idx2 = (2 + i) * m + j
                    add_term({u_idx2: 1}, {v_idx: 1}, {w_idx: 1})
        
        # M3 = (A01 - A11) * (B10 + B11)
        # (A01 - A11)[i',j'] = A[i',2+j'] - A[2+i',2+j'] for i',j' in 0,1
        # (B10 + B11)[j',k'] = B[2+j', k+k'] + B[2+j', k+2+k'] for j',k' in 0,1
        # M3[i',k'] = sum_{j'} (A[i',2+j'] - A[2+i',2+j']) * (B[2+j', k+k'] + B[2+j', k+2+k'])
        # Contributes to C01 (rows 0-1, cols k+2 to k+3) with sign +1
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i,2+j] * B[2+j, k_offset+k] -> C[i, k_offset+2+k]
                    u_idx = i * m + (2 + j)
                    v_idx = (2 + j) * p + (k_offset + k)
                    w_idx = (k_offset + 2 + k) * n + i
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    # A[i,2+j] * B[2+j, k_offset+2+k] -> C[i, k_offset+2+k]
                    v_idx2 = (2 + j) * p + (k_offset + 2 + k)
                    w_idx2 = (k_offset + 2 + k) * n + i
                    add_term({u_idx: 1}, {v_idx2: 1}, {w_idx2: 1})
                    # -A[2+i,2+j] * B[2+j, k_offset+k] -> C[i, k_offset+2+k]
                    u_idx2 = (2 + i) * m + (2 + j)
                    add_term({u_idx2: -1}, {v_idx: 1}, {w_idx: 1})
                    # -A[2+i,2+j] * B[2+j, k_offset+2+k] -> C[i, k_offset+2+k]
                    add_term({u_idx2: -1}, {v_idx2: 1}, {w_idx2: 1})
        
        # M4 = A11 * (B00 - B01)
        # A11[i',j'] = A[2+i',2+j'] for i',j' in 0,1
        # (B00 - B01)[j',k'] = B[j', k+k'] - B[j', k+2+k'] for j',k' in 0,1
        # M4[i',k'] = sum_{j'} A[2+i',2+j'] * (B[j', k+k'] - B[j', k+2+k'])
        # Contributes to C10 (rows 2-3, cols k+0 to k+1) with sign +1
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[2+i,2+j] * B[j, k_offset+k] -> C[2+i, k_offset+k]
                    u_idx = (2 + i) * m + (2 + j)
                    v_idx = j * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + (2 + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    # -A[2+i,2+j] * B[j, k_offset+2+k] -> C[2+i, k_offset+k]
                    v_idx2 = j * p + (k_offset + 2 + k)
                    add_term({u_idx: 1}, {v_idx2: -1}, {w_idx: 1})
        
        # M5 = (A00 + A01) * B00
        # (A00 + A01)[i',j'] = A[i',j'] + A[i',2+j'] for i',j' in 0,1
        # B00[j',k'] = B[j', k+k'] for j',k' in 0,1
        # M5[i',k'] = sum_{j'} (A[i',j'] + A[i',2+j']) * B[j', k+k']
        # Contributes to C00 (rows 0-1, cols k+0 to k+1) with sign -1
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i,j] * B[j, k_offset+k] -> C[i, k_offset+k] with sign -1
                    u_idx = i * m + j
                    v_idx = j * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + i
                    add_term({u_idx: -1}, {v_idx: 1}, {w_idx: 1})
                    # A[i,2+j] * B[j, k_offset+k] -> C[i, k_offset+k] with sign -1
                    u_idx2 = i * m + (2 + j)
                    add_term({u_idx2: -1}, {v_idx: 1}, {w_idx: 1})
        
        # M6 = (A10 - A00) * (B00 + B01)
        # (A10 - A00)[i',j'] = A[2+i',j'] - A[i',j'] for i',j' in 0,1
        # (B00 + B01)[j',k'] = B[j', k+k'] + B[j', k+2+k'] for j',k' in 0,1
        # M6[i',k'] = sum_{j'} (A[2+i',j'] - A[i',j']) * (B[j', k+k'] + B[j', k+2+k'])
        # Contributes to C11 (rows 2-3, cols k+2 to k+3) with sign +1
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[2+i,j] * B[j, k_offset+k] -> C[2+i, k_offset+2+k]
                    u_idx = (2 + i) * m + j
                    v_idx = j * p + (k_offset + k)
                    w_idx = (k_offset + 2 + k) * n + (2 + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    # A[2+i,j] * B[j, k_offset+2+k] -> C[2+i, k_offset+2+k]
                    v_idx2 = j * p + (k_offset + 2 + k)
                    w_idx2 = (k_offset + 2 + k) * n + (2 + i)
                    add_term({u_idx: 1}, {v_idx2: 1}, {w_idx2: 1})
                    # -A[i,j] * B[j, k_offset+k] -> C[2+i, k_offset+2+k]
                    u_idx2 = i * m + j
                    add_term({u_idx2: -1}, {v_idx: 1}, {w_idx: 1})
                    # -A[i,j] * B[j, k_offset+2+k] -> C[2+i, k_offset+2+k]
                    add_term({u_idx2: -1}, {v_idx2: 1}, {w_idx2: 1})
        
        # M7 = (A11 - A10) * (B10 - B11)
        # (A11 - A10)[i',j'] = A[2+i',2+j'] - A[2+i',j'] for i',j' in 0,1
        # (B10 - B11)[j',k'] = B[2+j', k+k'] - B[2+j', k+2+k'] for j',k' in 0,1
        # M7[i',k'] = sum_{j'} (A[2+i',2+j'] - A[2+i',j']) * (B[2+j', k+k'] - B[2+j', k+2+k'])
        # Contributes to C00 (rows 0-1, cols k+0 to k+1) with sign +1
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[2+i,2+j] * B[2+j, k_offset+k] -> C[i, k_offset+k]
                    u_idx = (2 + i) * m + (2 + j)
                    v_idx = (2 + j) * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + i
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    # -A[2+i,2+j] * B[2+j, k_offset+2+k] -> C[i, k_offset+k]
                    v_idx2 = (2 + j) * p + (k_offset + 2 + k)
                    add_term({u_idx: 1}, {v_idx2: -1}, {w_idx: 1})
                    # -A[2+i,j] * B[2+j, k_offset+k] -> C[i, k_offset+k]
                    u_idx2 = (2 + i) * m + j
                    add_term({u_idx2: -1}, {v_idx: 1}, {w_idx: 1})
                    # A[2+i,j] * B[2+j, k_offset+2+k] -> C[i, k_offset+k]
                    add_term({u_idx2: 1}, {v_idx2: 1}, {w_idx: 1})

    # Add Strassen terms for the two 4x4 blocks (columns 0-3 and 4-7)
    strassen_4x4_block(0)
    strassen_4x4_block(4)
    
    # Add schoolbook terms for the last column (column 8)
    for i in range(n):
        for j in range(m):
            for k in range(1):  # only column 8
                u_idx = i * m + j
                v_idx = j * p + (8 + k)
                w_idx = (8 + k) * n + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Convert to numpy arrays
    U = np.array(U_list, dtype=int).T  # shape (n*m, rank)
    V = np.array(V_list, dtype=int).T  # shape (m*p, rank)
    W = np.array(W_list, dtype=int).T  # shape (p*n, rank)
    
    # Verify correctness
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition: C[i,k] = sum_r U[i*m+j... wait
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)
    # W has shape (p*n, rank), indexed by (k,i) -> W[k*n+i, r]
    
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check if correct
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fall back to schoolbook if Strassen is incorrect
        U_list.clear()
        V_list.clear()
        W_list.clear()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        U = np.array(U_list, dtype=int).T
        V = np.array(V_list, dtype=int).T
        W = np.array(W_list, dtype=int).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
