"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_222_rows():
    # Standard Strassen for 2x2 * 2x2 -> 2x2, rank 7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M4
    # C10 = M1 + M2 - M3 - M7
    # C11 = M2 - M1 + M5 + M7
    # Each row: (u_coeffs, v_coeffs, w_coeffs)
    # u: (A00, A01, A10, A11)
    # v: (B00, B01, B10, B11)
    # w: (C00, C01, C10, C11)
    rows = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 0)),   # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),   # M2
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 0)),  # M3
        ((0, 0, 0, 1), (1, -1, 0, 0), (0, 1, 0, 0)),  # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (1, 0, 0, 0)),   # M5
        ((1, 0, -1, 0), (1, 0, 1, 0), (1, 0, 0, 0)),  # M6
        ((0, 1, 0, -1), (0, 1, 0, 1), (0, 0, -1, 1)), # M7
    ]
    return rows

def _schoolbook_122_rows():
    # 1x2 * 2x2 -> 1x2, rank 4
    # u: (a0, a1), v: (b00, b01, b10, b11), w: (c0, c1)
    rows = [
        ((1, 0), (1, 0, 0, 0), (1, 0)),  # a0*b00 -> c0
        ((1, 0), (0, 1, 0, 0), (0, 1)),  # a0*b01 -> c1
        ((0, 1), (0, 0, 1, 0), (1, 0)),  # a1*b10 -> c0
        ((0, 1), (0, 0, 0, 1), (0, 1)),  # a1*b11 -> c1
    ]
    return rows

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Partition:
    # A rows (i): 2 + 1 (i=0,1 and i=2)
    # B rows (j): 2 + 2 (j=0,1 and j=2,3)
    # C cols (k): 2 + 2 + 2 (k=0,1 and k=2,3 and k=4,5)
    
    # Block sizes:
    # A partition: [2, 1]
    # B partition: [2, 2]
    # C partition: [2, 2, 2]
    
    a_parts = [2, 1]
    b_parts = [2, 2]
    c_parts = [2, 2, 2]
    
    # Build global U, V, W
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (18, rank)
    
    # We'll accumulate rows for U, V, W
    U_rows = []  # list of lists, each of length 12
    V_rows = []  # list of lists, each of length 24
    W_rows = []  # list of lists, each of length 18
    
    def add_block(i_start, j_start, k_start, size_a, size_b, size_c, block_type):
        """
        Add a block decomposition for A[i_start:i_start+size_a, :] * B[:, j_start:j_start+size_b]
        contributing to C[:, k_start:k_start+size_c].
        Actually, more precisely:
        The full product C = A * B where A is n x m, B is m x p, C is n x p.
        We partition:
        A rows into segments (i indices)
        B cols into segments (j indices) -- wait, B is m x p, so rows are j, cols are k
        C rows are i (same as A), C cols are k (same as B)
        
        Actually let me reconsider the indexing.
        A is n x m: A[i, j] where i in [0,n), j in [0,m)
        B is m x p: B[j, k] where j in [0,m), k in [0,p)
        C is n x p: C[i, k] where i in [0,n), k in [0,p)
        
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        We partition:
        i indices (rows of A, rows of C): a_parts
        j indices (cols of A, rows of B): b_parts
        k indices (cols of B, cols of C): c_parts
        
        For each combination of (a_seg, b_seg, c_seg), we have a block:
        A_block: size_a x size_b
        B_block: size_b x size_c
        C_block: size_a x size_c
        
        But wait, the j partition is shared between A's columns and B's rows. So for each (a_seg, b_seg, c_seg):
        - A segment: rows i in a_seg, cols j in b_seg
        - B segment: rows j in b_seg, cols k in c_seg
        - C segment: rows i in a_seg, cols k in c_seg
        
        So we iterate over a_seg_idx, b_seg_idx, c_seg_idx.
        """
        nonlocal U_rows, V_rows, W_rows
        
        if size_a == 2 and size_b == 2 and size_c == 2:
            block_rows = _strassen_222_rows()
        elif size_a == 1 and size_b == 2 and size_c == 2:
            block_rows = _schoolbook_122_rows()
        else:
            # Fallback to schoolbook for other sizes
            block_rows = []
            for ia in range(size_a):
                for jb in range(size_b):
                    for kc in range(size_c):
                        u = [0]*size_a
                        v = [0]*size_b
                        w = [0]*size_c
                        u[ia] = 1
                        v[jb] = 1
                        w[kc] = 1
                        block_rows.append(((u[0] if size_a>0 else 0, u[1] if size_a>1 else 0),
                                           (v[0], v[1], v[2] if size_b>2 else 0, v[3] if size_b>3 else 0),
                                           (w[0], w[1], w[2] if size_c>2 else 0)))
            # This fallback is complex; let's handle it differently
            return
        
        # Compute global index offsets
        # i_start: global row index in A/C
        # j_start: global col index in A / row index in B
        # k_start: global col index in B/C
        
        for (u_coeffs, v_coeffs, w_coeffs) in block_rows:
            # Build full U row (length n*m = 12)
            U_row = [0] * (n * m)
            V_row = [0] * (m * p)
            W_row = [0] * (p * n)
            
            # Fill U_row: for each local (ia, jb), U[i_start+ia, j_start+jb] = u_coeffs[ia*size_b + jb]
            # u_coeffs is indexed as [ia * size_b + jb]
            for ia in range(size_a):
                for jb in range(size_b):
                    val = u_coeffs[ia * size_b + jb] if (ia * size_b + jb) < len(u_coeffs) else 0
                    gi = i_start + ia
                    gj = j_start + jb
                    U_row[gi * m + gj] = val
            
            # Fill V_row: for each local (jb, kc), V[j_start+jb, k_start+kc] = v_coeffs[jb*size_c + kc]
            for jb in range(size_b):
                for kc in range(size_c):
                    val = v_coeffs[jb * size_c + kc] if (jb * size_c + kc) < len(v_coeffs) else 0
                    gj = j_start + jb
                    gk = k_start + kc
                    V_row[gj * p + gk] = val
            
            # Fill W_row: for each local (ia, kc), W[k_start+kc, i_start+ia] = w_coeffs[ia*size_c + kc]
            # Note: W index is k*n + i, so W[gk * n + gi]
            for ia in range(size_a):
                for kc in range(size_c):
                    val = w_coeffs[ia * size_c + kc] if (ia * size_c + kc) < len(w_coeffs) else 0
                    gi = i_start + ia
                    gk = k_start + kc
                    W_row[gk * n + gi] = val
            
            U_rows.append(U_row)
            V_rows.append(V_row)
            W_rows.append(W_row)
    
    # Now iterate over all block combinations
    # a_seg_idx, b_seg_idx, c_seg_idx
    a_starts = []
    s = 0
    for size in a_parts:
        a_starts.append((s, size))
        s += size
    
    b_starts = []
    s = 0
    for size in b_parts:
        b_starts.append((s, size))
        s += size
    
    c_starts = []
    s = 0
    for size in c_parts:
        c_starts.append((s, size))
        s += size
    
    for i_start, size_a in a_starts:
        for j_start, size_b in b_starts:
            for k_start, size_c in c_starts:
                add_block(i_start, j_start, k_start, size_a, size_b, size_c, "auto")
    
    rank = len(U_rows)
    
    # Convert to list of lists format
    U = U_rows
    V = V_rows
    W = W_rows
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
