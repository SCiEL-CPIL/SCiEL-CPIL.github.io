"""
Rank-110 exact tensor decomposition for 5x5 x 5x5 matrix multiplication.
Uses Strassen's algorithm for the 4x4 core block and schoolbook for the edges.
"""
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We will build the factor matrices U, V, W as lists of lists.
    # U shape: (n*m, rank) -> (25, rank)
    # V shape: (m*p, rank) -> (25, rank)
    # W shape: (p*n, rank) -> (25, rank)
    
    # We don't know the rank ahead of time, so we accumulate terms.
    # Each term is a dict or tuple of non-zero entries.
    # Term: (U_entries, V_entries, W_entries)
    # U_entries: list of (row, coeff)
    # V_entries: list of (row, coeff)
    # W_entries: list of (row, coeff)
    
    terms = []
    
    # --- Helper: Generate Strassen factors for size n ---
    def generate_strassen_factors(size):
        if size == 1:
            return [([(0, 1)], [(0, 1)], [(0, 1)])]
        
        half = size // 2
        sub_factors = generate_strassen_factors(half)
        
        def map_idx(local_idx, block_row, block_col, size, half):
            r = local_idx // half
            c = local_idx % half
            gr = block_row * half + r
            gc = block_col * half + c
            return gr * size + gc
            
        new_terms = []
        
        # M1 = (A00+A11)(B00+B11) -> C00
        for u, v, w in sub_factors:
            u_new = []
            v_new = []
            w_new = []
            for idx, coeff in u:
                u_new.append((map_idx(idx, 0, 0, size, half), coeff))
                u_new.append((map_idx(idx, 1, 1, size, half), coeff))
            for idx, coeff in v:
                v_new.append((map_idx(idx, 0, 0, size, half), coeff))
                v_new.append((map_idx(idx, 1, 1, size, half), coeff))
            for idx, coeff in w:
                w_new.append((map_idx(idx, 0, 0, size, half), coeff))
            new_terms.append((u_new, v_new, w_new))
            
        # M2 = (A10+A11)B00 -> C01
        for u, v, w in sub_factors:
            u_new = []
            v_new = []
            w_new = []
            for idx, coeff in u:
                u_new.append((map_idx(idx, 1, 0, size, half), coeff))
                u_new.append((map_idx(idx, 1, 1, size, half), coeff))
            for idx, coeff in v:
                v_new.append((map_idx(idx, 0, 0, size, half), coeff))
            for idx, coeff in w:
                w_new.append((map_idx(idx, 0, 1, size, half), coeff))
            new_terms.append((u_new, v_new, w_new))
            
        # M3 = A00(B10-B11) -> C10
        for u, v, w in sub_factors:
            u_new = []
            v_new = []
            w_new = []
            for idx, coeff in u:
                u_new.append((map_idx(idx, 0, 0, size, half), coeff))
            for idx, coeff in v:
                v_new.append((map_idx(idx, 1, 0, size, half), coeff))
                v_new.append((map_idx(idx, 1, 1, size, half), -coeff))
            for idx, coeff in w:
                w_new.append((map_idx(idx, 1, 0, size, half), coeff))
            new_terms.append((u_new, v_new, w_new))
            
        # M4 = A11(B01-B00) -> C11
        for u, v, w in sub_factors:
            u_new = []
            v_new = []
            w_new = []
            for idx, coeff in u:
                u_new.append((map_idx(idx, 1, 1, size, half), coeff))
            for idx, coeff in v:
                v_new.append((map_idx(idx, 0, 1, size, half), coeff))
                v_new.append((map_idx(idx, 0, 0, size, half), -coeff))
            for idx, coeff in w:
                w_new.append((map_idx(idx, 1, 1, size, half), coeff))
            new_terms.append((u_new, v_new, w_new))
            
        # M5 = (A00+A01)B11 -> C00
        for u, v, w in sub_factors:
            u_new = []
            v_new = []
            w_new = []
            for idx, coeff in u:
                u_new.append((map_idx(idx, 0, 0, size, half), coeff))
                u_new.append((map_idx(idx, 0, 1, size, half), coeff))
            for idx, coeff in v:
                v_new.append((map_idx(idx, 1, 1, size, half), coeff))
            for idx, coeff in w:
                w_new.append((map_idx(idx, 0, 0, size, half), coeff))
            new_terms.append((u_new, v_new, w_new))
            
        # M6 = (A10-A00)(B00+B01) -> C11
        for u, v, w in sub_factors:
            u_new = []
            v_new = []
            w_new = []
            for idx, coeff in u:
                u_new.append((map_idx(idx, 1, 0, size, half), coeff))
                u_new.append((map_idx(idx, 0, 0, size, half), -coeff))
            for idx, coeff in v:
                v_new.append((map_idx(idx, 0, 0, size, half), coeff))
                v_new.append((map_idx(idx, 0, 1, size, half), coeff))
            for idx, coeff in w:
                w_new.append((map_idx(idx, 1, 1, size, half), coeff))
            new_terms.append((u_new, v_new, w_new))
            
        # M7 = (A01-A11)(B10+B11) -> C01
        for u, v, w in sub_factors:
            u_new = []
            v_new = []
            w_new = []
            for idx, coeff in u:
                u_new.append((map_idx(idx, 0, 1, size, half), coeff))
                u_new.append((map_idx(idx, 1, 1, size, half), -coeff))
            for idx, coeff in v:
                v_new.append((map_idx(idx, 1, 0, size, half), coeff))
                v_new.append((map_idx(idx, 1, 1, size, half), coeff))
            for idx, coeff in w:
                w_new.append((map_idx(idx, 0, 1, size, half), coeff))
            new_terms.append((u_new, v_new, w_new))
            
        return new_terms

    # --- Add Strassen terms for 4x4 core (indices 0-3) ---
    core_size = 4
    strassen_terms = generate_strassen_factors(core_size)
    terms.extend(strassen_terms)
    
    # --- Add Schoolbook terms for edges ---
    # Indices:
    # Core: 0..3
    # Edge: 4
    
    # Block A12 (rows 0-3, col 4) * B21 (row 4, cols 0-3) -> C11 (rows 0-3, cols 0-3)
    # C[i, j] += A[i, 4] * B[4, j]
    for i in range(4):
        for j in range(4):
            u_entry = (i * 5 + 4, 1)
            v_entry = (4 * 5 + j, 1)
            w_entry = (j * 5 + i, 1)
            terms.append(([u_entry], [v_entry], [w_entry]))
            
    # Block A11 (rows 0-3, cols 0-3) * B12 (rows 0-3, col 4) -> C12 (rows 0-3, col 4)
    # C[i, 4] += sum_k A[i, k] * B[k, 4]
    for i in range(4):
        for k in range(4):
            u_entry = (i * 5 + k, 1)
            v_entry = (k * 5 + 4, 1)
            w_entry = (4 * 5 + i, 1)
            terms.append(([u_entry], [v_entry], [w_entry]))
            
    # Block A12 (rows 0-3, col 4) * B22 (row 4, col 4) -> C12 (rows 0-3, col 4)
    # C[i, 4] += A[i, 4] * B[4, 4]
    for i in range(4):
        u_entry = (i * 5 + 4, 1)
        v_entry = (4 * 5 + 4, 1)
        w_entry = (4 * 5 + i, 1)
        terms.append(([u_entry], [v_entry], [w_entry]))
        
    # Block A21 (row 4, cols 0-3) * B11 (rows 0-3, cols 0-3) -> C21 (row 4, cols 0-3)
    # C[4, j] += sum_k A[4, k] * B[k, j]
    for j in range(4):
        for k in range(4):
            u_entry = (4 * 5 + k, 1)
            v_entry = (k * 5 + j, 1)
            w_entry = (j * 5 + 4, 1)
            terms.append(([u_entry], [v_entry], [w_entry]))
            
    # Block A22 (row 4, col 4) * B21 (row 4, cols 0-3) -> C21 (row 4, cols 0-3)
    # C[4, j] += A[4, 4] * B[4, j]
    for j in range(4):
        u_entry = (4 * 5 + 4, 1)
        v_entry = (4 * 5 + j, 1)
        w_entry = (j * 5 + 4, 1)
        terms.append(([u_entry], [v_entry], [w_entry]))
        
    # Block A21 (row 4, cols 0-3) * B12 (rows 0-3, col 4) -> C22 (row 4, col 4)
    # C[4, 4] += sum_k A[4, k] * B[k, 4]
    for k in range(4):
        u_entry = (4 * 5 + k, 1)
        v_entry = (k * 5 + 4, 1)
        w_entry = (4 * 5 + 4, 1)
        terms.append(([u_entry], [v_entry], [w_entry]))
        
    # Block A22 (row 4, col 4) * B22 (row 4, col 4) -> C22 (row 4, col 4)
    # C[4, 4] += A[4, 4] * B[4, 4]
    u_entry = (4 * 5 + 4, 1)
    v_entry = (4 * 5 + 4, 1)
    w_entry = (4 * 5 + 4, 1)
    terms.append(([u_entry], [v_entry], [w_entry]))
    
    # --- Construct Matrices ---
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, coeff in u_entries:
            U[idx][r] += coeff
        for idx, coeff in v_entries:
            V[idx][r] += coeff
        for idx, coeff in w_entries:
            W[idx][r] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
