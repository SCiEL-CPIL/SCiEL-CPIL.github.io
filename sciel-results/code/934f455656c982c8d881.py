import numpy as np
import time

# EVOLVE-BLOCK-START
_n, _m, _p = 4, 4, 4
_rank = 49

def _build_decomposition():
    """
    Build the rank-49 Strassen-based decomposition for 4x4 matrix multiplication.
    This uses the recursive Strassen algorithm on 2x2 blocks, each of which
    is further decomposed into 2x2 Strassen.
    
    Total rank = 7 (top level Strassen terms) * 7 (bottom level Strassen terms per block) = 49.
    """
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    # Index helpers
    # A is 4x4. A[i, j]. U index = i * 4 + j
    # B is 4x4. B[j, k]. V index = j * 4 + k
    # C is 4x4. C[i, k]. W index = k * 4 + i (Trace convention W[k*n+i, r])
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(j, k): return j * 4 + k
    def w_idx(i, k): return k * 4 + i
    
    # Strassen decomposition for a 2x2 matrix multiplication C = A * B
    # where A, B, C are 2x2 matrices with entries that are themselves linear
    # combinations of the base matrix entries.
    
    # Let's define the 7 Strassen terms for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Results:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will iterate over the 7 top-level Strassen terms.
    # For each top-level term t, we have:
    # A_t = linear_comb(A_subblocks)
    # B_t = linear_comb(B_subblocks)
    # Then A_t * B_t is a 2x2 matrix product of two 2x2 matrices.
    # We decompose this 2x2 product into 7 bottom-level Strassen terms.
    
    # Top-level Strassen definitions for 4x4 (partitioned into 2x2 blocks A11, A12, A21, A22):
    # A11 is 2x2 (indices 0,0; 0,1; 1,0; 1,1)
    # A12 is 2x2 (indices 0,2; 0,3; 1,2; 1,3)
    # A21 is 2x2 (indices 2,0; 2,1; 3,0; 3,1)
    # A22 is 2x2 (indices 2,2; 2,3; 3,2; 3,3)
    
    # Similarly for B.
    
    # Let's define the linear combinations for each top-level term.
    # We represent a 2x2 matrix of linear combos as a list of 4 dicts,
    # where each dict maps (base_row, base_col) to coefficient.
    # base_row/base_col are 0..3.
    
    def add_dicts(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) + v
        if 0 in res.values():
            res = {k: v for k, v in res.items() if v != 0}
        return res
    
    def sub_dicts(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) - v
        if 0 in res.values():
            res = {k: v for k, v in res.items() if v != 0}
        return res
    
    def mul_dict(d1, c):
        return {k: v * c for k, v in d1.items()}
    
    # Base elements
    # A[r, c] -> {(r, c): 1}
    def A_elem(r, c): return {(r, c): 1}
    def B_elem(r, c): return {(r, c): 1}
    
    # Top-level 2x2 blocks of A (each is a 2x2 matrix of base elements)
    # A11:
    A11 = [[A_elem(0,0), A_elem(0,1)], [A_elem(1,0), A_elem(1,1)]]
    # A12:
    A12 = [[A_elem(0,2), A_elem(0,3)], [A_elem(1,2), A_elem(1,3)]]
    # A21:
    A21 = [[A_elem(2,0), A_elem(2,1)], [A_elem(3,0), A_elem(3,1)]]
    # A22:
    A22 = [[A_elem(2,2), A_elem(2,3)], [A_elem(3,2), A_elem(3,3)]]
    
    # B11:
    B11 = [[B_elem(0,0), B_elem(0,1)], [B_elem(1,0), B_elem(1,1)]]
    # B12:
    B12 = [[B_elem(0,2), B_elem(0,3)], [B_elem(1,2), B_elem(1,3)]]
    # B21:
    B21 = [[B_elem(2,0), B_elem(2,1)], [B_elem(3,0), B_elem(3,1)]]
    # B22:
    B22 = [[B_elem(2,2), B_elem(2,3)], [B_elem(3,2), B_elem(3,3)]]
    
    # Top-level Strassen terms (A_t, B_t)
    top_terms = []
    
    # T0: (A11 + A22) * (B11 + B22)
    A_t0 = [[add_dicts(A11[0][0], A22[0][0]), add_dicts(A11[0][1], A22[0][1])],
            [add_dicts(A11[1][0], A22[1][0]), add_dicts(A11[1][1], A22[1][1])]]
    B_t0 = [[add_dicts(B11[0][0], B22[0][0]), add_dicts(B11[0][1], B22[0][1])],
            [add_dicts(B11[1][0], B22[1][0]), add_dicts(B11[1][1], B22[1][1])]]
    top_terms.append((A_t0, B_t0))
    
    # T1: (A21 + A22) * B11
    A_t1 = [[add_dicts(A21[0][0], A22[0][0]), add_dicts(A21[0][1], A22[0][1])],
            [add_dicts(A21[1][0], A22[1][0]), add_dicts(A21[1][1], A22[1][1])]]
    B_t1 = B11
    top_terms.append((A_t1, B_t1))
    
    # T2: A11 * (B12 - B22)
    A_t2 = A11
    B_t2 = [[sub_dicts(B12[0][0], B22[0][0]), sub_dicts(B12[0][1], B22[0][1])],
            [sub_dicts(B12[1][0], B22[1][0]), sub_dicts(B12[1][1], B22[1][1])]]
    top_terms.append((A_t2, B_t2))
    
    # T3: A22 * (B21 - B11)
    A_t3 = A22
    B_t3 = [[sub_dicts(B21[0][0], B11[0][0]), sub_dicts(B21[0][1], B11[0][1])],
            [sub_dicts(B21[1][0], B11[1][0]), sub_dicts(B21[1][1], B11[1][1])]]
    top_terms.append((A_t3, B_t3))
    
    # T4: (A11 + A12) * B22
    A_t4 = [[add_dicts(A11[0][0], A12[0][0]), add_dicts(A11[0][1], A12[0][1])],
            [add_dicts(A11[1][0], A12[1][0]), add_dicts(A11[1][1], A12[1][1])]]
    B_t4 = B22
    top_terms.append((A_t4, B_t4))
    
    # T5: (A21 - A11) * (B11 + B12)
    A_t5 = [[sub_dicts(A21[0][0], A11[0][0]), sub_dicts(A21[0][1], A11[0][1])],
            [sub_dicts(A21[1][0], A11[1][0]), sub_dicts(A21[1][1], A11[1][1])]]
    B_t5 = [[add_dicts(B11[0][0], B12[0][0]), add_dicts(B11[0][1], B12[0][1])],
            [add_dicts(B11[1][0], B12[1][0]), add_dicts(B11[1][1], B12[1][1])]]
    top_terms.append((A_t5, B_t5))
    
    # T6: (A12 - A22) * (B21 + B22)
    A_t6 = [[sub_dicts(A12[0][0], A22[0][0]), sub_dicts(A12[0][1], A22[0][1])],
            [sub_dicts(A12[1][0], A22[1][0]), sub_dicts(A12[1][1], A22[1][1])]]
    B_t6 = [[add_dicts(B21[0][0], B22[0][0]), add_dicts(B21[0][1], B22[0][1])],
            [add_dicts(B21[1][0], B22[1][0]), add_dicts(B21[1][1], B22[1][1])]]
    top_terms.append((A_t6, B_t6))
    
    # Now, for each top term t, we have A_t (2x2 of dicts) and B_t (2x2 of dicts).
    # We apply the bottom-level Strassen decomposition to C_t = A_t * B_t.
    # C_t is a 2x2 matrix where each element is a bilinear form.
    # We decompose this into 7 scalar products of linear forms.
    
    # Bottom Strassen terms for A_t (elements a00, a01, a10, a11) and B_t (b00, b01, b10, b11)
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # Contributions to C_t elements:
    # C_t[0][0] = M1 + M4 - M5 + M7
    # C_t[0][1] = M3 + M5
    # C_t[1][0] = M2 + M4
    # C_t[1][1] = M1 - M2 + M3 + M6
    
    # We need to map the bottom Strassen products to the global U, V, W.
    
    for t_idx, (A_t, B_t) in enumerate(top_terms):
        a00, a01 = A_t[0][0], A_t[0][1]
        a10, a11 = A_t[1][0], A_t[1][1]
        b00, b01 = B_t[0][0], B_t[0][1]
        b10, b11 = B_t[1][0], B_t[1][1]
        
        # Define the 7 bottom Strassen linear forms
        # Each form is a dict of {base_index: coeff}
        
        def form_sum(d1, d2): return add_dicts(d1, d2)
        def form_diff(d1, d2): return sub_dicts(d1, d2)
        
        # X and Y forms for each M
        M_forms = []
        
        # M1
        x1 = form_sum(a00, a11)
        y1 = form_sum(b00, b11)
        M_forms.append((x1, y1))
        
        # M2
        x2 = form_sum(a10, a11)
        y2 = b00
        M_forms.append((x2, y2))
        
        # M3
        x3 = a00
        y3 = form_diff(b01, b11)
        M_forms.append((x3, y3))
        
        # M4
        x4 = a11
        y4 = form_diff(b10, b00)
        M_forms.append((x4, y4))
        
        # M5
        x5 = form_sum(a00, a01)
        y5 = b11
        M_forms.append((x5, y5))
        
        # M6
        x6 = form_diff(a10, a00)
        y6 = form_sum(b00, b01)
        M_forms.append((x6, y6))
        
        # M7
        x7 = form_diff(a01, a11)
        y7 = form_sum(b10, b11)
        M_forms.append((x7, y7))
        
        # Now fill U and V for these 7 terms
        # U index: row * 4 + col. Here the "row" and "col" are from the base 4x4 matrix.
        # The dict keys are (r, c).
        
        for m_idx, (x, y) in enumerate(M_forms):
            global_idx = t_idx * 7 + m_idx
            
            for (r, c), coeff in x.items():
                U[u_idx(r, c), global_idx] += coeff
            
            for (r, c), coeff in y.items():
                V[v_idx(r, c), global_idx] += coeff
    
    # Now fill W.
    # We need to determine which C element each top-level term contributes to.
    # C = C11 C12 / C21 C22 (2x2 blocks of 2x2 matrices)
    # C11 = T0 + T3 - T4 + T6
    # C12 = T2 + T4
    # C21 = T1 + T3
    # C22 = T0 - T1 + T2 + T5
    
    # For each top term T_t, it produces a 2x2 matrix C_t.
    # C_t[0][0], C_t[0][1], C_t[1][0], C_t[1][1] are linear combinations of the 7 bottom Strassen products.
    # Specifically:
    # C_t[0][0] = M1 + M4 - M5 + M7
    # C_t[0][1] = M3 + M5
    # C_t[1][0] = M2 + M4
    # C_t[1][1] = M1 - M2 + M3 + M6
    
    # The contribution of T_t to the final C is scaled and placed into the appropriate block.
    # Let's define the contribution of each bottom Strassen product M_k of top term T_t to the final C.
    
    # Block locations for C_t elements:
    # If T_t contributes to block (br, bc) of C with coefficient cf:
    #   C_t[0][0] (element at br, bc) goes to C[br, bc]
    #   C_t[0][1] (element at br, bc+1) goes to C[br, bc+1]
    #   C_t[1][0] (element at br+1, bc) goes to C[br+1, bc]
    #   C_t[1][1] (element at br+1, bc+1) goes to C[br+1, bc+1]
    
    # Let's list the contributions:
    # T0 (coeff 1) to C11 (block 0,0)
    # T3 (coeff 1) to C11 (block 0,0)
    # T4 (coeff -1) to C11 (block 0,0)
    # T6 (coeff 1) to C11 (block 0,0)
    # T2 (coeff 1) to C12 (block 0,1)
    # T4 (coeff 1) to C12 (block 0,1)
    # T1 (coeff 1) to C21 (block 1,0)
    # T3 (coeff 1) to C21 (block 1,0)
    # T0 (coeff 1) to C22 (block 1,1)
    # T1 (coeff -1) to C22 (block 1,1)
    # T2 (coeff 1) to C22 (block 1,1)
    # T5 (coeff 1) to C22 (block 1,1)
    
    contributions = [
        # T0
        [(0, 0, 1), (1, 1, 1)],
        # T1
        [(1, 0, 1), (1, 1, -1)],
        # T2
        [(0, 1, 1), (1, 1, 1)],
        # T3
        [(0, 0, 1), (1, 0, 1)],
        # T4
        [(0, 0, -1), (0, 1, 1)],
        # T5
        [(1, 1, 1)],
        # T6
        [(0, 0, 1)]
    ]
    
    for t_idx in range(7):
        block_contribs = contributions[t_idx]
        for (br, bc, cf) in block_contribs:
            # C_t[0][0] = M1 + M4 - M5 + M7
            # C_t[0][1] = M3 + M5
            # C_t[1][0] = M2 + M4
            # C_t[1][1] = M1 - M2 + M3 + M6
            
            # M1 (idx 0)
            # M2 (idx 1)
            # M3 (idx 2)
            # M4 (idx 3)
            # M5 (idx 4)
            # M6 (idx 5)
            # M7 (idx 6)
            
            m_to_c = {
                0: [(0, 0, 1), (1, 1, 1)],      # M1 -> C00 (+1), C11 (+1)
                1: [(1, 0, 1), (1, 1, -1)],     # M2 -> C10 (+1), C11 (-1)
                2: [(0, 1, 1), (1, 1, 1)],      # M3 -> C01 (+1), C11 (+1)
                3: [(0, 0, 1), (1, 0, 1)],      # M4 -> C00 (+1), C10 (+1)
                4: [(0, 0, -1), (0, 1, 1)],     # M5 -> C00 (-1), C01 (+1)
                5: [(1, 1, 1)],                 # M6 -> C11 (+1)
                6: [(0, 0, 1)]                  # M7 -> C00 (+1)
            }
            
            for m_idx in range(7):
                global_idx = t_idx * 7 + m_idx
                for (cr, cc, coeff_in_ct) in m_to_c[m_idx]:
                    # This M contributes to C_t[cr, cc] with coefficient coeff_in_ct
                    # C_t[cr, cc] goes to C[br+cr, bc+cc] with coefficient cf
                    final_r = br + cr
                    final_c = bc + cc
                    W[w_idx(final_r, final_c), global_idx] += cf * coeff_in_ct

    return U, V, W

# Build once at import time
_U, _V, _W = _build_decomposition()

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Returns precomputed factor matrices.
    """
    return {"U": _U.tolist(), "V": _V.tolist(), "W": _W.tolist()}
# EVOLVE-BLOCK-END
