"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 50
    rank = 50
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term for a specific 2x2x2 block
    # i_offset, j_offset, k_offset
    # a, b, c are 2x2 matrices (flattened or not)
    # We define the 7 Strassen terms for 2x2x2
    # T1: (A00+A11)(B00+B11) -> C00 += T1, C11 += T1
    # T2: (A10+A11)B00 -> C10 += T2
    # T3: A00(B01-B11) -> C01 += T3
    # T4: A11(B10-B00) -> C10 += T4
    # T5: (A00+A01)B11 -> C01 += T5, C11 += T5? No, C11 += T5? 
    # Let's use standard Strassen coefficients.
    
    # Strassen Terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # We need to map these to U, V, W contributions.
    # For a term Mx, U has the A coeffs, V has the B coeffs, W has the C coeffs.
    
    # Term 1: M1
    # A: A00=1, A11=1
    # B: B00=1, B11=1
    # C: C00=1, C11=1
    # Term 2: M2
    # A: A10=1, A11=1
    # B: B00=1
    # C: C10=1
    # Term 3: M3
    # A: A00=1
    # B: B01=1, B11=-1
    # C: C01=1
    # Term 4: M4
    # A: A11=1
    # B: B10=1, B00=-1
    # C: C10=1
    # Term 5: M5
    # A: A00=1, A01=1
    # B: B11=1
    # C: C01=1, C11=1? 
    # Wait, C11 = M1 - M3 + M7. M5 is not in C11?
    # Standard Strassen:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    # So M5 only contributes to C01.
    
    # Let's verify M5: (A00+A01)B11.
    # A00 B11 + A01 B11.
    # In C01: A00 B01 + A01 B11.
    # M3 = A00(B01-B11) = A00 B01 - A00 B11.
    # M3 + M5 = A00 B01 - A00 B11 + A00 B11 + A01 B11 = A00 B01 + A01 B11. Correct.
    
    # Term 6: M6
    # A: A10=1, A00=-1
    # B: B00=1, B01=1
    # C: C00=1
    # Term 7: M7
    # A: A01=1, A11=-1
    # B: B10=1, B11=1
    # C: C11=1

    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [A00, A01, A10, A11]
        # V: [B00, B01, B10, B11]
        # W: [C00, C01, C10, C11] (Note: W index is k*n+i. 
        # k=0,i=0 -> C00. k=0,i=1 -> C10? No.
        # W index: k*n + i.
        # n=2.
        # k=0, i=0 -> 0. (C00)
        # k=0, i=1 -> 1. (C10)
        # k=1, i=0 -> 2. (C01)
        # k=1, i=1 -> 3. (C11)
        # So W order: [C00, C10, C01, C11]
        
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1: C00, C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2: C10
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]), # M3: C01
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 0]), # M4: C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]), # M5: C01
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]), # M6: C00
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]), # M7: C11
    ]
    
    def add_2x2x2_block(i_off, j_off, k_off, rank_offset):
        nonlocal col
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # U indices: (i_off + i) * m + (j_off + j)
            # V indices: (j_off + j) * p + (k_off + k)
            # W indices: (k_off + k) * n + (i_off + i)
            
            for i in range(2):
                for j in range(2):
                    u_val = u_coeffs[i*2 + j]
                    if u_val != 0:
                        U[(i_off + i) * m + (j_off + j)][col] = u_val
            
            for j in range(2):
                for k in range(2):
                    v_val = v_coeffs[j*2 + k]
                    if v_val != 0:
                        V[(j_off + j) * p + (k_off + k)][col] = v_val
            
            for k in range(2):
                for i in range(2):
                    w_val = w_coeffs[k*2 + i]
                    if w_val != 0:
                        W[(k_off + k) * n + (i_off + i)][col] = w_val
            
            col += 1
            if col >= rank:
                return

    def add_2x2x1_block(i_off, j_off, k_off, rank_offset):
        nonlocal col
        # Schoolbook for 2x2x1
        # A00 B00 -> C00
        # A01 B00 -> C01? No, B is 2x1. B00, B10.
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 1: A00 B00 -> C00
        U[(i_off + 0) * m + (j_off + 0)][col] = 1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        W[(k_off + 0) * n + (i_off + 0)][col] = 1
        col += 1
        
        # Term 2: A01 B10 -> C01? No, C index is i.
        # C_{i, k}. k is fixed (k_off).
        # C_{0, k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k}
        # C_{1, k} = A_{1,0} B_{0,k} + A_{1,1} B_{1,k}
        
        # Term 2: A01 B10 -> C00? No.
        # A_{0,1} B_{1,k} contributes to C_{0,k}.
        U[(i_off + 0) * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 0)][col] = 1
        W[(k_off + 0) * n + (i_off + 0)][col] = 1
        col += 1
        
        # Term 3: A10 B00 -> C10
        U[(i_off + 1) * m + (j_off + 0)][col] = 1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        W[(k_off + 0) * n + (i_off + 1)][col] = 1
        col += 1
        
        # Term 4: A11 B10 -> C10
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 0)][col] = 1
        W[(k_off + 0) * n + (i_off + 1)][col] = 1
        col += 1

    # Block 1: i in [0,1], j in [0,1]
    # K split: 0,1 (2x2x2), 2,3 (2x2x2), 4,5 (2x2x2), 6 (2x2x1)
    add_2x2x2_block(0, 0, 0, 0)
    add_2x2x2_block(0, 0, 2, 0)
    add_2x2x2_block(0, 0, 4, 0)
    add_2x2x1_block(0, 0, 6, 0)
    
    # Block 2: i in [0,1], j in [2,3]
    add_2x2x2_block(0, 2, 0, 0)
    add_2x2x2_block(0, 2, 2, 0)
    add_2x2x2_block(0, 2, 4, 0)
    add_2x2x1_block(0, 2, 6, 0)
    
    # Verification
    # Generate random matrices
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C_ik = sum_r W_ki_r * (sum_j U_ij_r A_ij) * (sum_k V_jk_r B_jk)
    # Wait, the formula is:
    # C_ik = sum_r W_{ki,r} * (sum_{j} U_{ij,r} A_{ij}) * (sum_{k'} V_{j k', r} B_{j k'})
    # This is not correct because the sums are over different indices for U and V.
    
    # Correct check:
    # Tensor T_ijk = sum_r U_ij_r V_jk_r W_ki_r
    # We want T_ijk = 1 for all i,j,k?
    # No, we want sum_j T_ijk A_ij B_jk = C_ik.
    # So T_ijk should be 1.
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j][r]
                if u_val == 0: continue
                for k in range(p):
                    v_val = V[j*p + k][r]
                    if v_val == 0: continue
                    w_val = W[k*n + i][r]
                    if w_val == 0: continue
                    T[i, j, k] += u_val * v_val * w_val
                    
    # Check if T is all ones
    if not np.allclose(T, 1.0):
        # Fallback to schoolbook if verification fails
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

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
