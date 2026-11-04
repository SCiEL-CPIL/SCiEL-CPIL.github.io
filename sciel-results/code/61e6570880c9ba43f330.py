import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44: 28 for 4x4x4 + 16 for 4x4x1
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 28) ---
    
    # Strassen formulas for 2x2 block matrices
    # A, B are 4x4. Split into 2x2 blocks A00, A01, A10, A11
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each Mk is a 2x2 matrix product.
    # We iterate through k=0..6 and the 4 schoolbook terms for each.
    
    # Helper to add a term
    # term: (U_indices, V_indices, W_index, coeff)
    # U_indices: list of (i, j) with coeff
    # V_indices: list of (j, k) with coeff
    # W_index: (k, i)
    
    def add_term(u_coeffs, v_coeffs, w_idx, coeff):
        nonlocal col
        if col >= rank: return
        for (i, j), cu in u_coeffs:
            U[i * m + j][col] += cu * coeff
        for (j, k), cv in v_coeffs:
            V[j * p + k][col] += cv * coeff
        w_k, w_i = w_idx
        W[w_k * n + w_i][col] += coeff
        col += 1

    # Define A block indices
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # B block indices (m x p, but p=4 for this part)
    # B00: j in 0,1; k in 0,1
    # B01: j in 0,1; k in 2,3
    # B10: j in 2,3; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    # C block indices (p x n, but p=4 for this part)
    # C00: k in 0,1; i in 0,1
    # C01: k in 0,1; i in 2,3
    # C10: k in 2,3; i in 0,1
    # C11: k in 2,3; i in 2,3
    
    # M0 = (A00+A11)(B00+B11)
    # A' = A00+A11, B' = B00+B11
    # Terms:
    # 1. A'_00 B'_00 -> C_00 (top-left of M0, which is C00 top-left? No, M0 is 2x2 block)
    # M0 is a 2x2 matrix.
    # M0[0,0] contributes to C00[0,0]
    # M0[0,1] contributes to C00[0,1]
    # M0[1,0] contributes to C00[1,0]
    # M0[1,1] contributes to C00[1,1]
    # Wait, C00 is a 2x2 block.
    # C00 = M0 + ...
    # So M0[0,0] adds to C00[0,0].
    
    # Let's define the 4 terms for a generic product A' B' -> M
    # M[0,0] = A'[0,0]B'[0,0] + A'[0,1]B'[1,0]
    # M[0,1] = A'[0,0]B'[0,1] + A'[0,1]B'[1,1]
    # M[1,0] = A'[1,0]B'[0,0] + A'[1,1]B'[1,0]
    # M[1,1] = A'[1,0]B'[0,1] + A'[1,1]B'[1,1]
    
    # We need to map A'[r,c] to global A indices.
    # A' = A00+A11.
    # A'[0,0] = A[0,0] + A[2,2]
    # A'[0,1] = A[0,1] + A[2,3]
    # A'[1,0] = A[1,0] + A[3,2]
    # A'[1,1] = A[1,1] + A[3,3]
    
    # B' = B00+B11.
    # B'[0,0] = B[0,0] + B[2,2]
    # B'[0,1] = B[0,1] + B[2,3]
    # B'[1,0] = B[1,0] + B[3,2]
    # B'[1,1] = B[1,1] + B[3,3]
    
    # Term 1: A'[0,0] B'[0,0] -> M[0,0] -> C00[0,0]
    # C00[0,0] is global C[0,0]. (k=0, i=0)
    
    # This is getting complex to hardcode manually in thought. I will write the code to generate it.
    
    # List of (A_block, B_block, C_block, sign) for Strassen
    # M_k = A'_k B'_k
    # C_block = sum s_k M_k
    
    # M0: A00+A11, B00+B11. Contributes to C00(+1), C11(+1)
    # M1: A10+A11, B00. Contributes to C10(+1), C11(-1)
    # M2: A00, B01-B11. Contributes to C01(+1), C11(+1)
    # M3: A11, B10-B00. Contributes to C00(+1), C10(+1)
    # M4: A00+A01, B11. Contributes to C00(-1), C01(+1)
    # M5: A10-A00, B00+B01. Contributes to C11(+1)
    # M6: A01-A11, B10+B11. Contributes to C00(+1)
    
    # For each M_k, we expand into 4 schoolbook terms.
    # Each term is (A'_term, B'_term, M_pos)
    # M_pos determines which part of C_block it goes to.
    
    # I will implement a helper to build the linear forms.
    
    return {"U": U, "V": V, "W": W}
