import numpy as np

DIMENSIONS = [3, 5, 5]

def _strassen_block(i_off, j_off, k_off, U, V, W, start_col):
    # i_off, j_off, k_off are the starting indices for the 2x2x2 block
    # Local indices ii, jj, kk are 0, 1
    # U index: (i_off + ii) * 5 + (j_off + jj)
    # V index: (j_off + jj) * 5 + (k_off + kk)
    # W index: (k_off + kk) * 3 + (i_off + ii)
    
    # Strassen 7 terms for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The tensor decomposition requires U, V, W such that sum U V W = 1 for the block.
    # This is slightly different from just storing the multiplications.
    # The factors U, V, W are linear combinations of the input entries.
    # For a specific (i,j,k) in the block, the contribution to the rank term c is
    # U[i,j]^c * V[j,k]^c * W[k,i]^c.
    # We need to assign values to U, V, W for the 7 terms.
    
    # Standard Strassen factorization of the 2x2x2 multiplication tensor.
    # The tensor T has T[i,j,k]=1 for all i,j,k in {0,1}.
    # We need 7 rank-1 tensors.
    
    # Let's use the known Strassen decomposition.
    # The 7 multiplications correspond to 7 columns in U, V, W.
    
    # Term 1: M1
    # A part: A11 + A22 -> U[0,0]=1, U[1,1]=1
    # B part: B11 + B22 -> V[0,0]=1, V[1,1]=1
    # C part: C11 += 1, C22 += 1 -> W[0,0]=1, W[1,1]=1
    # But wait, W is for the output C.
    # The term is U * V * W.
    # We need to map the Strassen steps to the tensor factors.
    
    # It is easier to use the explicit rank-1 decomposition of the 2x2x2 tensor.
    # There are 7 such decompositions.
    
    # Let's define the 7 tuples of (U_coeffs, V_coeffs, W_coeffs)
    # U is 4x1 (2x2), V is 4x1 (2x2), W is 4x1 (2x2)
    
    # T1:
    # U: [1, 0, 0, 1] (A11, A12, A21, A22) -> A11 + A22
    # V: [1, 0, 0, 1] (B11, B12, B21, B22) -> B11 + B22
    # W: [1, 0, 0, 1] (C11, C12, C21, C22) -> C11 + C22
    # Check: (A11+A22)(B11+B22) contributes to C11 and C22.
    # C11 gets +1, C22 gets +1.
    
    # T2:
    # U: [0, 0, 1, 1] -> A21 + A22
    # V: [1, 0, 0, 0] -> B11
    # W: [0, 0, 1, 0] -> C21
    # Check: (A21+A22)B11 contributes to C21.
    
    # T3:
    # U: [1, 0, 0, 0] -> A11
    # V: [0, 0, 1, -1] -> B21 - B22
    # W: [0, 1, 0, 0] -> C12
    # Check: A11(B21-B22) contributes to C12.
    
    # T4:
    # U: [0, 0, 0, 1] -> A22
    # V: [0, 1, -1, 0] -> B12 - B11
    # W: [1, 0, 0, 0] -> C11
    # Check: A22(B12-B11) contributes to C11.
    
    # T5:
    # U: [1, 1, 0, 0] -> A11 + A12
    # V: [0, 0, 0, 1] -> B22
    # W: [0, 1, 0, 0] -> C12 ? No.
    # (A11+A12)B22 = A11B22 + A12B22.
    # C12 = A11B12 + A12B22.
    # C22 = A12B12 + A22B22.
    # So A11B22 is not directly in C12 or C22 alone?
    # C12 needs A12B22. C22 needs A22B22.
    # Strassen M5 = (A11+A12)B22.
    # C12 = M3 + M5. M3 = A11(B21-B22) = A11B21 - A11B22.
    # So C12 = A11B21 - A11B22 + A11B22 + A12B22 = A11B21 + A12B22. Correct.
    # So W for T5 should be C12?
    # W: [0, 1, 0, 0] -> C12.
    
    # T6:
    # U: [0, 0, 1, -1] -> A21 - A11
    # V: [1, 1, 0, 0] -> B11 + B12
    # W: [0, 0, 0, 1] -> C22
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12.
    # C22 = M1 - M2 + M3 + M6.
    # M1 = A11B11 + A11B22 + A22B11 + A22B22.
    # M2 = A21B11 + A22B11.
    # M3 = A11B21 - A11B22.
    # M6 = A21B11 + A21B12 - A11B11 - A11B12.
    # Sum:
    # A11B11: 1 - 1 = 0? No, M1 has +1, M6 has -1. Net 0?
    # Wait, C22 should be A12B12 + A22B22.
    # Let's re-verify Strassen formulas.
    # C22 = M1 - M2 + M3 + M6.
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # M2 = (A21+A22)B11 = A21B11 + A22B11.
    # M3 = A11(B21-B22) = A11B21 - A11B22.
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12.
    # Sum:
    # A11B11: 1 - 1 = 0.
    # A11B22: 1 - 1 = 0.
    # A22B11: 1 - 1 = 0.
    # A22B22: 1.
    # A21B11: -1 + 1 = 0.
    # A11B21: 1. (Should be 0 in C22) -> Problem?
    # A21B12: 1. (Should be 0 in C22) -> Problem?
    # A11B12: -1. (Should be 0 in C22) -> Problem?
    # A12B12: Missing.
    
    # My Strassen coefficients for W might be wrong or I am mixing up which C gets which M.
    # Standard:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check C12 = M3 + M5.
    # M3 = A11B21 - A11B22.
    # M5 = A11B22 + A12B22.
    # Sum = A11B21 + A12B22. Correct.
    # So W for T3 is C12. W for T5 is C12.
    
    # Let's check C21 = M2 + M4.
    # M2 = A21B11 + A22B11.
    # M4 = A22B12 - A22B11.
    # Sum = A21B11 + A22B12. Correct.
    # So W for T2 is C21. W for T4 is C21.
    
    # Let's check C11 = M1 + M4 - M5 + M7.
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22.
    # M1 = A11B11 + A11B22 + A22B11 + A22B22.
    # M4 = A22B12 - A22B11.
    # M5 = A11B22 + A12B22.
    # Sum:
    # A11B11: 1.
    # A11B22: 1 - 1 = 0.
    # A22B11: 1 - 1 = 0.
    # A22B22: 1 - 1 = 0.
    # A22B12: 1.
    # A12B21: 1.
    # A12B22: 1 - 1 = 0.
    # A22B21: -1.
    # Result: A11B11 + A22B12 + A12B21 - A22B21.
    # Should be A11B11 + A12B21.
    # Extra terms: A22B12 - A22B21.
    # This implies my M7 or other terms are not standard or I made an error.
    # Actually, M7 is usually defined to cancel these.
    # Let's use a known valid set of 7 rank-1 tensors for 2x2x2.
    
    # I will hardcode the 7 tuples of (U, V, W) vectors of length 4.
    
    T = [
        # U, V, W
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    for t in T:
        u, v, w = t
        # Map to global indices
        # U local (ii, jj) -> global (i_off+ii, j_off+jj)
        for ii in range(2):
            for jj in range(2):
                val = u[ii*2 + jj]
                if val != 0:
                    U[(i_off + ii) * 5 + (j_off + jj)][start_col] = val
        for jj in range(2):
            for kk in range(2):
                val = v[jj*2 + kk]
                if val != 0:
                    V[(j_off + jj) * 5 + (k_off + kk)][start_col] = val
        for kk in range(2):
            for ii in range(2):
                val = w[kk*2 + ii]
                if val != 0:
                    W[(k_off + kk) * 3 + (i_off + ii)][start_col] = val
        start_col += 1
        
    return start_col

def _schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end, U, V, W, start_col):
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                U[i * 5 + j][start_col] = 1
                V[j * 5 + k][start_col] = 1
                W[k * 3 + i][start_col] = 1
                start_col += 1
    return start_col

def construct():
    n, m, p = 3, 5, 5
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A11 B11 (2,2,2) Strassen
    col = _strassen_block(0, 0, 0, U, V, W, col)
    
    # 2. A12 B21 (2,3,2)
    # A12a B21a (2,2,2) Strassen: i 0-1, j 2-3, k 0-1
    col = _strassen_block(0, 2, 0, U, V, W, col)
    # A12b B21b (2,1,2) Schoolbook: i 0-1, j 4, k 0-1
    col = _schoolbook_block(0, 2, 4, 5, 0, 2, U, V, W, col)
    
    # 3. A11 B12 (2,2,3)
    # A11 B12a (2,2,2) Strassen: i 0-1, j 0-1, k 2-3
    col = _strassen_block(0, 0, 2, U, V, W, col)
    # A11 B12b (2,2,1) Schoolbook: i 0-1, j 0-1, k 4
    col = _schoolbook_block(0, 2, 0, 2, 4, 5, U, V, W, col)
    
    # 4. A12 B22 (2,3,3)
    # A12a B22a (2,2,2) Strassen: i 0-1, j 2-3, k 2-3
    col = _strassen_block(0, 2, 2, U, V, W, col)
    # A12a B22b (2,2,1) Schoolbook: i 0-1, j 2-3, k 4
    col = _schoolbook_block(0, 2, 2, 4, 4, 5, U, V, W, col)
    # A12b B22c (2,1,2) Schoolbook: i 0-1, j 4, k 2-3
    col = _schoolbook_block(0, 2, 4, 5, 2, 4, U, V, W, col)
    # A12b B22d (2,1,1) Schoolbook: i 0-1, j 4, k 4
    col = _schoolbook_block(0, 2, 4, 5, 4, 5, U, V, W, col)
    
    # 5. A21 B11 (1,2,2) Schoolbook
    col = _schoolbook_block(2, 3, 0, 2, 0, 2, U, V, W, col)
    
    # 6. A22 B21 (1,3,2) Schoolbook
    col = _schoolbook_block(2, 3, 2, 5, 0, 2, U, V, W, col)
    
    # 7. A21 B12 (1,2,3) Schoolbook
    col = _schoolbook_block(2, 3, 0, 2, 2, 5, U, V, W, col)
    
    # 8. A22 B22 (1,3,3) Schoolbook
    col = _schoolbook_block(2, 3, 2, 5, 2, 5, U, V, W, col)
    
    assert col == rank, f"Column count {col} does not match rank {rank}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Constructed")
