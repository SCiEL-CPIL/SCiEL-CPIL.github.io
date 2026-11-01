import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank
    # 2 blocks of 2x2x2 (3 k-blocks each) -> 2 * 3 * 7 = 42
    # 1 block of 2x1x2 (3 k-blocks each) -> 1 * 3 * 4 = 12
    # Total 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(i_s, j_s, k_s, c_offset):
        # M1 = (A00 + A11)(B00 + B11)
        U[i_s * m + j_s][c_offset] += 1
        U[(i_s + 1) * m + (j_s + 1)][c_offset] += 1
        V[j_s * p + k_s][c_offset] += 1
        V[(j_s + 1) * p + (k_s + 1)][c_offset] += 1
        W[k_s * n + i_s][c_offset] += 1
        W[(k_s + 1) * n + (i_s + 1)][c_offset] += 1
        col = c_offset + 1
        
        # M2 = (A10 + A11) B00  (Note: A21 is A[1,0], A22 is A[1,1])
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # My indexing: i_s, j_s is top-left.
        # A11 is at (i_s, j_s). A22 is at (i_s+1, j_s+1).
        # A21 is at (i_s+1, j_s). A12 is at (i_s, j_s+1).
        
        # M2
        U[(i_s + 1) * m + j_s][col] += 1
        U[(i_s + 1) * m + (j_s + 1)][col] += 1
        V[j_s * p + k_s][col] += 1
        # C21 = M2 + M4. C[1,0] is at (k_s, i_s+1).
        W[k_s * n + (i_s + 1)][col] += 1
        col += 1
        
        # M3
        U[i_s * m + j_s][col] += 1
        V[j_s * p + (k_s + 1)][col] += 1
        V[(j_s + 1) * p + (k_s + 1)][col] -= 1
        # C12 = M3 + M5. C[0,1] is at (k_s+1, i_s).
        W[(k_s + 1) * n + i_s][col] += 1
        col += 1
        
        # M4
        U[(i_s + 1) * m + (j_s + 1)][col] += 1
        V[(j_s + 1) * p + k_s][col] += 1
        V[j_s * p + k_s][col] -= 1
        # C21 = M2 + M4. C[1,0] is at (k_s, i_s+1).
        W[k_s * n + (i_s + 1)][col] += 1
        # C11 = M1 + M4 - M5 + M7. C[0,0] is at (k_s, i_s).
        W[k_s * n + i_s][col] += 1
        col += 1
        
        # M5
        U[i_s * m + j_s][col] += 1
        U[i_s * m + (j_s + 1)][col] += 1
        V[(j_s + 1) * p + (k_s + 1)][col] += 1
        # C11 = ... - M5. C[0,0] at (k_s, i_s).
        W[k_s * n + i_s][col] -= 1
        # C12 = M3 + M5. C[0,1] at (k_s+1, i_s).
        W[(k_s + 1) * n + i_s][col] += 1
        col += 1
        
        # M6
        U[(i_s + 1) * m + j_s][col] += 1
        U[i_s * m + j_s][col] -= 1
        V[j_s * p + k_s][col] += 1
        V[j_s * p + (k_s + 1)][col] += 1
        # C22 = M1 - M2 + M3 + M6. C[1,1] at (k_s+1, i_s+1).
        W[(k_s + 1) * n + (i_s + 1)][col] += 1
        col += 1
        
        # M7
        U[i_s * m + (j_s + 1)][col] += 1
        U[(i_s + 1) * m + (j_s + 1)][col] -= 1
        V[(j_s + 1) * p + k_s][col] += 1
        V[(j_s + 1) * p + (k_s + 1)][col] += 1
        # C11 = M1 + M4 - M5 + M7. C[0,0] at (k_s, i_s).
        W[k_s * n + i_s][col] += 1
        # C22 = M1 - M2 + M3 + M6. Wait, M7 is in C11.
        # C22 formula: M1 - M2 + M3 + M6.
        # So M7 does not contribute to C22.
        # Let's re-verify Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Correct.
        col += 1
        
        return col

    # Helper to add Outer Product 2x1x2 block
    def add_outer(i_s, j_s, k_s, c_offset):
        # A is 2x1, B is 1x2
        # C00 = A00 B00
        U[i_s * m + j_s][c_offset] += 1
        V[j_s * p + k_s][c_offset] += 1
        W[k_s * n + i_s][c_offset] += 1
        col = c_offset + 1
        
        # C01 = A00 B01
        U[i_s * m + j_s][col] += 1
        V[j_s * p + (k_s + 1)][col] += 1
        W[(k_s + 1) * n + i_s][col] += 1
        col += 1
        
        # C10 = A10 B00
        U[(i_s + 1) * m + j_s][col] += 1
        V[j_s * p + k_s][col] += 1
        W[k_s * n + (i_s + 1)][col] += 1
        col += 1
        
        # C11 = A10 B01
        U[(i_s + 1) * m + j_s][col] += 1
        V[j_s * p + (k_s + 1)][col] += 1
        W[(k_s + 1) * n + (i_s + 1)][col] += 1
        col += 1
        
        return col

    # j blocks: (0, 2), (2, 2), (4, 1)
    # k blocks: (0, 2), (2, 2), (4, 2)
    
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for j_start, j_size in j_blocks:
        for k_start, k_size in k_blocks:
            if j_size == 2 and k_size == 2:
                col = add_strassen(0, j_start, k_start, col)
            elif j_size == 1 and k_size == 2:
                col = add_outer(0, j_start, k_start, col)
            else:
                # Should not happen with this partition
                pass
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
