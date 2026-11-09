import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Partitions
    # n=3: I2={0,1}, I1={2}
    # m=5: J2={0,1}, J2p={2,3}, J1={4}
    # p=6: K2={0,1}, K2p={2,3}, K2pp={4,5}
    
    I2 = [0, 1]
    I1 = [2]
    J2 = [0, 1]
    J2p = [2, 3]
    J1 = [4]
    K2 = [0, 1]
    K2p = [2, 3]
    K2pp = [4, 5]
    
    # Total rank
    # 6 blocks of 2x2x2 (rank 7)
    # 3 blocks of 2x1x2 (rank 1)
    # 6 blocks of 1x2x2 (rank 1)
    # 3 blocks of 1x1x2 (rank 1)
    rank = 6 * 7 + 3 * 1 + 6 * 1 + 3 * 1
    # 42 + 3 + 6 + 3 = 54
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a 2x2x2 block using Strassen
    def add_strassen_block(i_offset, j_offset, k_offset, start_col):
        # Strassen coefficients for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U: A coefficients (u, v)
        # V: B coefficients (v, w)
        # W: C coefficients (w, u)
        
        # M1
        U[i_offset*2+0, j_offset*2+0, start_col+0] = 1
        U[i_offset*2+1, j_offset*2+1, start_col+0] = 1
        V[j_offset*2+0, k_offset*2+0, start_col+0] = 1
        V[j_offset*2+1, k_offset*2+1, start_col+0] = 1
        W[k_offset*2+0, i_offset*2+0, start_col+0] = 1
        W[k_offset*2+1, i_offset*2+1, start_col+0] = 1
        
        # M2
        U[i_offset*2+1, j_offset*2+0, start_col+1] = 1
        U[i_offset*2+1, j_offset*2+1, start_col+1] = 1
        V[j_offset*2+0, k_offset*2+0, start_col+1] = 1
        W[k_offset*2+0, i_offset*2+1, start_col+1] = 1
        
        # M3
        U[i_offset*2+0, j_offset*2+0, start_col+2] = 1
        V[j_offset*2+0, k_offset*2+1, start_col+2] = 1
        V[j_offset*2+1, k_offset*2+1, start_col+2] = -1
        W[k_offset*2+1, i_offset*2+0, start_col+2] = 1
        
        # M4
        U[i_offset*2+1, j_offset*2+1, start_col+3] = 1
        V[j_offset*2+1, k_offset*2+0, start_col+3] = 1
        V[j_offset*2+0, k_offset*2+0, start_col+3] = -1
        W[k_offset*2+0, i_offset*2+1, start_col+3] = 1
        
        # M5
        U[i_offset*2+0, j_offset*2+0, start_col+4] = 1
        U[i_offset*2+0, j_offset*2+1, start_col+4] = 1
        V[j_offset*2+1, k_offset*2+1, start_col+4] = 1
        W[k_offset*2+1, i_offset*2+0, start_col+4] = 1
        
        # M6
        U[i_offset*2+1, j_offset*2+0, start_col+5] = 1
        U[i_offset*2+0, j_offset*2+0, start_col+5] = -1
        V[j_offset*2+0, k_offset*2+0, start_col+5] = 1
        V[j_offset*2+0, k_offset*2+1, start_col+5] = 1
        W[k_offset*2+1, i_offset*2+1, start_col+5] = 1
        
        # M7
        U[i_offset*2+0, j_offset*2+1, start_col+6] = 1
        U[i_offset*2+1, j_offset*2+1, start_col+6] = -1
        V[j_offset*2+1, k_offset*2+0, start_col+6] = 1
        V[j_offset*2+1, k_offset*2+1, start_col+6] = 1
        W[k_offset*2+0, i_offset*2+0, start_col+6] = 1
        W[k_offset*2+1, i_offset*2+0, start_col+6] = 1

        # Note: The W indices above need to be checked against C formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My W assignments:
        # M1: W(0,0)=1, W(1,1)=1. -> C00+=1, C11+=1. Correct.
        # M2: W(0,1)=1. -> C10+=1. Correct.
        # M3: W(1,0)=1. -> C01+=1. Correct.
        # M4: W(0,1)=1. -> C10+=1. Correct.
        # M5: W(1,0)=1. -> C01+=1. Correct.
        # M6: W(1,1)=1. -> C11+=1. Correct.
        # M7: W(0,0)=1, W(1,0)=1. -> C00+=1, C01+=1. 
        # Wait. C01 = M3 + M5. M7 should not contribute to C01?
        # Let's re-verify Strassen.
        # C11 = M1 - M2 + M3 + M6.
        # C00 = M1 + M4 - M5 + M7.
        # C01 = M3 + M5.
        # C10 = M2 + M4.
        
        # My M7 assignment:
        # U: A01 - A11.
        # V: B10 + B11.
        # Product: (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
        # This term contributes to C?
        # Standard Strassen:
        # M7 = (A12 - A22)(B21 + B22) in 1-based indexing.
        # A12 is row 1 col 2 (0,1). A22 is row 2 col 2 (1,1).
        # B21 is row 2 col 1 (1,0). B22 is row 2 col 2 (1,1).
        # So M7 = (A01 - A11)(B10 + B11).
        # C11 = M1 - M2 + M3 + M6.
        # C00 = M1 + M4 - M5 + M7.
        # C01 = M3 + M5.
        # C10 = M2 + M4.
        
        # So M7 contributes to C00.
        # My W for M7: W(0,0)=1. (C00).
        # I also set W(1,0)=1 in the code above?
        # "W[k_offset*2+1, i_offset*2+0, start_col+6] = 1" -> W(1,0)=1.
        # This would add M7 to C01. That is WRONG.
        # M7 should only add to C00.
        # Let's check the formula for M7 again.
        # M7 = (A01-A11)(B10+B11).
        # Expansion: A01B10 + A01B11 - A11B10 - A11B11.
        # These terms are in C00?
        # C00 = A00B00 + A01B10.
        # C01 = A00B01 + A01B11.
        # C10 = A10B00 + A11B10.
        # C11 = A10B01 + A11B11.
        
        # The terms in M7 are A01B10 (in C00), A01B11 (in C01), A11B10 (in C10), A11B11 (in C11).
        # So M7 touches all 4 entries?
        # No, the linear combination in C formulas handles the cancellation.
        # C00 = ... + M7.
        # C01 = ... (no M7).
        # C10 = ... (no M7).
        # C11 = ... (no M7).
        
        # So W for M7 should be 1 for (0,0) and 0 for others.
        # My code had W(1,0)=1. I need to remove that.
        
        return start_col + 7

    # We need to fix the add_strassen_block function or just write it correctly in the loop.
    # Given the complexity, I will define the coefficients explicitly.
    
    # Let's restart the construction with a clean helper.
    
    def get_strassen_coeffs():
        # Returns lists of (u,v,val), (v,w,val), (w,u,val) for each of 7 products
        # u,v,w are 0 or 1
        prods = []
        # M1
        prods.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))
        # M2
        prods.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]))
        # M3
        prods.append(([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1)]))
        # M4
        prods.append(([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,1,1)]))
        # M5
        prods.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1)]))
        # M6
        prods.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]))
        # M7
        prods.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))
        return prods

    strassen_coeffs = get_strassen_coeffs()
    
    # 2x2x2 Blocks
    # I2 x J2 x K2
    col = add_strassen_block(0, 0, 0, col) # I2(0), J2(0), K2(0) -> offset 0,0,0
    col = add_strassen_block(0, 0, 2, col) # I2(0), J2(0), K2p(2) -> offset 0,0,2
    col = add_strassen_block(0, 0, 4, col) # I2(0), J2(0), K2pp(4) -> offset 0,0,4
    col = add_strassen_block(0, 2, 0, col) # I2(0), J2p(2), K2(0) -> offset 0,2,0
    col = add_strassen_block(0, 2, 2, col) # I2(0), J2p(2), K2p(2) -> offset 0,2,2
    col = add_strassen_block(0, 2, 4, col) # I2(0), J2p(2), K2pp(4) -> offset 0,2,4
    
    # Note: The add_strassen_block function I wrote earlier had hardcoded indices.
    # I need to make it generic or rewrite it.
    # Since I can't easily edit the function definition inside the block without breaking it,
    # I will rewrite the logic in the main loop.
    
    # Let's clear U,V,W and do it properly.
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    
    def add_block_222(i_off, j_off, k_off, c_start):
        for s in range(7):
            u_coeff, v_coeff, w_coeff = strassen_coeffs[s]
            for (u, v, val) in u_coeff:
                U[(i_off + u) * m + (j_off + v), c_start + s] = val
            for (v, w, val) in v_coeff:
                V[(j_off + v) * p + (k_off + w), c_start + s] = val
            for (w, u, val) in w_coeff:
                W[(k_off + w) * n + (i_off + u), c_start + s] = val
        return c_start + 7

    def add_block_212(i_off, j_off, k_off, c_start):
        # 2x1x2, rank 1
        # A is 2x1, B is 1x2
        # C00 = A00 B00
        # C01 = A00 B01
        # C10 = A10 B00
        # C11 = A10 B01
        # U: A00=1, A10=1. (u=0,v=0; u=1,v=0)
        # V: B00=1, B01=1. (v=0,w=0; v=0,w=1)
        # W: C00=1, C01=1, C10=1, C11=1.
        # W indices (w,u).
        # (0,0)=1, (1,0)=1, (0,1)=1, (1,1)=1.
        U[(i_off + 0) * m + (j_off + 0), c_start] = 1
        U[(i_off + 1) * m + (j_off + 0), c_start] = 1
        V[(j_off + 0) * p + (k_off + 0), c_start] = 1
        V[(j_off + 0) * p + (k_off + 1), c_start] = 1
        W[(k_off + 0) * n + (i_off + 0), c_start] = 1
        W[(k_off + 1) * n + (i_off + 0), c_start] = 1
        W[(k_off + 0) * n + (i_off + 1), c_start] = 1
        W[(k_off + 1) * n + (i_off + 1), c_start] = 1
        return c_start + 1

    def add_block_122(i_off, j_off, k_off, c_start):
        # 1x2x2, rank 1
        # A is 1x2, B is 2x2
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # U: A00=1, A01=1. (u=0,v=0; u=0,v=1)
        # V: B00=1, B01=1, B10=1, B11=1.
        # W: C00=1, C01=1. (w=0,u=0; w=1,u=0)
        U[(i_off + 0) * m + (j_off + 0), c_start] = 1
        U[(i_off + 0) * m + (j_off + 1), c_start] = 1
        V[(j_off + 0) * p + (k_off + 0), c_start] = 1
        V[(j_off + 0) * p + (k_off + 1), c_start] = 1
        V[(j_off + 1) * p + (k_off + 0), c_start] = 1
        V[(j_off + 1) * p + (k_off + 1), c_start] = 1
        W[(k_off + 0) * n + (i_off + 0), c_start] = 1
        W[(k_off + 1) * n + (i_off + 0), c_start] = 1
        return c_start + 1

    def add_block_112(i_off, j_off, k_off, c_start):
        # 1x1x2, rank 1
        # A is 1x1, B is 1x2
        # C00 = A00 B00
        # C01 = A00 B01
        U[(i_off + 0) * m + (j_off + 0), c_start] = 1
        V[(j_off + 0) * p + (k_off + 0), c_start] = 1
        V[(j_off + 0) * p + (k_off + 1), c_start] = 1
        W[(k_off + 0) * n + (i_off + 0), c_start] = 1
        W[(k_off + 1) * n + (i_off + 0), c_start] = 1
        return c_start + 1

    # Execute blocks
    # 6 x 2x2x2
    col = add_block_222(0, 0, 0, col)
    col = add_block_222(0, 0, 2, col)
    col = add_block_222(0, 0, 4, col)
    col = add_block_222(0, 2, 0, col)
    col = add_block_222(0, 2, 2, col)
    col = add_block_222(0, 2, 4, col)
    
    # 3 x 2x1x2 (I2 x J1 x K*)
    col = add_block_212(0, 4, 0, col)
    col = add_block_212(0, 4, 2, col)
    col = add_block_212(0, 4, 4, col)
    
    # 6 x 1x2x2 (I1 x J* x K*)
    col = add_block_122(2, 0, 0, col)
    col = add_block_122(2, 0, 2, col)
    col = add_block_122(2, 0, 4, col)
    col = add_block_122(2, 2, 0, col)
    col = add_block_122(2, 2, 2, col)
    col = add_block_122(2, 2, 4, col)
    
    # 3 x 1x1x2 (I1 x J1 x K*)
    col = add_block_112(2, 4, 0, col)
    col = add_block_112(2, 4, 2, col)
    col = add_block_112(2, 4, 4, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
