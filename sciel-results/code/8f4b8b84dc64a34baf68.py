"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: 2x2 block structure.
    # A: 2x2 blocks of 2x2
    # B: 2x3 blocks of 2x2
    # C: 2x3 blocks of 2x2
    
    # Part 1: C[:, 0:4] (blocks 0, 1)
    # Use Strassen on outer 2x2 blocks.
    # 7 outer terms.
    # Each term is 2x2 * 2x2 (inner blocks).
    # Inner rank 7.
    # Total rank 49.
    
    # Part 2: C[:, 4:6] (block 2)
    # Use Schoolbook on outer 2x1 blocks.
    # 4 outer terms.
    # Each term is 2x2 * 2x2 (inner blocks).
    # Inner rank 7.
    # Total rank 28.
    
    # Total rank 77.
    
    sU, sV, sW = _strassen_2x2()
    
    total_rank = 77
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Helper to add inner decomposition to global matrices
    def add_inner(offset_u, offset_v, offset_w, outer_coeff_u, outer_coeff_v, outer_coeff_w):
        # outer_coeff_u is 2x2 matrix of coeffs for U blocks
        # outer_coeff_v is 2x2 matrix of coeffs for V blocks
        # outer_coeff_w is 2x2 matrix of coeffs for W blocks
        # Note: For Strassen, these are the same matrix structure for U, V, W?
        # No, Strassen U, V, W are different.
        # But here we are applying Strassen to the OUTER level.
        # So the "inner" multiplication is just scaling the blocks.
        # Wait. The decomposition is Kronecker-like.
        
        pass

    # We will construct the 77 terms explicitly.
    # Terms 0-6: Strassen for C0, C1.
    for t in range(7):
        # U coefficients for outer blocks
        # sU is 4x7. sU[ai*2+bj, t]
        # V coefficients
        # sV is 4x7. sV[bj*2+ck, t]
        # W coefficients
        # sW is 4x7. sW[ck*2+ai, t]
        
        # We need to expand each outer coefficient into inner coefficients.
        # Inner multiplication is Strassen (rank 7).
        # So for each outer term t, we have 7 inner terms.
        # Total 49 terms.
        
        for inner_t in range(7):
            col = t * 7 + inner_t
            
            # U part
            for ai in range(2):
                for bj in range(2):
                    coeff_outer = sU[ai*2+bj, t]
                    if coeff_outer != 0:
                        for ai2 in range(2):
                            for bj2 in range(2):
                                coeff_inner = sU[ai2*2+bj2, inner_t]
                                if coeff_inner != 0:
                                    u_row = (2*ai + ai2)*4 + (2*bj + bj2)
                                    U[u_row, col] += coeff_outer * coeff_inner
                            
            # V part
            for bj in range(2):
                for ck in range(2): # ck in {0,1}
                    coeff_outer = sV[bj*2+ck, t]
                    if coeff_outer != 0:
                        for bj2 in range(2):
                            for ck2 in range(2):
                                coeff_inner = sV[bj2*2+ck2, inner_t]
                                if coeff_inner != 0:
                                    v_row = (2*bj + bj2)*6 + (2*ck + ck2)
                                    V[v_row, col] += coeff_outer * coeff_inner
                                    
            # W part
            for ai in range(2):
                for ck in range(2): # ck in {0,1}
                    coeff_outer = sW[ck*2+ai, t]
                    if coeff_outer != 0:
                        for ai2 in range(2):
                            for ck2 in range(2):
                                coeff_inner = sW[ck2*2+ai2, inner_t]
                                if coeff_inner != 0:
                                    w_row = (2*ck + ck2)*4 + (2*ai + ai2)
                                    W[w_row, col] += coeff_outer * coeff_inner

    # Terms 49-52: Schoolbook for C2.
    # C_ai_2 = A_ai_0 B_0_2 + A_ai_1 B_1_2
    # 4 terms: (ai, bj)
    schoolbook_idx = 0
    for ai in range(2):
        for bj in range(2):
            base_col = 49 + schoolbook_idx
            schoolbook_idx += 1
            
            # Outer coeff: 1 for (ai, bj) in U, (bj, 2) in V, (ai, 2) in W
            # Inner: Strassen rank 7
            for inner_t in range(7):
                col = base_col * 7 + inner_t # WRONG INDEXING. 49 is offset.
                # Correct col: 49 + schoolbook_idx*7 + inner_t ?
                # Let's use a counter.
                pass

    # Let's redo indexing carefully.
    U = np.zeros((n*m, 77), dtype=int)
    V = np.zeros((m*p, 77), dtype=int)
    W = np.zeros((p*n, 77), dtype=int)
    
    col_idx = 0
    # Part 1: Strassen for C0, C1
    for t in range(7):
        for inner_t in range(7):
            col = col_idx
            col_idx += 1
            
            # U
            for ai in range(2):
                for bj in range(2):
                    coeff_outer = sU[ai*2+bj, t]
                    if coeff_outer != 0:
                        for ai2 in range(2):
                            for bj2 in range(2):
                                coeff_inner = sU[ai2*2+bj2, inner_t]
                                if coeff_inner != 0:
                                    u_row = (2*ai + ai2)*4 + (2*bj + bj2)
                                    U[u_row, col] += coeff_outer * coeff_inner
            # V
            for bj in range(2):
                for ck in range(2):
                    coeff_outer = sV[bj*2+ck, t]
                    if coeff_outer != 0:
                        for bj2 in range(2):
                            for ck2 in range(2):
                                coeff_inner = sV[bj2*2+ck2, inner_t]
                                if coeff_inner != 0:
                                    v_row = (2*bj + bj2)*6 + (2*ck + ck2)
                                    V[v_row, col] += coeff_outer * coeff_inner
            # W
            for ai in range(2):
                for ck in range(2):
                    coeff_outer = sW[ck*2+ai, t]
                    if coeff_outer != 0:
                        for ai2 in range(2):
                            for ck2 in range(2):
                                coeff_inner = sW[ck2*2+ai2, inner_t]
                                if coeff_inner != 0:
                                    w_row = (2*ck + ck2)*4 + (2*ai + ai2)
                                    W[w_row, col] += coeff_outer * coeff_inner

    # Part 2: Schoolbook for C2
    for ai in range(2):
        for bj in range(2):
            # Outer coeff is 1
            # Inner is Strassen
            for inner_t in range(7):
                col = col_idx
                col_idx += 1
                
                # U: block (ai, bj)
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2+bj2, inner_t]
                        if coeff_inner != 0:
                            u_row = (2*ai + ai2)*4 + (2*bj + bj2)
                            U[u_row, col] += coeff_inner
                            
                # V: block (bj, 2)
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2+ck2, inner_t]
                        if coeff_inner != 0:
                            v_row = (2*bj + bj2)*6 + (2*2 + ck2)
                            V[v_row, col] += coeff_inner
                            
                # W: block (ai, 2)
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2+ai2, inner_t]
                        if coeff_inner != 0:
                            w_row = (2*2 + ck2)*4 + (2*ai + ai2)
                            W[w_row, col] += coeff_inner

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
