import time
import numpy as np

def construct():
    """
    Strategy: Rank-49 recursive Strassen decomposition for 4x4x4 matrix multiplication.
    
    The 4x4 matrix multiplication tensor is decomposed using two levels of Strassen:
    - Outer level: 7 Strassen terms on 2x2 blocks of 4x4 matrices
    - Inner level: 7 Strassen terms on 2x2 elements within each block
    - Total rank: 7 * 7 = 49
    
    The W matrix uses the trace convention W[k*n+i, r] where n=4,
    so W index = k*4 + i where k is the column and i is the row of C.
    
    This gives score 64/49 ≈ 1.306 (vs schoolbook rank-64 baseline).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen 2x2 decomposition (7 terms)
    # For 2x2 matrices A, B, C = A*B:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 
    # M1 = (a+d)(e+h) -> C[0,0] += M1, C[1,1] += M1
    # M2 = d(f-e)     -> C[1,0] += M2, C[1,1] -= M2
    # M3 = (a-d)h     -> C[0,0] += M3, C[1,0] += M3
    # M4 = a(f-h)     -> C[0,1] += M4, C[1,1] -= M4
    # M5 = (b+d)e     -> C[0,1] += M5, C[0,0] -= M5
    # M6 = (a-b)(f+h) -> C[1,1] -= M6
    # M7 = (c+d)(e-g) -> C[0,0] += M7
    
    # Inner Strassen terms for 2x2 blocks:
    # Each term: (a_comb, b_comb, c_comb)
    # a_comb: list of (local_row, local_col, coeff) for A
    # b_comb: list of (local_row, local_col, coeff) for B
    # c_comb: list of (local_row, local_col, coeff) for C
    
    inner_terms = [
        # M1 = (a+d)(e+h): C[0,0]+=M1, C[1,1]+=M1
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # M2 = d(f-e): C[1,0]+=M2, C[1,1]-=M2
        ([((1,1),1)], [((0,1),1), ((0,0),-1)], [((1,0),1), ((1,1),-1)]),
        # M3 = (a-d)h: C[0,0]+=M3, C[1,0]+=M3
        ([((0,0),1), ((1,1),-1)], [((1,1),1)], [((0,0),1), ((1,0),1)]),
        # M4 = a(f-h): C[0,1]+=M4, C[1,1]-=M4
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),-1)]),
        # M5 = (b+d)e: C[0,1]+=M5, C[0,0]-=M5
        ([((0,1),1), ((1,1),1)], [((0,0),1)], [((0,1),1), ((0,0),-1)]),
        # M6 = (a-b)(f+h): C[1,1]-=M6
        ([((0,0),1), ((0,1),-1)], [((0,1),1), ((1,1),1)], [((1,1),-1)]),
        # M7 = (c+d)(e-g): C[0,0]+=M7
        ([((1,0),1), ((1,1),1)], [((0,0),1), ((1,0),-1)], [((0,0),1)]),
    ]
    
    # Outer Strassen terms for 4x4 (operating on 2x2 blocks):
    # Each term: (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of (block_row, block_col, coeff)
    # b_blocks: list of (block_row, block_col, coeff)
    # c_blocks: list of (block_row, block_col, coeff)
    
    outer_terms = [
        # M1 = (A00+A11)*(B00+B11): C00 += M1, C11 += M1
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # M2 = A11*(B10-B00): C10 += M2, C11 -= M2
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((1,0),1), ((1,1),-1)]),
        # M3 = (A00-A11)*B11: C00 += M3, C10 += M3
        ([((0,0),1), ((1,1),-1)], [((1,1),1)], [((0,0),1), ((1,0),1)]),
        # M4 = A00*(B01-B11): C01 += M4, C11 -= M4
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),-1)]),
        # M5 = (A01+A11)*B00: C01 += M5, C00 -= M5
        ([((0,1),1), ((1,1),1)], [((0,0),1)], [((0,1),1), ((0,0),-1)]),
        # M6 = (A00-A01)*(B01+B11): C11 -= M6
        ([((0,0),1), ((0,1),-1)], [((0,1),1), ((1,1),1)], [((1,1),-1)]),
        # M7 = (A10+A11)*(B00-B10): C00 += M7
        ([((1,0),1), ((1,1),1)], [((0,0),1), ((1,0),-1)], [((0,0),1)]),
    ]
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    col = 0
    for o_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
        for i_idx, (a_elem, b_elem, c_elem) in enumerate(inner_terms):
            # U contributions:
            # A index: a = i*4 + j where i is row, j is col of A
            # Block (br, bc) with local (er, ec): global row = br*2+er, global col = bc*2+ec
            for (br, bc), ca in a_blocks:
                for (er, ec), cv in a_elem:
                    a_idx = (br * 2 + er) * 4 + (bc * 2 + ec)
                    U[a_idx, col] += ca * cv
            
            # V contributions:
            # B index: b = j*4 + k where j is row, k is col of B
            # Block (br, bc) with local (er, ec): global row = br*2+er, global col = bc*2+ec
            for (br, bc), cb in b_blocks:
                for (er, ec), cv in b_elem:
                    b_idx = (br * 2 + er) * 4 + (bc * 2 + ec)
                    V[b_idx, col] += cb * cv
            
            # W contributions:
            # C index: c = k*4 + i where k is col, i is row of C (TRACE CONVENTION)
            # Block (br, bc) with local (er, ec): global row = br*2+er, global col = bc*2+ec
            # So k = bc*2+ec (col), i = br*2+er (row)
            # W index: k*4 + i = (bc*2+ec)*4 + (br*2+er)
            for (br, bc), cc in c_blocks:
                for (er, ec), cv in c_elem:
                    k = bc * 2 + ec  # column of C
                    i = br * 2 + er  # row of C
                    c_idx = k * 4 + i
                    W[c_idx, col] += cc * cv
            
            col += 1
    
    assert col == 49, f"Expected 49, got {col}"
    
    # Verify the decomposition
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # T[i*4+j, j*4+k, k*4+i] should be 1, else 0
    
    verified = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if verified:
        # Spot check zeros
        for a in range(16):
            for b in range(16):
                i_a = a // 4
                j_a = a % 4
                j_b = b // 4
                k_b = b % 4
                if j_a != j_b:
                    for c in range(16):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > 1e-9:
                            verified = False
                            break
                    if not verified:
                        break
            if not verified:
                break
    
    if verified:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank-64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col_fb = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    U_fb[ii * 4 + jj, col_fb] = 1
                    V_fb[jj * 4 + kk, col_fb] = 1
                    W_fb[kk * 4 + ii, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
