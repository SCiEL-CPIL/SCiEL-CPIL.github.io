import numpy as np
import time

_START = time.time()
DEADLINE = _START + 115  # 5s safety margin before 120s

def construct():
    n, m, p = 4, 4, 7
    
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use Strassen for the 4x4x4 part (first 4 cols of B) 
    # and a more efficient approach for the remaining 3 cols.
    
    # Actually, let me reconsider the structure.
    # C = A * B where A is 4x4, B is 4x7.
    # We can think of B as [B1 | B2] where B1 is 4x4 and B2 is 4x3.
    # Then C = [A*B1 | A*B2].
    
    # For A*B1 (4x4 * 4x4 = 4x4), we use Strassen with rank 49.
    # For A*B2 (4x4 * 4x3 = 4x3), we need a better decomposition than schoolbook (rank 48).
    
    # Let's use a block approach for A*B2:
    # A = [A00 A01; A10 A11] (2x2 blocks of 2x2)
    # B2 = [B2_top; B2_bot] where B2_top is 2x3 and B2_bot is 2x3
    # C2 = [A00*B2_top + A01*B2_bot; A10*B2_top + A11*B2_bot]
    
    # Each block multiplication is 2x2 * 2x3 = 2x3.
    # Schoolbook for 2x2*2x3 uses rank 12.
    # Can we do better? The tensor rank of 2x2*2x3...
    # Let's just use schoolbook for the 2x2*2x3 parts.
    # 4 blocks * 12 = 48. Total rank = 49 + 48 = 97.
    
    # But the current program has rank 93. Let me check how.
    # Current: 49 + 44 = 93. So they used rank 11 for 2x2*2x3? That seems low.
    # Wait, 2x2*2x3 schoolbook is 2*2*3 = 12 terms.
    # Maybe they combined terms?
    
    # Let me just implement Strassen for 4x4*4x4 and schoolbook for 4x4*4x3.
    # Rank = 49 + 48 = 97. This is worse than 93.
    
    # Let me look at the current program's approach more carefully.
    # They use 4*(2x2x3) with rank 44, meaning 11 per 2x2x3 block.
    # That's impossible with standard algorithms. Unless they share terms.
    
    # Actually, I think the key insight is that we can decompose the entire 
    # 4x4*4x7 problem more cleverly.
    
    # Let me try a different approach: decompose B into two 4x4 matrices, 
    # one with 4 non-zero columns and one with 3 non-zero columns.
    # But that's what I said before.
    
    # For now, let me implement the straightforward approach and see if I can 
    # optimize the 4x4*4x3 part.
    
    # Strassen 4x4*4x4: rank 49
    # Then for the remaining 3 columns, use schoolbook: rank 4*4*3 = 48
    # Total: 97
    
    # But we can do better for 4x4*4x3 by using a block Strassen-like approach.
    # A = [A00 A01; A10 A11], B' (4x3) = [B00; B10] where B00, B10 are 2x3
    # C = [A00*B00 + A01*B10; A10*B00 + A11*B10]
    
    # For each 2x2*2x3 multiplication, use rank 12 (schoolbook).
    # 4 such multiplications: rank 48.
    
    # To improve, note that A00*B00 and A10*B00 share B00, and 
    # A01*B10 and A11*B10 share B10.
    # But the A factors are different, so we can't directly share.
    
    # Alternative: Precompute combinations of A blocks, then multiply by B' blocks.
    # M1 = (A00+A10)*B00 -> contributes to C0 and C1
    # M2 = A01*B10 -> contributes to C0
    # M3 = A11*B10 -> contributes to C1
    # But M1 doesn't directly help because we need A00*B00 and A10*B00 separately.
    
    # I think rank 97 is a safe upper bound. Let me implement that.
    
    def build_strassen_4x4x4():
        """Build Strassen decomposition for 4x4*4x4, rank 49.
        Returns U, V, W lists of length 16, 16, 16 for the 4x4x4 part."""
        
        # We need to define the Strassen algorithm for 4x4*4x4.
        # This uses 7 recursive calls on 2x2*2x2, each with rank 7.
        # Total rank = 7*7 = 49.
        
        # The outer Strassen terms for 4x4:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each outer term, we have an inner 2x2*2x2 Strassen with 7 sub-terms.
        # Each sub-term has coefficients in U, V, W.
        
        U = [[0.0]*49 for _ in range(16)]
        V = [[0.0]*49 for _ in range(16)]
        W = [[0.0]*49 for _ in range(16)]
        
        # Define the 7 Strassen terms for 2x2*2x2
        # Each term: (u_coeffs, v_coeffs, w_coeffs) where each is a dict from (i,j) to coeff
        # For 2x2*2x2, U indices are (i,j) for A[i,j], V for B[j,k], W for C[i,k]
        
        # Standard Strassen 2x2*2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        inner_terms = [
            # M1: (A00+A11)(B00+B11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2: (A10+A11)B00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3: A00(B01-B11)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4: A11(B10-B00)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5: (A00+A01)B11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6: (A10-A00)(B00+B01)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7: (A01-A11)(B10+B11)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        # Outer terms: for each of the 7 outer terms, we need to map the 2x2 blocks
        # to 4x4 indices.
        # Block (bi,bj) of A maps to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # Block (ci,cj) of B maps to rows 2*ci..2*ci+1, cols 2*cj..2*cj+1
        # Block (di,dj) of C maps to rows 2*di..2*di+1, cols 2*dj..2*dj+1
        
        # For each outer term, we specify which A blocks, B blocks, and C blocks are involved.
        # Then for each inner term, we combine.
        
        # outer_terms[outer_idx] = (a_blocks, b_blocks, c_blocks)
        # where a_blocks, b_blocks, c_blocks are dicts from block_index to coeff
        # block_index for A: 0=A00, 1=A01, 2=A10, 3=A11 (row_major)
        # Actually, let's use (br, bc) for block row, block col.
        
        # A blocks: A[br, bc] where br, bc in {0,1}
        # B blocks: B[br, bc] where br, bc in {0,1}
        # C blocks: C[br, bc] where br, bc in {0,1}
        
        outer_terms = [
            # M1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, None),  # C blocks determined by inner
            # M2: (A[1,0]+A[1,1])*B[0,0]
            ({(1,0):1, (1,1):1}, {(0,0):1}, None),
            # M3: A[0,0]*(B[0,1]-B[1,1])
            ({(0,0):1}, {(0,1):1, (1,1):-1}, None),
            # M4: A[1,1]*(B[1,0]-B[0,0])
            ({(1,1):1}, {(1,0):1, (0,0):-1}, None),
            # M5: (A[0,0]+A[0,1])*B[1,1]
            ({(0,0):1, (0,1):1}, {(1,1):1}, None),
            # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, None),
            # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, None),
        ]
        
        # C block coefficients for each outer term (which C blocks get which inner results)
        # For each outer term, the 2x2 result is placed in specific C blocks.
        # C[br,bc] gets contributions from outer terms.
        
        # Let me re-derive:
        # M1 -> C[0,0] += M1, C[1,1] += M1
        # M2 -> C[1,0] += M2, C[1,1] -= M2
        # M3 -> C[0,1] += M3, C[1,1] += M3
        # M4 -> C[0,0] += M4, C[1,0] += M4
        # M5 -> C[0,0] -= M5, C[0,1] += M5
        # M6 -> C[1,1] += M6
        # M7 -> C[0,0] += M7
        
        c_block_coeffs = {
            0: {(0,0):1, (1,1):1},      # M1
            1: {(1,0):1, (1,1):-1},      # M2
            2: {(0,1):1, (1,1):1},       # M3
            3: {(0,0):1, (1,0):1},       # M4
            4: {(0,0):-1, (0,1):1},      # M5
            5: {(1,1):1},                # M6
            6: {(0,0):1},                # M7
        }
        
        for outer_idx in range(7):
            a_blocks, b_blocks, _ = outer_terms[outer_idx]
            c_coeffs = c_block_coeffs[outer_idx]
            
            for inner_idx in range(7):
                r = outer_idx * 7 + inner_idx
                u_in, v_in, w_in = inner_terms[inner_idx]
                
                for (li, lj), cu in u_in.items():
                    for (ab, coeff_a) in a_blocks.items():
                        ar, ac = ab
                        i = 2*ar + li
                        j = 2*ac + lj
                        u_idx = i*4 + j
                        U[u_idx][r] += cu * coeff_a
                
                for (li, lj), cv in v_in.items():
                    for (bb, coeff_b) in b_blocks.items():
                        br, bc = bb
                        i = 2*br + li
                        j = 2*bc + lj
                        v_idx = i*4 + j
                        V[v_idx][r] += cv * coeff_b
                
                for (li, lj), cw in w_in.items():
                    for (cb, coeff_c) in c_coeffs.items():
                        cr, cc = cb
                        i = 2*cr + li
                        k = 2*cc + lj
                        w_idx = k*4 + i
                        W[w_idx][r] += cw * coeff_c
        
        return U, V, W
    
    # Build the 4x4*4x3 part using schoolbook
    def build_schoolbook_4x4x3():
        """Build schoolbook decomposition for 4x4*4x3, rank 48."""
        U = [[0.0]*48 for _ in range(16)]
        V = [[0.0]*48 for _ in range(28)]  # 4x7
        W = [[0.0]*48 for _ in range(28)]  # 7x4
        
        r = 0
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):  # cols 4,5,6
                    u_idx = i*4 + j
                    v_idx = j*7 + k
                    w_idx = k*4 + i
                    U[u_idx][r] = 1.0
                    V[v_idx][r] = 1.0
                    W[w_idx][r] = 1.0
                    r += 1
        
        return U, V, W
    
    if time.time() - _START > DEADLINE:
        # Fallback to schoolbook
        rank = 112
        U = np.zeros((16, rank))
        V = np.zeros((28, rank))
        W = np.zeros((28, rank))
        r = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, r] = 1.0
                    V[j*7+k, r] = 1.0
                    W[k*4+i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build Strassen part for cols 0-3
    U_s, V_s, W_s = build_strassen_4x4x4()
    
    # Build schoolbook part for cols 4-6
    U_b, V_b, W_b = build_schoolbook_4x4x3()
    
    # Combine
    total_rank = 49 + 48
    U_final = [[0.0]*total_rank for _ in range(16)]
    V_final = [[0.0]*total_rank for _ in range(28)]
    W_final = [[0.0]*total_rank for _ in range(28)]
    
    for i in range(16):
        for r in range(49):
            U_final[i][r] = U_s[i][r]
            W_final[i][r] = W_s[i][r]
        for r in range(48):
            U_final[i][49+r] = U_b[i][r]
    
    for j in range(28):
        for r in range(49):
            V_final[j][r] = V_s[j][r] if j < 16 else 0.0
        for r in range(48):
            V_final[j][49+r] = V_b[j][r]
    
    for i in range(28):
        for r in range(49):
            W_final[i][r] = W_s[i][r] if i < 16 else 0.0
        for r in range(48):
            W_final[i][49+r] = W_b[i][r]
    
    # Verify
    if time.time() - _START < DEADLINE:
        U_np = np.array(U_final, dtype=float)
        V_np = np.array(V_final, dtype=float)
        W_np = np.array(W_final, dtype=float)
        
        target = np.zeros((16, 28, 28))
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    target[i*4+j, j*7+k, k*4+i] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        
        max_err = np.max(np.abs(recon - target))
        if max_err > 1e-10:
            # Fallback
            rank = 112
            U = np.zeros((16, rank))
            V = np.zeros((28, rank))
            W = np.zeros((28, rank))
            r = 0
            for i in range(4):
                for j in range(4):
                    for k in range(7):
                        U[i*4+j, r] = 1.0
                        V[j*7+k, r] = 1.0
                        W[k*4+i, r] = 1.0
                        r += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_final, "V": V_final, "W": W_final}
