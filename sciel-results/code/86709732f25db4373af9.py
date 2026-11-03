"""Direct construction of rank-minimal (or near-minimal) tensor for 4x4 * 4x8."""
import time

DIMENSIONS = [4, 4, 8]

def _make_strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2.
    U: 4x7 (rows indexed i*2+j), V: 4x7 (rows j*2+k), W: 4x7 (rows k*2+i)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # Index: U[i*2+j], V[j*2+k], W[k*2+i]
    # C00 = a00b00 + a01b10, C01 = a00b01 + a01b11, C10 = a10b00 + a11b10, C11 = a10b01 + a11b11
    # Strassen:
    # M1=(a00+a11)(b00+b11), M2=(a10+a11)b00, M3=a00(b01-b11), M4=a11(b10-b00)
    # M5=(a00+a10)b11, M6=(a10-a00)(b00+b10), M7=(a01-a11)(b01+b11)
    # C00=M1-M4+M5+M6, C01=M3+M1-M5+M7, C10=M2+M4-M1+M3, C11=M1-M2+M6+M7
    terms = [
        # (U_coeffs, V_coeffs, W_coeffs)  where coeffs are dicts {index: val}
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:-1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1},   {0:0, 1:0, 2:1, 3:-1}),      # M2
        ({0:1}, {1:-1, 3:1},  {0:0, 1:1, 2:1, 3:0}),       # M3
        ({3:1}, {0:-1, 2:1},  {0:-1, 1:0, 2:1, 3:0}),      # M4
        ({0:1, 2:1}, {3:1},   {0:1, 1:-1, 2:0, 3:0}),      # M5
        ({0:-1, 2:1}, {0:1, 2:1}, {0:1, 1:0, 2:0, 3:1}),   # M6
        ({1:-1, 3:1}, {1:1, 3:1}, {0:0, 1:1, 2:0, 3:1}),   # M7
    ]
    for t, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][t] = val
        for idx, val in v_c.items():
            V[idx][t] = val
        for idx, val in w_c.items():
            W[idx][t] = val
    return U, V, W

def _verify_2x2():
    """Verify the 2x2 Strassen decomposition is correct."""
    U, V, W = _make_strassen_2x2()
    # Check: sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r] == 1 if matches, else 0
    import numpy as np
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = 7
    # Reconstruct tensor T[i,j,k] = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1.0
                val = 0.0
                for r in range(rank):
                    val += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                if abs(val - expected) > 1e-10:
                    raise ValueError(f"2x2 mismatch at ({i},{j},{k}): got {val}, expected {expected}")
    # Also verify zeros: for non-matching (i,j,k) pairs... actually all pairs should be 1 in 2x2? No.
    # In 2x2*2x2, the tensor has T[i,j,k] = 1 if... actually the bilinear form is C[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k? No, T is indexed such that the product A* B gives C where
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor is T_{i,j,k} with the property that
    # sum_{r} U_{(i,j),r} V_{(j,k),r} W_{(i,k),r} = T_{i,j,k} which should be 1 if the (i,j,k) 
    # triple corresponds to a valid path... Actually in the schoolbook, T_{i,j,k} = 1 for all i,j,k 
    # doesn't make sense. Let me reconsider.
    # The matrix multiplication tensor T has T_{i,j,k} = 1 if k... no. 
    # C[i,k] = sum_j A[i,j] B[j,k]. So the tensor is such that <T, A, B> = C[i,k] for each i,k.
    # The tensor entry T_{i,j,k} appears in the bilinear form as: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k}
    # Hmm, actually the correct interpretation: the tensor T is defined so that 
    # sum_r U_{alpha,r} V_{beta,r} W_{gamma,r} = T_{alpha,beta,gamma} where alpha=(i,j), beta=(j,k), gamma=(k,i)
    # and T_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k. So yes, all entries are 1.
    # Let me check a "zero" case: there are no zero cases; every (i,j,k) triple is valid.
    pass

def _strassen_4x4_4x4():
    """
    Rank-49 decomposition for 4x4 * 4x4 using Strassen's algorithm.
    
    U: (16, 49) - rows indexed by i*4+j for A[i,j]
    V: (16, 49) - rows indexed by j*4+k for B[j,k]
    W: (16, 49) - rows indexed by k*4+i for C[i,k] (note: output convention is W[k*n+i])
    
    Wait, the W convention is W[k*n+i] where the output is C[i,k].
    So for 4x4*4x4, W has 16 rows indexed by k*4+i.
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Get the 2x2 Strassen decomposition
    u2, v2, w2 = _make_strassen_2x2()
    # u2: 4x7, indexed by i*2+j
    # v2: 4x7, indexed by j*2+k
    # w2: 4x7, indexed by k*2+i
    
    # Block layout for 4x4:
    # A = [[A00, A01], [A10, A11]] where A00 is rows 0-1, cols 0-1
    # B = [[B00, B01], [B10, B11]] where B00 is rows 0-1, cols 0-1
    # C = [[C00, C01], [C10, C11]] where C00 is rows 0-1, cols 0-1
    
    # Block (br, bc) maps to:
    # A: rows br*2 to br*2+1, cols bc*2 to bc*2+1 -> U index (br*2+i)*4 + (bc*2+j)
    # B: rows br*2 to br*2+1, cols bc*2 to bc*2+1 -> V index (br*2+j)*4 + (bc*2+k)
    # C: rows br*2 to br*2+1, cols bc*2 to bc*2+1 -> W index (bc*2+k)*4 + (br*2+i)
    
    # Strassen multipliers:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M1 - M5 + M7
    # C10 = M2 + M4 - M1 + M3
    # C11 = M1 - M2 + M6 + M7
    
    # For each multiplier Mt, we need to combine contributions from multiple 2x2 blocks.
    # Each Mt is a product of a linear combination of A-blocks and a linear combination of B-blocks.
    # We can express this as: for each term in the linear combo of A-blocks (with sign sa) 
    # and each term in the linear combo of B-blocks (with sign sb),
    # we add sa*sb * (2x2 Strassen term) to the appropriate U, V, W entries.
    
    # Define the Strassen multipliers as (A_block_list, B_block_list)
    # where each list is [(block_id, sign), ...]
    
    strassen_mults = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)]),
        # M5
        ([(0, 1), (2, 1)], [(3, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    ]
    
    # C block assignments: for each multiplier m (0-6), which C blocks get which sign
    # C00: M1(+), M4(-), M5(+), M6(+)
    # C01: M3(+), M1(+), M5(-), M7(+)
    # C10: M2(+), M4(+), M1(-), M3(+)
    # C11: M1(+), M2(-), M6(+), M7(+)
    
    c_assignments = [
        # For multiplier m, list of (c_block_id, sign)
        [(0, 1), (1, 1), (2, -1), (3, 1)],  # M1
        [(2, 1), (3, -1)],                     # M2
        [(1, 1), (2, 1)],                      # M3
        [(0, -1), (2, 1)],                     # M4
        [(0, 1), (1, -1)],                     # M5
        [(0, 1), (3, 1)],                      # M6
        [(1, 1), (3, 1)],                      # M7
    ]
    
    # Block ID to (row_offset, col_offset)
    block_pos = {
        0: (0, 0),  # top-left
        1: (0, 2),  # top-right
        2: (2, 0),  # bottom-left
        3: (2, 2),  # bottom-right
    }
    
    for m in range(7):
        a_terms, b_terms = strassen_mults[m]
        c_terms = c_assignments[m]
        
        for (a_block, a_sign) in a_terms:
            a_row_off, a_col_off = block_pos[a_block]
            for (b_block, b_sign) in b_terms:
                b_row_off, b_col_off = block_pos[b_block]
                for (c_block, c_sign) in c_terms:
                    c_row_off, c_col_off = block_pos[c_block]
                    
                    # For this combination, add a_sign * b_sign * c_sign * (2x2 strassen term)
                    # to the appropriate positions
                    scale = a_sign * b_sign * c_sign
                    if scale == 0:
                        continue
                    
                    # The 2x2 Strassen term r (0-6) contributes to:
                    # U at position corresponding to A block (a_row_off, a_col_off)
                    # V at position corresponding to B block (b_row_off, b_col_off)
                    # W at position corresponding to C block (c_row_off, c_col_off)
                    
                    for r in range(7):
                        col_idx = m * 7 + r
                        
                        # U: A block (a_row_off, a_col_off), 2x2 element (i,j) -> U[(a_row_off+i)*4 + (a_col_off+j), col_idx]
                        for i in range(2):
                            for j in range(2):
                                u2_val = u2[i*2+j][r]
                                if u2_val != 0:
                                    U[(a_row_off+i)*4 + (a_col_off+j)][col_idx] += scale * u2_val
                        
                        # V: B block (b_row_off, b_col_off), 2x2 element (j,k) -> V[(b_row_off+j)*4 + (b_col_off+k), col_idx]
                        for j in range(2):
                            for k in range(2):
                                v2_val = v2[j*2+k][r]
                                if v2_val != 0:
                                    V[(b_row_off+j)*4 + (b_col_off+k)][col_idx] += scale * v2_val
                        
                        # W: C block (c_row_off, c_col_off), 2x2 element (k,i) -> W[(c_col_off+k)*4 + (c_row_off+i), col_idx]
                        for k in range(2):
                            for i in range(2):
                                w2_val = w2[k*2+i][r]
                                if w2_val != 0:
                                    W[(c_col_off+k)*4 + (c_row_off+i)][col_idx] += scale * w2_val
    
    return U, V, W


def _verify_strassen_4x4():
    """Verify the 4x4 Strassen decomposition."""
    import numpy as np
    U, V, W = _strassen_4x4_4x4()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = 49
    
    # Check: for all i,j,k, sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should be 1
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = sum(U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"4x4 Strassen mismatch at ({i},{j},{k}): got {val}")
    return True


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    deadline = time.time() + 100
    
    # Use 2x Strassen 4x4*4x4 for the two halves of B (columns 0-3 and 4-7)
    # Total rank = 98
    
    U4, V4, W4 = _strassen_4x4_4x4()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # First half: B columns 0-3 (k=0,1,2,3)
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            # V: for first half, B rows j (0-3), cols k (0-3) -> V[j*8+k, r]
            # V4 has rows j*4+k for k=0-3
            for j in range(4):
                for k in range(4):
                    V[j * 8 + k][r] = V4[j * 4 + k][r]
            # W: for first half, output C[i,k] with k=0-3 -> W[k*4+i, r]
            for k in range(4):
                for i2 in range(4):
                    W[k * 4 + i2][r] = W4[k * 4 + i2][r]
    
    # Second half: B columns 4-7 (k=4,5,6,7)
    for i in range(16):
        for r in range(49):
            r2 = 49 + r
            U[i][r2] = U4[i][r]
            # V: for second half, B rows j (0-3), cols k (4-7) -> V[j*8+k, r2]
            for j in range(4):
                for k in range(4):
                    V[j * 8 + (k + 4)][r2] = V4[j * 4 + k][r]
            # W: for second half, output C[i,k] with k=4-7 -> W[k*4+i, r2]
            for k in range(4):
                for i2 in range(4):
                    W[(k + 4) * 4 + i2][r2] = W4[k * 4 + i2][r]
    
    # Verify the full decomposition if time permits
    if time.time() < deadline - 20:
        try:
            import numpy as np
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            # Sample a few entries to verify
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        val = sum(U_arr[i*4+j, r] * V_arr[j*8+k, r] * W_arr[k*4+i, r] for r in range(rank))
                        if abs(val - 1.0) > 1e-10:
                            raise ValueError(f"Full decomposition mismatch at ({i},{j},{k}): got {val}")
        except Exception as e:
            # If verification fails, fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
