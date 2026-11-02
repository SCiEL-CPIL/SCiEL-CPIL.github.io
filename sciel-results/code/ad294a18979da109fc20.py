import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4_decomposition():
    """
    Constructs the rank-49 decomposition for 4x4x4 matrix multiplication
    using recursive Strassen.
    
    The 4x4 matrices are split into 2x2 blocks.
    The 7 Strassen products for the 4x4 level are computed, and each of those
    2x2 products is computed using the 7 Strassen products for the 2x2 level.
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)

    # Strassen 2x2 decomposition coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Each M_i is a bilinear form.
    # We represent each M_i as a list of (coeff_a, coeff_b) for the 4 entries of A and B.
    # A entries: 00, 01, 10, 11
    # B entries: 00, 01, 10, 11
    
    # For 2x2 Strassen, the rank is 7.
    # We define the 7 rank-1 tensors for 2x2 multiplication.
    # Each term is (u_vec, v_vec, w_vec) where u, v are length 4, w is length 4.
    
    def vec(idx_dict):
        # Convert {pos: val} to [v0, v1, v2, v3] for 2x2
        return [idx_dict.get((i,j), 0) for i in range(2) for j in range(2)]
        
    # M1: (a+d)(e+h) = (a+0b+0c+d) * (e+0f+0g+h)
    # u: a=1, d=1 -> [1, 0, 0, 1]
    # v: e=1, h=1 -> [1, 0, 0, 1]
    # w: C00+=1, C11+=1 -> [1, 0, 0, 1]
    s1 = (vec({(0,0):1, (1,1):1}), vec({(0,0):1, (1,1):1}), vec({(0,0):1, (1,1):1}))
    
    # M2: (c+d)e = (0a+0b+c+d) * (e+0f+0g+0h)
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # w: C10+=1 -> [0, 0, 1, 0]
    s2 = (vec({(1,0):1, (1,1):1}), vec({(0,0):1}), vec({(1,0):1}))
    
    # M3: a(f-h) = (a+0b+0c+0d) * (0e+f+0g-h)
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # w: C01+=1, C11+=1 -> [0, 1, 0, 1]
    s3 = (vec({(0,0):1}), vec({(0,1):1, (1,1):-1}), vec({(0,1):1, (1,1):1}))
    
    # M4: d(g-e) = (0a+0b+0c+d) * (-e+0f+g+0h)
    # u: [0, 0, 0, 1]
    # v: [-1, 0, 1, 0]
    # w: C00+=1, C10+=1 -> [1, 0, 1, 0]
    s4 = (vec({(1,1):1}), vec({(0,0):-1, (1,0):1}), vec({(0,0):1, (1,0):1}))
    
    # M5: (a+b)h = (a+b+0c+0d) * (0e+0f+0g+h)
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # w: C00-=1, C01+=1 -> [-1, 1, 0, 0]
    s5 = (vec({(0,0):1, (0,1):1}), vec({(1,1):1}), vec({(0,0):-1, (0,1):1}))
    
    # M6: (c-a)(e+f) = (-a+0b+c+0d) * (e+f+0g+0h)
    # u: [-1, 0, 1, 0]
    # v: [1, 1, 0, 0]
    # w: C11+=1 -> [0, 0, 0, 1]
    s6 = (vec({(0,0):-1, (1,0):1}), vec({(0,0):1, (0,1):1}), vec({(1,1):1}))
    
    # M7: (b-d)(g+h) = (0a+b+0c-d) * (0e+0f+g+h)
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    # w: C00+=1 -> [1, 0, 0, 0]
    s7 = (vec({(0,1):1, (1,1):-1}), vec({(1,0):1, (1,1):1}), vec({(0,0):1}))
    
    strassen_2x2 = [s1, s2, s3, s4, s5, s6, s7]

    # Now, the 4x4 level Strassen.
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11.
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11.
    #
    # The 7 block products M1..M7 are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to express these block products as linear combinations of the scalar products.
    # Each block product is a 2x2 matrix product, so it requires 7 scalar multiplications (rank 7).
    # Total 7 * 7 = 49 multiplications.
    
    # Let's define the 7 block-level Strassen terms.
    # For each term k (1..7), we have:
    #   Block_A_k = linear combo of A00, A01, A10, A11
    #   Block_B_k = linear combo of B00, B01, B10, B11
    #   Block_C_k = linear combo of C00, C01, C10, C11
    
    # We iterate through the 7 block terms.
    # For each block term, we iterate through the 7 scalar terms of the 2x2 Strassen.
    
    block_strassen = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # A_block_coeffs: dict {block_idx: coeff} where block_idx is 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Same for B and C.
        
        # M1: (A00+A11)(B00+B11)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        
        # M2: (A10+A11)B00
        ({2:1, 3:1}, {0:1}, {2:1}),
        
        # M3: A00(B01-B11)
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        
        # M4: A11(B10-B00)
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        
        # M5: (A00+A01)B11
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        
        # M6: (A10-A00)(B00+B01)
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        
        # M7: (A01-A11)(B10+B11)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    # Mapping from block index to (row_offset, col_offset)
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    block_offsets = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1)
    }
    
    for k in range(7):
        a_blocks, b_blocks, c_blocks = block_strassen[k]
        
        # For this block term, we use 7 scalar multiplications (Strassen 2x2)
        for s in range(7):
            r = k * 7 + s
            u_scalar, v_scalar, w_scalar = strassen_2x2[s]
            
            # u_scalar is a vector of length 4, representing coefficients for the 2x2 block entries.
            # The 2x2 block entries are indexed 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) within the block.
            
            # Construct U, V, W contributions
            # U index: i*m + j. i,j in 0..3.
            # Block (br, bc) contains rows 2*br..2*br+1 and cols 2*bc..2*bc+1.
            # Local row lr, local col lc in 0..1.
            
            # u_scalar[l] corresponds to local entry (lr, lc) where l = lr*2 + lc.
            # We need to map this to global (i, j).
            
            # For U (A matrix):
            for b_idx, coeff_a in a_blocks.items():
                br, bc = block_offsets[b_idx]
                # For each local entry in the 2x2 block
                for l in range(4):
                    u_val = u_scalar[l]
                    if u_val == 0: continue
                    lr = l // 2
                    lc = l % 2
                    i = 2 * br + lr
                    j = 2 * bc + lc
                    U[i * m + j][r] += coeff_a * u_val
            
            # For V (B matrix):
            for b_idx, coeff_b in b_blocks.items():
                br, bc = block_offsets[b_idx]
                for l in range(4):
                    v_val = v_scalar[l]
                    if v_val == 0: continue
                    lr = l // 2
                    lc = l % 2
                    j = 2 * br + lr
                    k_idx = 2 * bc + lc
                    V[j * p + k_idx][r] += coeff_b * v_val
            
            # For W (C matrix):
            # W index: k*n + i. Note the trace convention W[k*n+i,r].
            # The problem statement says "output factor follows the official trace convention W[k*n+i,r]".
            # This means W is of shape (p*n, rank) and W[k*n+i, r] is the coefficient for C[i,k].
            # Let's verify: C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
            # Yes, this matches the standard tensor decomposition where the third index is the output.
            
            for c_idx, coeff_c in c_blocks.items():
                cr, cc = block_offsets[c_idx]
                # c_idx corresponds to block (cr, cc) in C.
                # Local entry (lr, lc) in this block corresponds to C[2*cr+lr, 2*cc+lc].
                # So i = 2*cr+lr, k = 2*cc+lc.
                for l in range(4):
                    w_val = w_scalar[l]
                    if w_val == 0: continue
                    lr = l // 2
                    lc = l % 2
                    i = 2 * cr + lr
                    k_idx = 2 * cc + lc
                    W[k_idx * n + i][r] += coeff_c * w_val

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
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

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    try:
        result = _strassen_4x4_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])

        if _verify_decomposition(U, V, W, n, m, p, rank):
            return result
    except Exception as e:
        pass

    # Fallback to schoolbook if Strassen fails or times out
    if time.time() > deadline:
        # Return an empty or minimal result to avoid timeout penalty? 
        # Better to return schoolbook if we have time, but if we are close to deadline, 
        # we might just return schoolbook quickly.
        pass
        
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
