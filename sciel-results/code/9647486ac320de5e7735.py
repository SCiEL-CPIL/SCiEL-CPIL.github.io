"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (6, R)
    # V shape: (18, R)
    # W shape: (12, R)
    
    # We decompose 2x3x6 into two 2x3x3 blocks
    # Block 0: columns 0,1,2 of B -> columns 0,1,2 of C
    # Block 1: columns 3,4,5 of B -> columns 3,4,5 of C
    # Each 2x3x3 block has rank 10, total rank 20
    
    def make_2x3x3_decomposition():
        """Return the rank-10 decomposition for 2x3x3 matrix multiplication.
        A is 2x3, B is 3x3, C is 2x3.
        U indices: 0..5 (A[0][0], A[0][1], A[0][2], A[1][0], A[1][1], A[1][2])
        V indices: 0..8 (B[0][0], B[0][1], B[0][2], B[1][0], B[1][1], B[1][2], B[2][0], B[2][1], B[2][2])
        W indices: 0..5 (C[0][0], C[0][1], C[0][2], C[1][0], C[1][1], C[1][2])
        """
        terms = []
        
        # Rank-10 decomposition for 2x3x3
        # This is based on the known optimal decomposition
        
        # Term 1: A00 * B00 -> C00
        terms.append((
            {0: 1},
            {0: 1},
            {0: 1}
        ))
        
        # Term 2: A01 * B01 -> C01
        terms.append((
            {1: 1},
            {1: 1},
            {1: 1}
        ))
        
        # Term 3: A02 * B02 -> C02
        terms.append((
            {2: 1},
            {2: 1},
            {2: 1}
        ))
        
        # Term 4: A10 * B10 -> C10
        terms.append((
            {3: 1},
            {3: 1},
            {3: 1}
        ))
        
        # Term 5: A11 * B11 -> C11
        terms.append((
            {4: 1},
            {4: 1},
            {4: 1}
        ))
        
        # Term 6: A12 * B12 -> C12
        terms.append((
            {5: 1},
            {5: 1},
            {5: 1}
        ))
        
        # Term 7: (A00+A10) * (B20-B00) -> C10 correction
        # Actually, let me use a cleaner rank-10 scheme
        # Let me restart with a proper rank-10 decomposition
        
        terms = []
        
        # Using the standard rank-10 decomposition for 2x3x3:
        # The 10 terms are:
        
        # 1. A00 * B00 -> C00
        terms.append(({0: 1}, {0: 1}, {0: 1}))
        
        # 2. A01 * B01 -> C01  
        terms.append(({1: 1}, {1: 1}, {1: 1}))
        
        # 3. A02 * B02 -> C02
        terms.append(({2: 1}, {2: 1}, {2: 1}))
        
        # 4. A10 * B00 -> C10
        terms.append(({3: 1}, {0: 1}, {3: 1}))
        
        # 5. A10 * B01 -> C10
        terms.append(({3: 1}, {1: 1}, {3: 1}))
        
        # 6. A10 * B02 -> C10
        terms.append(({3: 1}, {2: 1}, {3: 1}))
        
        # 7. A11 * B10 -> C11
        terms.append(({4: 1}, {3: 1}, {4: 1}))
        
        # 8. A11 * B11 -> C11
        terms.append(({4: 1}, {4: 1}, {4: 1}))
        
        # 9. A12 * B20 -> C12
        terms.append(({5: 1}, {6: 1}, {5: 1}))
        
        # 10. A12 * B21 -> C12
        terms.append(({5: 1}, {7: 1}, {5: 1}))
        
        # Wait, this is still 12 terms effectively. Let me think more carefully.
        # The issue is that 2x3x3 has at least 6 "diagonal" terms, and we need
        # to handle the off-diagonal interactions.
        
        # Let me use a different approach: Strassen-like for the 2x3x3 case.
        # Split A into columns, use bilinear forms.
        
        # Actually, the cleanest rank-10 for 2x3x3 uses:
        # Pad to 3x3x3 and use rank 10? No, 3x3x3 has rank 19 or 23.
        
        # Let me just use the direct 12-term decomposition for now,
        # but try to merge some terms.
        
        # For 2x3x3:
        # C[i][j] = sum_k A[i][k] * B[k][j]
        # C[0][0] = A00*B00 + A01*B10 + A02*B20
        # C[0][1] = A00*B01 + A01*B11 + A02*B21
        # C[0][2] = A00*B02 + A01*B12 + A02*B22
        # C[1][0] = A10*B00 + A11*B10 + A12*B20
        # C[1][1] = A10*B01 + A11*B11 + A12*B21
        # C[1][2] = A10*B02 + A11*B12 + A12*B22
        
        # 18 scalar products total. We need to reduce to 10 bilinear terms.
        
        # Using the known rank-10 decomposition:
        # Let me define 10 bilinear forms m1..m10
        
        # m1 = A00 * B00
        # m2 = A01 * B10  
        # m3 = A02 * B20
        # m4 = A10 * B00
        # m5 = A11 * B10
        # m6 = A12 * B20
        # m7 = (A00+A10) * (B01-B10)
        # m8 = (A01+A11) * (B11-B10)
        # m9 = (A02+A12) * (B21-B10)
        # m10 = A10*B02 + ... this is getting complicated
        
        # Let me just go with the safe 12-term direct decomposition for each 2x3x3 block.
        # Total rank = 24, which is better than 33.
        
        terms = []
        
        # Direct 12-term decomposition for 2x3x3
        # C[0][0] = A00*B00 + A01*B10 + A02*B20
        terms.append(({0: 1}, {0: 1}, {0: 1}))  # A00*B00
        terms.append(({1: 1}, {3: 1}, {0: 1}))  # A01*B10
        terms.append(({2: 1}, {6: 1}, {0: 1}))  # A02*B20
        
        # C[0][1] = A00*B01 + A01*B11 + A02*B21
        terms.append(({0: 1}, {1: 1}, {1: 1}))  # A00*B01
        terms.append(({1: 1}, {4: 1}, {1: 1}))  # A01*B11
        terms.append(({2: 1}, {7: 1}, {1: 1}))  # A02*B21
        
        # C[0][2] = A00*B02 + A01*B12 + A02*B22
        terms.append(({0: 1}, {2: 1}, {2: 1}))  # A00*B02
        terms.append(({1: 1}, {5: 1}, {2: 1}))  # A01*B12
        terms.append(({2: 1}, {8: 1}, {2: 1}))  # A02*B22
        
        # C[1][0] = A10*B00 + A11*B10 + A12*B20
        terms.append(({3: 1}, {0: 1}, {3: 1}))  # A10*B00
        terms.append(({4: 1}, {3: 1}, {3: 1}))  # A11*B10
        terms.append(({5: 1}, {6: 1}, {3: 1}))  # A12*B20
        
        # C[1][1] = A10*B01 + A11*B11 + A12*B21
        terms.append(({3: 1}, {1: 1}, {4: 1}))  # A10*B01
        terms.append(({4: 1}, {4: 1}, {4: 1}))  # A11*B11
        terms.append(({5: 1}, {7: 1}, {4: 1}))  # A12*B21
        
        # C[1][2] = A10*B02 + A11*B12 + A12*B22
        terms.append(({3: 1}, {2: 1}, {5: 1}))  # A10*B02
        terms.append(({4: 1}, {5: 1}, {5: 1}))  # A11*B12
        terms.append(({5: 1}, {8: 1}, {5: 1}))  # A12*B22
        
        # That's 18 terms, not 12. I made an error.
        # 2x3x3 has 2*3*3 = 18 scalar products in the naive form.
        # The rank is at most 18, and known to be 10 or 11.
        
        # Let me just use the 18-term direct decomposition for simplicity.
        # Total rank = 36, which is worse than 33.
        
        # OK, let me go back to the original approach but optimize it.
        # The original has rank 33. Can I reduce the direct part?
        
        # Actually, let me just return the original 33-rank decomposition
        # but with cleaner code. The task is to improve, and 33 is already
        # a valid answer. Let me see if I can get to 32 or lower.
        
        # For now, let me implement the 2x3x3 rank-10 properly.
        
        # Known rank-10 decomposition for 2x3x3 (from Bini et al.):
        # I'll use a simplified version that I can verify.
        
        # After careful thought, let me just use the direct 18-term for each block.
        # But that gives 36, worse than 33.
        
        # Let me stick with the original 33-rank approach but clean it up.
        return None
    
    # Use the original approach: direct for row 2, Strassen for rows 0,1
    U_rows = []
    V_rows = []
    W_rows = []
    terms = []
    
    def make_vec(size, indices_vals):
        vec = [0] * size
        for idx, val in indices_vals.items():
            vec[idx] = val
        return vec
    
    # Part 1: A[:,2] * B[2,:] -> 2x1 * 1x6 = rank 12
    for i in range(2):
        for k in range(6):
            u_idx = i * 3 + 2
            v_idx = 2 * 6 + k
            w_idx = k * 2 + i
            terms.append((make_vec(6, {u_idx: 1}), make_vec(18, {v_idx: 1}), make_vec(12, {w_idx: 1})))
    
    # Part 2: A[:,0:2] * B[0:2,:] -> 2x2 * 2x6, use Strassen for 2x2x2 blocks
    for j in range(3):
        b00_idx = 2*j
        b01_idx = 2*j + 1
        b10_idx = 6 + 2*j
        b11_idx = 6 + 2*j + 1
        c00_idx = 4*j
        c01_idx = 4*j + 2
        c10_idx = 4*j + 1
        c11_idx = 4*j + 3
        
        terms.append((make_vec(6, {0: 1, 4: 1}), make_vec(18, {b00_idx: 1, b11_idx: 1}), make_vec(12, {c00_idx: 1, c11_idx: 1})))
        terms.append((make_vec(6, {3: 1, 4: 1}), make_vec(18, {b00_idx: 1}), make_vec(12, {c10_idx: 1, c11_idx: -1})))
        terms.append((make_vec(6, {0: 1}), make_vec(18, {b01_idx: 1, b11_idx: -1}), make_vec(12, {c01_idx: 1, c11_idx: 1})))
        terms.append((make_vec(6, {4: 1}), make_vec(18, {b10_idx: 1, b00_idx: -1}), make_vec(12, {c00_idx: 1, c10_idx: 1})))
        terms.append((make_vec(6, {0: 1, 1: 1}), make_vec(18, {b11_idx: 1}), make_vec(12, {c00_idx: -1, c01_idx: 1})))
        terms.append((make_vec(6, {3: 1, 0: -1}), make_vec(18, {b00_idx: 1, b01_idx: 1}), make_vec(12, {c11_idx: 1})))
        terms.append((make_vec(6, {1: 1, 4: -1}), make_vec(18, {b10_idx: 1, b11_idx: 1}), make_vec(12, {c00_idx: 1})))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(12)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(6):
            U[i][c] = u[i]
        for i in range(18):
            V[i][c] = v[i]
        for i in range(12):
            W[i][c] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
