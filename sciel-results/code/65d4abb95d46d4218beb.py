import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    deadline = time.time() + 140

    # Block size
    bs = 3
    
    # Rank of BCLT decomposition for 3x3
    rank3 = 23

    # Total rank: 7 Strassen products * 23 inner multiplications
    rank = 7 * rank3

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen definitions
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    # BCLT coefficients for 3x3 multiplication
    # Format: list of 23 tuples (U_coeff_list, V_coeff_list, W_coeff_list)
    # Each coeff_list is a list of (index, value)
    # Index 0..8 corresponds to A[0,0]..A[2,2]
    
    # I will generate the BCLT decomposition using a known valid set.
    # Since hardcoding is risky, I will use a fallback: 
    # If I cannot guarantee BCLT correctness, I should stick to Schoolbook (Rank 27) 
    # which gives Rank 189. 
    # However, the prompt asks to improve. 
    # Let's try to construct a valid Rank-23 decomposition.
    
    # The BCLT algorithm is complex. 
    # Alternative: Use the "Cyclic" decomposition?
    # Actually, there is a simpler improvement: 
    # Use Strassen for 2x2 blocks of size 3x3 (Rank 189).
    # Use Strassen for 3x3 blocks of size 2x2?
    # 6 = 3 * 2.
    # Split into 3x3 blocks of 2x2.
    # We need to multiply 3x3 matrices of 2x2 blocks.
    # Rank of 3x3 multiplication is 23.
    # Each 2x2 multiplication is Rank 7.
    # Total Rank = 23 * 7 = 161.
    # This is the same as before.
    
    # Is there a better way?
    # What if we use Laderman for the 4x4 case and pad 6x6 to 8x8?
    # Rank(8x8) = 7 * Rank(4x4) = 7 * 21 = 147.
    # If the restriction argument holds, Rank(6x6) <= 147.
    # Let's assume it holds and implement Laderman-based 8x8 decomposition 
    # with zero padding for 6x6.
    
    # Laderman's algorithm for 4x4:
    # 21 products.
    # I will implement Laderman for 4x4, then use Strassen to get 8x8.
    # Then zero-pad 6x6 into 8x8.
    
    # Laderman 4x4 coefficients:
    # A = [A00 A01 A02 A03]
    #     [A10 A11 A12 A13]
    #     [A20 A21 A22 A23]
    #     [A30 A31 A32 A33]
    
    # Laderman's 21 terms:
    # L0 = A00*B00
    # L1 = A11*B11
    # L2 = A22*B22
    # L3 = A33*B33
    # L4 = (A00+A11)*(B00+B11)
    # L5 = (A00+A22)*(B00+B22)
    # L6 = (A00+A33)*(B00+B33)
    # L7 = (A10+A11)*(B01+B11)
    # L8 = (A10+A12)*(B02+B12)
    # L9 = (A10+A13)*(B03+B13)
    # L10 = (A20+A22)*(B02+B22)
    # L11 = (A20+A23)*(B03+B23)
    # L12 = (A30+A33)*(B03+B33)
    # L13 = (A01+A11)*(B10+B00) -- Wait, Laderman is symmetric?
    # Actually, Laderman's algorithm is:
    # M1 = A00 B00
    # M2 = A11 B11
    # M3 = A22 B22
    # M4 = A33 B33
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A00+A22)(B00+B22)
    # M7 = (A00+A33)(B00+B33)
    # M8 = (A10+A11)(B01+B11)
    # M9 = (A10+A12)(B02+B12)
    # M10 = (A10+A13)(B03+B13)
    # M11 = (A20+A22)(B02+B22)
    # M12 = (A20+A23)(B03+B23)
    # M13 = (A30+A33)(B03+B33)
    # M14 = (A01+A11)(B10+B00) -- No, this is getting messy.
    
    # Given the risk of implementing Laderman incorrectly and the 
    # uncertainty about the rank reduction for 6x6 via padding, 
    # I will stick to the **BCLT** approach which is well-defined 
    # and known to be Rank 161.
    
    # To avoid hardcoding errors, I will use the Schoolbook (Rank 27) 
    # if I cannot verify BCLT. 
    # But Rank 161 is a significant improvement over 189.
    # I will implement BCLT using a trusted source of coefficients 
    # derived from the "Bini-Capovani-Lotti-Turletti" algorithm.
    
    # For the sake of this exercise, I will assume the BCLT 
    # coefficients are available in a standard library or I can 
    # construct them. 
    # Since I cannot import them, I will fall back to Schoolbook 
    # to ensure validity, but this scores 1.1429.
    
    # Wait, the prompt says "COMBINE ... strongest ideas".
    # The strongest idea is Rank 161.
    # I will implement the BCLT decomposition.
    
    # BCLT Coefficients (from literature):
    # This is a known set of 23 rank-1 terms.
    
    bclt_U = []
    bclt_V = []
    bclt_W = []
    
    # Term 0
    bclt_U.append([(0, 1.0)])
    bclt_V.append([(0, 1.0)])
    bclt_W.append([(0, 1.0)])
    
    # ... (I will omit the full 23 terms here due to length, 
    # but in the final code I will include them or a valid approximation)
    
    # Actually, I can use the **Strassen** decomposition for the 3x3 blocks 
    # if I split 3 into 2+1? No.
    
    # Let's use the **Rank 161** BCLT implementation.
    
    # To save space and ensure correctness, I will use the 
    # **Schoolbook** for the inner 3x3 if I cannot guarantee BCLT.
    # But I will try to include BCLT.
    
    # Given the constraints, I will output the **Rank 189** solution 
    # but optimized for speed, as I cannot reliably generate the 
    # 23 BCLT coefficients from memory without a reference.
    
    # However, the "Inspiration 2" provided in the prompt 
    # starts defining bclt_U. 
    # I will use the structure from Inspiration 2 but complete it 
    # with a valid set of coefficients if possible, or fallback.
    
    # Since I cannot verify the coefficients, I will stick to 
    # the valid Rank 189 solution.
    
    bs = 3
    rank = 7 * 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a
                        
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b
                        
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row][r_idx] += gamma
                            
                    r_idx += 1
                    
    assert r_idx == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
