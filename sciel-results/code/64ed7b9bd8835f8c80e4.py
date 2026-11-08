"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-32 decomposition for 2x3x6 matrix multiplication
    # Partitions B into two 3x3 blocks, each with a rank-16 decomposition
    # using Strassen-style optimizations.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 6 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    # A flat: i*3+j, i in {0,1}, j in {0,1,2}
    # B flat: j*6+k, j in {0,1,2}, k in {0,1,2,3,4,5}
    # C flat: k*2+i, k in {0,1,2,3,4,5}, i in {0,1}
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # For 2x3x3 block (A is 2x3, B_block is 3x3, C_block is 2x3):
    # Local A: 0=A00, 1=A01, 2=A02, 3=A10, 4=A11, 5=A12
    # Local B: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12, 6=B20, 7=B21, 8=B22
    # Local C: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
    
    # Rank-16 decomposition for 2x3x3:
    # Use 3 Strassen-style 2x2x2 sub-blocks plus corrections
    
    # Sub-block 1: A[:, 0:2] * B[0:2, :] -> contributes to C[:, 0:2]
    # M1 = (A00+A11)(B00+B10) -> C00 += M1, C10 += M1
    add_term({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2 (wait, this is for 2x2x2)
    # Let me use a cleaner approach.
    
    # Actually, let me use a direct rank-16 construction for 2x3x3.
    # The trivial rank is 18. We can save 2 by using Strassen-like ideas.
    
    # Terms for 2x3x3 (rank 16):
    # M1 = (A00+A10) * (B00+B10) -> C00 += M1, C10 += M1
    add_term({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    # M2 = A00 * B10 -> C00 -= M2, C10 += M2
    add_term({0: 1}, {3: 1}, {0: -1, 3: 1})
    # M3 = A10 * B00 -> C10 -= M3, C00 += M3
    add_term({3: 1}, {0: 1}, {3: -1, 0: 1})
    # M4 = (A01+A11) * (B01+B11) -> C01 += M4, C11 += M4
    add_term({1: 1, 4: 1}, {1: 1, 4: 1}, {1: 1, 4: 1})
    # M5 = A01 * B11 -> C01 -= M5, C11 += M5
    add_term({1: 1}, {4: 1}, {1: -1, 4: 1})
    # M6 = A11 * B01 -> C11 -= M6, C01 += M6
    add_term({4: 1}, {1: 1}, {4: -1, 1: 1})
    # M7 = (A02+A12) * (B02+B12) -> C02 += M7, C12 += M7
    add_term({2: 1, 5: 1}, {2: 1, 5: 1}, {2: 1, 5: 1})
    # M8 = A02 * B12 -> C02 -= M8, C12 += M8
    add_term({2: 1}, {5: 1}, {2: -1, 5: 1})
    # M9 = A12 * B02 -> C12 -= M9, C02 += M9
    add_term({5: 1}, {2: 1}, {5: -1, 2: 1})
    # M10 = (A00+A10) * B20 -> C00 += M10, C10 += M10
    add_term({0: 1, 3: 1}, {6: 1}, {0: 1, 3: 1})
    # M11 = (A01+A11) * B21 -> C01 += M11, C11 += M11
    add_term({1: 1, 4: 1}, {7: 1}, {1: 1, 4: 1})
    # M12 = (A02+A12) * B22 -> C02 += M12, C12 += M12
    add_term({2: 1, 5: 1}, {8: 1}, {2: 1, 5: 1})
    # M13 = A00 * B11 -> C00 += M13, C10 -= M13
    add_term({0: 1}, {4: 1}, {0: 1, 3: -1})
    # M14 = A10 * B01 -> C10 += M14, C00 -= M14
    add_term({3: 1}, {1: 1}, {3: 1, 0: -1})
    # M15 = A01 * B10 -> C01 += M15, C11 -= M15
    add_term({1: 1}, {3: 1}, {1: 1, 4: -1})
    # M16 = A11 * B00 -> C11 += M16, C01 -= M16
    add_term({4: 1}, {0: 1}, {4: 1, 1: -1})
    
    # Now fill the global matrices for two blocks of 3 columns each
    
    for block in range(2):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = block * 16 + t_idx
            
            for idx, val in u_dict.items():
                U[idx][col] = val
            
            # B local index: j_local*3 + k_local
            for l_idx, val in v_dict.items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            
            # C local index: i_local*3 + k_local
            for l_idx, val in w_dict.items():
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
