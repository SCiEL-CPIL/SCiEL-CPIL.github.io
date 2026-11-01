"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen's algorithm for 7 blocks of 2x2
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-define the 7 Strassen terms for a 2x2 multiplication
    # Each term is a tuple of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {index: coeff}
    # V_coeffs: dict {index: coeff}
    # W_coeffs: dict {index: coeff}
    
    # Indices for A (2x2): 0:00, 1:01, 2:10, 3:11
    # Indices for B (2x2 block): 0:0, 1:1, 2:0, 3:1 (local)
    # Indices for C (2x2 block): 0:00, 1:10, 2:01, 3:11 (local, mapped to global)
    
    # Strassen terms for one 2x2 block
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
    
    terms = []
    
    # Term 0: M1
    terms.append({
        'U': {0: 1, 3: 1},
        'V': {0: 1, 3: 1}, # Local B indices: 0:00, 1:01, 2:10, 3:11
        'W': {0: 1, 3: 1}  # Local C indices: 0:00, 1:10, 2:01, 3:11
    })
    
    # Term 1: M2
    terms.append({
        'U': {2: 1, 3: 1},
        'V': {0: 1},
        'W': {1: 1, 3: -1}
    })
    
    # Term 2: M3
    terms.append({
        'U': {0: 1},
        'V': {1: 1, 3: -1},
        'W': {2: 1, 3: 1}
    })
    
    # Term 3: M4
    terms.append({
        'U': {3: 1},
        'V': {2: 1, 0: -1},
        'W': {0: 1, 1: 1}
    })
    
    # Term 4: M5
    terms.append({
        'U': {0: 1, 1: 1},
        'V': {3: 1},
        'W': {0: -1, 2: 1}
    })
    
    # Term 5: M6
    terms.append({
        'U': {2: 1, 0: -1},
        'V': {0: 1, 1: 1},
        'W': {3: 1}
    })
    
    # Term 6: M7
    terms.append({
        'U': {1: 1, 3: -1},
        'V': {2: 1, 3: 1},
        'W': {0: 1}
    })
    
    # Map local B indices to global V indices
    # B is 2x14. Block j has columns 2j, 2j+1.
    # Local B index 0 (0,0) -> Global V index 0*p + 2j
    # Local B index 1 (0,1) -> Global V index 0*p + 2j+1
    # Local B index 2 (1,0) -> Global V index 1*p + 2j
    # Local B index 3 (1,1) -> Global V index 1*p + 2j+1
    
    # Map local C indices to global W indices
    # C is 2x14. Block j has columns 2j, 2j+1.
    # Local C index 0 (0,0) -> Global W index (2j)*n + 0
    # Local C index 1 (1,0) -> Global W index (2j)*n + 1
    # Local C index 2 (0,1) -> Global W index (2j+1)*n + 0
    # Local C index 3 (1,1) -> Global W index (2j+1)*n + 1
    
    num_blocks = p // 2 # 7 blocks
    
    for j in range(num_blocks):
        for s in range(7):
            r = j * 7 + s
            term = terms[s]
            
            # Fill U
            for u_idx, coeff in term['U'].items():
                U[u_idx][r] = coeff
                
            # Fill V
            for v_idx_local, coeff in term['V'].items():
                # v_idx_local: 0->(0, 2j), 1->(0, 2j+1), 2->(1, 2j), 3->(1, 2j+1)
                row = v_idx_local // 2
                col_local = v_idx_local % 2
                global_col = 2 * j + col_local
                v_idx_global = row * p + global_col
                V[v_idx_global][r] = coeff
                
            # Fill W
            for w_idx_local, coeff in term['W'].items():
                # w_idx_local: 0->(0, 2j), 1->(1, 2j), 2->(0, 2j+1), 3->(1, 2j+1)
                # Note: Local C index mapping in Strassen:
                # 0: C00, 1: C10, 2: C01, 3: C11
                # Global W index: k*n + i
                # C00: i=0, k=2j -> 2j*n
                # C10: i=1, k=2j -> 2j*n + 1
                # C01: i=0, k=2j+1 -> (2j+1)*n
                # C11: i=1, k=2j+1 -> (2j+1)*n + 1
                
                if w_idx_local == 0:
                    w_idx_global = (2 * j) * n + 0
                elif w_idx_local == 1:
                    w_idx_global = (2 * j) * n + 1
                elif w_idx_local == 2:
                    w_idx_global = (2 * j + 1) * n + 0
                elif w_idx_local == 3:
                    w_idx_global = (2 * j + 1) * n + 1
                
                W[w_idx_global][r] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
