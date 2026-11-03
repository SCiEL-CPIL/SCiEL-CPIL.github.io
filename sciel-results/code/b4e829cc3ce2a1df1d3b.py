"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # P1: (a+d)(e+h) -> A: (1,1), B: (1,1)
    # P2: (c+d)e -> A: (1,1), B: (1,0)
    # P3: a(f-h) -> A: (1,0), B: (0,1)
    # P4: d(g-e) -> A: (1,1), B: (0,1)
    # P5: (a+b)h -> A: (1,1), B: (0,1)
    # P6: (c-a)(e+f) -> A: (1,1), B: (1,1)
    # P7: (b-d)(g+h) -> A: (1,1), B: (1,1)
    
    # We need to map local (i, j, k) in 2x2x2 to global (I, J, K)
    # and determine the coefficient for U, V, W.
    # U[I, col] += coeff_A * val
    # V[J, col] += coeff_B * val
    # W[K, col] += coeff_C * val
    # Actually, the rank-1 term is u * v * w.
    # If we scale U by 2, V by 1/2, W by 1, the product is unchanged.
    # But we want integer/half-integer entries.
    # Standard Strassen uses integers.
    
    # Let's define the 7 Strassen terms for a single 2x2x2 multiplication.
    # Each term is (U_vec, V_vec, W_vec) where vectors are length 4 (for 2x2) or 2 (for 2x1)?
    # No, U is 2x2, V is 2x2, W is 2x2.
    # U_vec has non-zeros at indices corresponding to A entries.
    # V_vec has non-zeros at indices corresponding to B entries.
    # W_vec has non-zeros at indices corresponding to C entries.
    
    # Strassen Terms (local indices 0..3 for 2x2 matrix flattened row-major):
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # P1 = (A00+A11)(B00+B11)
    # U: [1, 0, 1, 0], V: [1, 0, 0, 1], W: [1, 0, 0, 1] ? No.
    # W collects results.
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    # Let's list the (U, V, W) vectors for each Pi.
    # Pi contributes to M with coefficients.
    
    # P1: A=(1,0,1,0) i.e. A00+A11. B=(1,0,0,1) i.e. B00+B11.
    # W contributions: M00 += 1, M11 += 1. So W=(1,0,0,1).
    
    # P2: A=(0,0,1,1) i.e. A10+A11. B=(1,0,0,0) i.e. B00.
    # W contributions: M10 += 1. So W=(0,0,1,0).
    
    # P3: A=(1,0,0,0) i.e. A00. B=(0,1,0,-1) i.e. B01-B11.
    # W contributions: M01 += 1, M11 += 1. So W=(0,1,0,1).
    
    # P4: A=(0,0,0,1) i.e. A11. B=(0,-1,1,0) i.e. B10-B00.
    # W contributions: M00 += 1, M10 += 1. So W=(1,0,1,0).
    
    # P5: A=(1,1,0,0) i.e. A00+A01. B=(0,0,0,1) i.e. B11.
    # W contributions: M00 -= 1, M01 += 1. So W=(-1,1,0,0).
    
    # P6: A=(0,-1,1,0) i.e. A10-A00. B=(1,1,0,0) i.e. B00+B01.
    # W contributions: M11 += 1. So W=(0,0,0,1).
    
    # P7: A=(0,1,0,-1) i.e. A01-A11. B=(0,0,1,1) i.e. B10+B11.
    # W contributions: M00 += 1. So W=(1,0,0,0).

    strassen_terms = [
        # (U_vec, V_vec, W_vec)
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # P3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # P5
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # P7
    ]

    col_idx = 0
    for j_pair in range(3):
        # 4 sub-multiplications
        # 1. A00 * B0j
        # 2. A01 * B1j
        # 3. A10 * B0j
        # 4. A11 * B1j
        
        sub_mults = [
            (0, 0, 0), # A_br=0, A_bc=0, B_br=0, B_bc=j_pair (implicit in loop)
            (0, 1, 1), # A_br=0, A_bc=1, B_br=1, B_bc=j_pair
            (1, 0, 0), # A_br=1, A_bc=0, B_br=0, B_bc=j_pair
            (1, 1, 1), # A_br=1, A_bc=1, B_br=1, B_bc=j_pair
        ]
        
        for A_br, A_bc, B_br in sub_mults:
            # B_bc is j_pair
            B_bc = j_pair
            
            for u_vec, v_vec, w_vec in strassen_terms:
                # u_vec indices: 0..3 correspond to local i', j' in A block
                # local i' = idx // 2, local j' = idx % 2
                # global I = 2*A_br + i'
                # global J = 2*A_bc + j'
                
                # v_vec indices: 0..3 correspond to local j'', k'' in B block?
                # B is m x p. Block is 2x2.
                # local j'' is row in B block, local k'' is col in B block.
                # global J = 2*B_br + j''
                # global K = 2*B_bc + k''
                
                # w_vec indices: 0..3 correspond to local i'', k'' in C block
                # C is n x p. Block is 2x2.
                # local i'' is row in C block, local k'' is col in C block.
                # global I = 2*A_br + i'' (same as A row)
                # global K = 2*B_bc + k'' (same as B col)
                
                for idx in range(4):
                    val_u = u_vec[idx]
                    if val_u != 0:
                        li = idx // 2
                        lj = idx % 2
                        I = 2 * A_br + li
                        J = 2 * A_bc + lj
                        U[I * m + J][col_idx] += val_u
                        
                    val_v = v_vec[idx]
                    if val_v != 0:
                        lj2 = idx // 2
                        lk = idx % 2
                        J = 2 * B_br + lj2
                        K = 2 * B_bc + lk
                        V[J * p + K][col_idx] += val_v
                        
                    val_w = w_vec[idx]
                    if val_w != 0:
                        li2 = idx // 2
                        lk2 = idx % 2
                        I = 2 * A_br + li2
                        K = 2 * B_bc + lk2
                        W[K * n + I][col_idx] += val_w
                            
                col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
