"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 48 for 4x4x3
    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 using Rank 49 (Strassen of Strassen)
    # Strassen 2x2x2 coefficients
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
    
    # Outer Strassen Steps
    # Each step l has U_l (2x2) and V_l (2x2)
    # U_l entries are linear combos of A blocks
    # V_l entries are linear combos of B blocks
    
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We represent linear combos as dict: {global_index: coeff}
    # A global index: i * 4 + j
    # B global index: j * 7 + k (for first 4 cols, k=0..3)
    
    def add_to_vec(vec, idx, val):
        if val != 0:
            vec[idx] += val

    # Helper to get A block index
    def a_idx(i, j): return i * 4 + j
    # Helper to get B block index (first 4 cols)
    def b_idx(j, k): return j * 7 + k
    # Helper to get C block index (first 4 cols)
    def c_idx(k, i): return k * 4 + i

    # Define Outer Strassen Steps
    # Each step is a tuple (U_coeffs, V_coeffs, C_weights)
    # U_coeffs: dict mapping (r,s) to {a_global: coeff}
    # V_coeffs: dict mapping (s,t) to {b_global: coeff}
    # C_weights: dict mapping (I,K) to coeff, where I,K are block indices of C
    # Actually, C_weights should map to the specific M_l contribution.
    # Let's just iterate.
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # U1 = A00 + A11
    # V1 = B00 + B11
    # C00 += M1, C11 += M1
    
    # We will implement a generic inner Strassen function that takes
    # U_mat (2x2 of linear combos) and V_mat (2x2 of linear combos)
    # and adds the rank-7 decomposition to U, V, W.
    
    def inner_strassen(U_mat, V_mat, col_offset, W_weights):
        # U_mat: 2x2 list of dicts {a_idx: coeff}
        # V_mat: 2x2 list of dicts {b_idx: coeff}
        # W_weights: dict mapping (r, t) to coeff, for the contribution to M[r,t]
        # Returns new col_offset
        
        # Inner Strassen 7 steps
        # m1 = (U00+U11)(V00+V11)
        # m2 = (U10+U11)V00
        # m3 = U00(V01-V11)
        # m4 = U11(V10-V00)
        # m5 = (U00+U01)V11
        # m6 = (U10-U00)(V00+V01)
        # m7 = (U01-U11)(V10+V11)
        
        # Reconstruction of M from m1..m7
        # M00 = m1 + m4 - m5 + m7
        # M01 = m3 + m5
        # M10 = m2 + m4
        # M11 = m1 - m2 + m3 + m6
        
        # For each inner step, we form P (linear combo of U) and Q (linear combo of V)
        # P and Q are scalars (linear combos of A and B entries)
        # The product P*Q contributes to specific M entries with specific weights.
        
        # We need to expand P and Q into global A and B indices.
        
        def merge_dicts(d1, d2, sign1=1, sign2=1):
            res = {}
            for k, v in d1.items():
                res[k] = res.get(k, 0) + v * sign1
            for k, v in d2.items():
                res[k] = res.get(k, 0) + v * sign2
            return res

        steps = [
            (merge_dicts(U_mat[0][0], U_mat[1][1]), merge_dicts(V_mat[0][0], V_mat[1][1]), {(0,0):1, (1,1):1}),
            (merge_dicts(U_mat[1][0], U_mat[1][1]), V_mat[0][0], {(1,0):1}),
            (U_mat[0][0], merge_dicts(V_mat[0][1], V_mat[1][1], 1, -1), {(0,1):1}),
            (U_mat[1][1], merge_dicts(V_mat[1][0], V_mat[0][0], 1, -1), {(0,0):1, (1,0):1}),
            (merge_dicts(U_mat[0][0], U_mat[0][1]), V_mat[1][1], {(0,0):-1, (0,1):1}),
            (merge_dicts(U_mat[1][0], U_mat[0][0], 1, -1), merge_dicts(V_mat[0][0], V_mat[0][1]), {(1,1):1}),
            (merge_dicts(U_mat[0][1], U_mat[1][1], 1, -1), merge_dicts(V_mat[1][0], V_mat[1][1]), {(0,0):1, (1,1):-1})
        ]
        
        # Wait, the reconstruction weights for M entries from inner steps are:
        # M00: m1(1), m4(1), m5(-1), m7(1)
        # M01: m3(1), m5(1)
        # M10: m2(1), m4(1)
        # M11: m1(1), m2(-1), m3(1), m6(1)
        
        # My `steps` list above has the P, Q and the weights for M entries?
        # No, the 3rd element in my steps list is the weight of THIS step's product in the M entries.
        # Let's check step 1 (m1):
        # m1 contributes to M00 with 1, M11 with 1. Correct.
        # Step 2 (m2):
        # m2 contributes to M10 with 1, M11 with -1.
        # My list has {(1,0):1}. Missing M11.
        # Step 3 (m3):
        # m3 contributes to M01 with 1, M11 with 1.
        # My list has {(0,1):1}. Missing M11.
        # Step 4 (m4):
        # m4 contributes to M00 with 1, M10 with 1.
        # My list has {(0,0):1, (1,0):1}. Correct.
        # Step 5 (m5):
        # m5 contributes to M00 with -1, M01 with 1.
        # My list has {(0,0):-1, (0,1):1}. Correct.
        # Step 6 (m6):
        # m6 contributes to M11 with 1.
        # My list has {(1,1):1}. Correct.
        # Step 7 (m7):
        # m7 contributes to M00 with 1.
        # My list has {(0,0):1, (1,1):-1}. Wait.
        # M11 = m1 - m2 + m3 + m6. m7 is not in M11.
        # M00 = m1 + m4 - m5 + m7.
        # So m7 contributes to M00 with 1.
        # My list has {(0,0):1, (1,1):-1}. This is wrong.
        
        # Let's correct the weights for each inner step.
        # m1: M00:1, M11:1
        # m2: M10:1, M11:-1
        # m3: M01:1, M11:1
        # m4: M00:1, M10:1
        # m5: M00:-1, M01:1
        # m6: M11:1
        # m7: M00:1
        
        corrected_steps = [
            (merge_dicts(U_mat[0][0], U_mat[1][1]), merge_dicts(V_mat[0][0], V_mat[1][1]), {(0,0):1, (1,1):1}),
            (merge_dicts(U_mat[1][0], U_mat[1][1]), V_mat[0][0], {(1,0):1, (1,1):-1}),
            (U_mat[0][0], merge_dicts(V_mat[0][1], V_mat[1][1], 1, -1), {(0,1):1, (1,1):1}),
            (U_mat[1][1], merge_dicts(V_mat[1][0], V_mat[0][0], 1, -1), {(0,0):1, (1,0):1}),
            (merge_dicts(U_mat[0][0], U_mat[0][1]), V_mat[1][1], {(0,0):-1, (0,1):1}),
            (merge_dicts(U_mat[1][0], U_mat[0][0], 1, -1), merge_dicts(V_mat[0][0], V_mat[0][1]), {(1,1):1}),
            (merge_dicts(U_mat[0][1], U_mat[1][1], 1, -1), merge_dicts(V_mat[1][0], V_mat[1][1]), {(0,0):1})
        ]
        
        for P, Q, M_weights in corrected_steps:
            # P is dict {a_idx: coeff}
            # Q is dict {b_idx: coeff}
            # This is a rank-1 term: sum_a P[a] A[a] * sum_b Q[b] B[b]
            # It contributes to W at positions determined by M_weights and W_weights.
            
            # For each a in P, each b in Q:
            # U[a_idx] += P[a]
            # V[b_idx] += Q[b]
            # This is not quite right. The term is (linear combo A) * (linear combo B).
            # In the tensor decomposition, each scalar multiplication is one column.
            # So for this inner step, we have ONE column.
            # U column has non-zeros at keys of P.
            # V column has non-zeros at keys of Q.
            # W column has non-zeros at keys determined by M_weights and W_weights.
            
            # Add to U
            for a_idx, coeff in P.items():
                U[a_idx][col_offset] = coeff
            
            # Add to V
            for b_idx, coeff in Q.items():
                V[b_idx][col_offset] = coeff
            
            # Add to W
            # M_weights maps (r, t) to weight of this inner step in M[r,t]
            # W_weights maps (I, K) to weight of M[r,t] in C[I,K][r,t]?
            # No, W_weights should map (r,t) to the global C indices?
            # Let's pass a function or a dict of (r,t) -> (I,K, weight_outer)
            
            for (r, t), w_inner in M_weights.items():
                if (r, t) in W_weights:
                    I, K, w_outer = W_weights[(r, t)]
                    # Global C index: i = I*2 + r, k = K*2 + t
                    global_c_idx = c_idx(K*2 + t, I*2 + r)
                    W[global_c_idx][col_offset] += w_inner * w_outer
            
            col_offset += 1
            
        return col_offset

    # Now define the 7 Outer Strassen Steps
    # U_mat and V_mat are 2x2 lists of dicts
    
    def make_block_dict(coeffs):
        # coeffs: list of 4 tuples (block_idx, coeff)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        # Returns 2x2 list of dicts
        res = [[{} for _ in range(2)] for _ in range(2)]
        for b_idx, coeff in coeffs:
            r = b_idx // 2
            s = b_idx % 2
            # Expand block to global indices
            for i in range(2):
                for j in range(2):
                    a_global = a_idx(r*2 + i, s*2 + j)
                    res[r][s][a_global] = coeff
        return res

    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    def make_block_dict_B(coeffs):
        res = [[{} for _ in range(2)] for _ in range(2)]
        for b_idx, coeff in coeffs:
            s = b_idx // 2 # Row of B block (which is col of A block)
            t = b_idx % 2
            for i in range(2):
                for j in range(2):
                    b_global = b_idx(s*2 + i, t*2 + j) # Wait, b_idx function is j*7+k
                    # B block (s,t) has rows s*2..s*2+1, cols t*2..t*2+1
                    # Global index j*7 + k
                    for r_local in range(2):
                        for c_local in range(2):
                            j_global = s*2 + r_local
                            k_global = t*2 + c_local
                            b_global = b_idx(j_global, k_global)
                            res[s][t][b_global] = coeff
        return res

    # Outer Step 1: M1 = (A00+A11)(B00+B11)
    U1 = make_block_dict([(0,1), (3,1)])
    V1 = make_block_dict_B([(0,1), (3,1)])
    # C00 += M1, C11 += M1
    # M1[r,t] contributes to C00[r,t] and C11[r,t]
    W_weights_1 = {
        (0,0): (0,0,1), (0,1): (0,0,1), (1,0): (0,0,1), (1,1): (0,0,1),
        (0,0): (1,1,1), (0,1): (1,1,1), (1,0): (1,1,1), (1,1): (1,1,1)
    }
    # Wait, W_weights maps (r,t) to (I,K, weight).
    # But M1 is a 2x2 matrix. M1[r,t] is added to C00[r,t] and C11[r,t].
    # So for each (r,t), we have two contributions.
    # My inner_strassen function only allows one W_weights entry per (r,t).
    # I need to modify inner_strassen to handle multiple W contributions per (r,t).
    
    # Let's refactor: W_weights should be a list of tuples (I, K, weight) for each (r,t).
    # Or just a dict (r,t) -> list of (I,K,weight).
    
    # I will update the inner_strassen call to pass a list.
    
    col = inner_strassen(U1, V1, col, {
        (0,0): [(0,0,1), (1,1,1)],
        (0,1): [(0,0,1), (1,1,1)],
        (1,0): [(0,0,1), (1,1,1)],
        (1,1): [(0,0,1), (1,1,1)]
    })
    
    # Outer Step 2: M2 = (A10+A11)B00
    U2 = make_block_dict([(2,1), (3,1)])
    V2 = make_block_dict_B([(0,1)])
    # C10 += M2, C11 -= M2
    col = inner_strassen(U2, V2, col, {
        (0,0): [(1,0,1), (1,1,-1)],
        (0,1): [(1,0,1), (1,1,-1)],
        (1,0): [(1,0,1), (1,1,-1)],
        (1,1): [(1,0,1), (1,1,-1)]
    })
    
    # Outer Step 3: M3 = A00(B01-B11)
    U3 = make_block_dict([(0,1)])
    V3 = make_block_dict_B([(1,1), (3,-1)])
    # C01 += M3, C11 += M3
    col = inner_strassen(U3, V3, col, {
        (0,0): [(0,1,1), (1,1,1)],
        (0,1): [(0,1,1), (1,1,1)],
        (1,0): [(0,1,1), (1,1,1)],
        (1,1): [(0,1,1), (1,1,1)]
    })
    
    # Outer Step 4: M4 = A11(B10-B00)
    U4 = make_block_dict([(3,1)])
    V4 = make_block_dict_B([(2,1), (0,-1)])
    # C00 += M4, C10 += M4
    col = inner_strassen(U4, V4, col, {
        (0,0): [(0,0,1), (1,0,1)],
        (0,1): [(0,0,1), (1,0,1)],
        (1,0): [(0,0,1), (1,0,1)],
        (1,1): [(0,0,1), (1,0,1)]
    })
    
    # Outer Step 5: M5 = (A00+A01)B11
    U5 = make_block_dict([(0,1), (1,1)])
    V5 = make_block_dict_B([(3,1)])
    # C00 -= M5, C01 += M5
    col = inner_strassen(U5, V5, col, {
        (0,0): [(0,0,-1), (0,1,1)],
        (0,1): [(0,0,-1), (0,1,1)],
        (1,0): [(0,0,-1), (0,1,1)],
        (1,1): [(0,0,-1), (0,1,1)]
    })
    
    # Outer Step 6: M6 = (A10-A00)(B00+B01)
    U6 = make_block_dict([(2,1), (0,-1)])
    V6 = make_block_dict_B([(0,1), (1,1)])
    # C11 += M6
    col = inner_strassen(U6, V6, col, {
        (0,0): [(1,1,1)],
        (0,1): [(1,1,1)],
        (1,0): [(1,1,1)],
        (1,1): [(1,1,1)]
    })
    
    # Outer Step 7: M7 = (A01-A11)(B10+B11)
    U7 = make_block_dict([(1,1), (3,-1)])
    V7 = make_block_dict_B([(2,1), (3,1)])
    # C00 += M7
    col = inner_strassen(U7, V7, col, {
        (0,0): [(0,0,1)],
        (0,1): [(0,0,1)],
        (1,0): [(0,0,1)],
        (1,1): [(0,0,1)]
    })
    
    # Part 2: 4x4x3 Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 7 + (4 + k)][col] = 1
                W[(4 + k) * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
