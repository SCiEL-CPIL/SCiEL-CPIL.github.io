"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 20 decomposition
    # 7 for A2B2 (2x2x6)
    # 7 for A1aB1a (2x2x6)
    # 6 for A1bB1b (2x1x6)
    rank = 20
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Helper to add a term
    # term_idx: column index
    # u_coeffs: dict {(i, j): val}
    # v_coeffs: dict {(j, k): val} -- Note: V depends on k in the formula?
    # Wait, V[j*p+k, r] is the coefficient.
    # So for a fixed r, V[j*p+k, r] is fixed for all j,k.
    # This means L_B(r, k) = sum_j V[j*p+k, r] B[j,k].
    # So V entries are constants.
    # w_coeffs: dict {(k, i): val}
    
    def add_term(idx, u_dict, v_dict, w_dict):
        for (i, j), val in u_dict.items():
            U[i * m + j][idx] += val
        for (j, k), val in v_dict.items():
            V[j * p + k][idx] += val
        for (k, i), val in w_dict.items():
            W[k * n + i][idx] += val

    # Part 1: A2 B2 (2x2x6)
    # A2 is cols 3,4 of A. B2 is rows 3,4 of B.
    # A2 = [[A03, A04], [A13, A14]]
    # B2 = [[B30..B35], [B40..B45]]
    # Strassen 7 products
    # M1: (A03+A14) * (B3+B4) -> L_A = A03+A14, L_B_k = B3k+B4k
    # M2: (A13+A14) * B3 -> L_A = A13+A14, L_B_k = B3k
    # M3: A03 * (B4-B3) -> L_A = A03, L_B_k = B4k-B3k
    # M4: A14 * (B4-B3)? No, M4 = A22(B22-B12) = A14(B4-B3)
    # M5: (A03+A04) * B4 -> L_A = A03+A04, L_B_k = B4k
    # M6: (A13-A03) * (B3+B4) -> L_A = A13-A03, L_B_k = B3k+B4k
    # M7: (A04-A14) * (B4+B3)? M7 = (A12-A22)(B21+B22) = (A04-A14)(B4+B3)
    
    # W coefficients for Strassen 2x2x2 (standard):
    # C00 = M1 - M3 - M4 + M7
    # C01 = M5 + M7 - M1 - M2
    # C10 = M2 + M4 - M6 - M7
    # C11 = M1 - M2 + M3 + M6
    
    # Apply to 2x2x6 (each C is 1x6)
    # C0k = M1 - M3 - M4 + M7
    # C1k = M5 + M7 - M1 - M2
    # Wait, standard Strassen outputs:
    # C11 = M1 - M2 + M3 + M6
    # C12 = M5 + M7
    # C21 = M4 + M6
    # C22 = M1 - M3 + M4 + M7
    # Let's stick to standard:
    # C00 = M1 - M3 - M4 + M7
    # C01 = M5 + M7 - M1 - M2
    # C10 = M2 + M4 - M6 - M7
    # C11 = M1 - M2 + M3 + M6

    # Term 0: M1
    add_term(0, {(0,3):1, (1,4):1}, {(3,k):1 for k in range(p)}, {(k,0):1 for k in range(p)}, {(k,1):-1 for k in range(p)}) 
    # Wait, W dict keys are (k, i).
    # C00 (i=0, k) gets +1 from M1.
    # C01 (i=0, k) gets -1 from M1.
    # C10 (i=1, k) gets -1 from M1.
    # C11 (i=1, k) gets +1 from M1.
    # My add_term call above was messy. Let's do it properly.

    # Reset U,V,W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def set_term(idx, u_list, v_list, w_list):
        for (i, j), val in u_list.items():
            U[i * m + j][idx] = val
        for (j, k), val in v_list.items():
            V[j * p + k][idx] = val
        for (k, i), val in w_list.items():
            W[k * n + i][idx] = val

    # M1: LA=A03+A14, LB=B3+B4
    set_term(0, {(0,3):1, (1,4):1}, {(3,k):1 for k in range(p)}, {(4,k):1 for k in range(p)}, 
             {(k,0):1 for k in range(p)}, {(k,1):-1 for k in range(p)}) # C11? No.
    # Let's map W correctly.
    # C00 (i=0) += M1
    # C01 (i=0) -= M1
    # C10 (i=1) -= M1
    # C11 (i=1) += M1
    # So W[k,0] += 1, W[k,1] -= 1?
    # W index is k*n+i.
    # For i=0, W[k*2+0] += 1.
    # For i=1, W[k*2+1] -= 1.
    
    # This is getting complex to hardcode. I will generate the dicts.

    # M1
    u = {(0,3):1, (1,4):1}
    v = {(3,k):1 for k in range(p)}
    v.update({(4,k):1 for k in range(p)})
    w = {(k,0):1 for k in range(p)}
    w.update({(k,1):-1 for k in range(p)})
    set_term(0, u, v, w)

    # M2: LA=A13+A14, LB=B3
    u = {(1,3):1, (1,4):1}
    v = {(3,k):1 for k in range(p)}
    w = {(k,0):-1 for k in range(p)} # C01 -= M2
    w.update({(k,1):1 for k in range(p)}) # C10 += M2
    set_term(1, u, v, w)

    # M3: LA=A03, LB=B4-B3
    u = {(0,3):1}
    v = {(4,k):1 for k in range(p)}
    v.update({(3,k):-1 for k in range(p)})
    w = {(k,0):-1 for k in range(p)} # C00 -= M3
    w.update({(k,1):1 for k in range(p)}) # C11 += M3
    set_term(2, u, v, w)

    # M4: LA=A14, LB=B4-B3
    u = {(1,4):1}
    v = {(4,k):1 for k in range(p)}
    v.update({(3,k):-1 for k in range(p)})
    w = {(k,0):-1 for k in range(p)} # C00 -= M4
    w.update({(k,1):1 for k in range(p)}) # C10 += M4
    set_term(3, u, v, w)

    # M5: LA=A03+A04, LB=B4
    u = {(0,3):1, (0,4):1}
    v = {(4,k):1 for k in range(p)}
    w = {(k,0):1 for k in range(p)} # C01 += M5
    w.update({(k,1):0 for k in range(p)}) # C11? No, M5 is only in C01 and C11?
    # C01 = M5 + M7 - M1 - M2
    # C11 = M1 - M2 + M3 + M6
    # So M5 is only in C01.
    # W[k,0] += 1 (i=0 is C0? No, i=0 is row 0).
    # C01 is i=0, col 1? No, C_{i,k}. i is row.
    # C00 is i=0, k=0..5.
    # C01 is i=0, k=0..5? No, C is 2x6.
    # C_{0,k} is row 0. C_{1,k} is row 1.
    # Strassen outputs C00, C01, C10, C11 refer to 2x2 blocks.
    # Here "C01" means the block in row 0, col 1 of the 2x2 block structure?
    # No, B is 2x6. So there is no "col 1" block.
    # The 4 outputs are:
    # C_{0, :} (row 0 of result)
    # C_{1, :} (row 1 of result)
    # Wait, Strassen 2x2x2 produces 4 scalars.
    # Strassen 2x2x6 produces 4 vectors of length 6.
    # The formulas are applied element-wise for each k.
    # C_{0,k} = M1 - M3 - M4 + M7
    # C_{1,k} = M5 + M7 - M1 - M2
    # Wait, standard Strassen:
    # C11 = M1 - M2 + M3 + M6
    # C12 = M5 + M7
    # C21 = M4 + M6
    # C22 = M1 - M3 + M4 + M7
    # Mapping to 2x6:
    # Row 0 of C corresponds to C11 and C12?
    # If B is 2x6, we can view it as two 2x3 blocks? Or just apply the formulas to each column k.
    # The formulas for C_{i,k} are:
    # C_{0,k} = M1 - M3 - M4 + M7  (This is C11? No, C11 is top-left)
    # Let's assume standard indexing:
    # C_{0,k} uses M1, M3, M4, M7.
    # C_{1,k} uses M5, M7, M1, M2.
    # Let's verify:
    # C_{0,k} = A_{00}B_{0k} + A_{01}B_{1k}.
    # M1 = (A00+A11)(B0+B1)
    # M3 = A00(B1-B0)
    # M4 = A11(B1-B0)
    # M7 = (A01-A11)(B1+B0)
    # Sum = A00B0 + A00B1 + A11B0 + A11B1 - A00B1 + A00B0 - A11B1 + A11B0 + A01B1 + A01B0 - A11B1 - A11B0
    # = 2 A00B0 + A11B0 + A01B1 + A01B0 - A11B1. Not quite.
    
    # I will use a known correct set of coefficients for 2x2x2 and apply them to 2x2x6.
    # Coeffs for M1..M7 in C00, C01, C10, C11:
    # C00: 1, 0, -1, -1, 0, 0, 1
    # C01: -1, -1, 0, 0, 1, 0, 1
    # C10: 0, 1, 0, 1, 0, -1, -1
    # C11: 1, -1, 1, 0, 0, 1, 0

    # For 2x2x6, C00 is C_{0,:}, C01 is C_{0,:}? No.
    # C00 is top-left 2x2 block.
    # In 2x2x6, the "top-left" is Row 0.
    # So C_{0,k} corresponds to C00 and C01 combined?
    # No, Strassen computes 4 independent 2x2 blocks.
    # Here we have only 2 rows.
    # So we only need the formulas for Row 0 and Row 1.
    # Row 0: C_{0,k} = C_{00,k} + C_{01,k}?
    # If B is 2x6, we can split B into B_left (2x3) and B_right (2x3).
    # Then C = A B_left + A B_right.
    # This doubles the rank.
    # Better: Apply the 2x2x2 formulas to each column k.
    # C_{0,k} = 1*M1 - 1*M3 - 1*M4 + 1*M7
    # C_{1,k} = -1*M1 - 1*M2 + 1*M5 + 1*M7
    
    # Let's re-verify C1k formula:
    # C11 = M1 - M2 + M3 + M6
    # C12 = M5 + M7
    # If we just take Row 1 of the 2x2 result:
    # C_{1,k} = C_{11,k} + C_{12,k}? No.
    # The 2x2 result is [[C00, C01], [C10, C11]].
    # If B is 2x6, we can't map it to 2x2 directly without splitting.
    # However, we can treat B as 2x6 and use the linear forms.
    # The linear forms for M1..M7 are the same.
    # The output formulas are linear combinations of M1..M7.
    # For each k, C_{0,k} and C_{1,k} are computed.
    # C_{0,k} = M1 - M3 - M4 + M7
    # C_{1,k} = M5 + M7 - M1 - M2
    
    # So W coefficients:
    # For i=0 (C0): M1:1, M2:0, M3:-1, M4:-1, M5:0, M6:0, M7:1
    # For i=1 (C1): M1:-1, M2:-1, M3:0, M4:0, M5:1, M6:0, M7:1

    # I will implement this logic in the code.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
