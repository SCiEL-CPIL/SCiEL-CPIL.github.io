import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 98 decomposition
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # U_in[t][i, j], V_in[t][j, k], W_in[t][k, i]
    # t=0: (A00+A11)(B00+B11)
    # t=1: (A10+A11)B00
    # t=2: A00(B01-B11)
    # t=3: A11(B10-B00)
    # t=4: (A00+A01)B11
    # t=5: (A10-A00)(B00+B01)
    # t=6: (A01-A11)(B10+B11)
    
    U_in = [
        [[1, 0], [0, 1]], # A00+A11 -> [[1,0],[0,1]]? No.
        # A00 is [[A00_00, A00_01], [A00_10, A00_11]]
        # A11 is [[A11_00, A11_01], [A11_10, A11_11]]
        # Sum is element-wise.
        # So U_in[0] = [[1, 1], [1, 1]]? No.
        # The linear form is x = A00 + A11.
        # x[i, j] = A00[i, j] + A11[i, j].
        # So coefficients are 1 for (0,0) and 1 for (1,1) in block indices?
        # No, U_in is for the inner 2x2 matrix multiplication.
        # The "entries" of the inner matrix are scalars.
        # So U_in[t] is a 2x2 matrix of coefficients for the elements of X.
        # X is a 2x2 matrix.
        # X[0,0], X[0,1], X[1,0], X[1,1].
        # For t=0: X = A00 + A11.
        # This X is the "A" matrix of the inner multiplication.
        # But wait, in the nested approach, X_s is the A-matrix for the inner multiplication.
        # X_s is a 2x2 matrix.
        # The inner Strassen decomposition of P = X Y uses X as the first factor.
        # So U_in[t] gives the linear combination of X's elements.
        # For t=0: X[0,0] + X[1,1].
        # So U_in[0] = [[1, 0], [0, 1]].
        # For t=1: X[1,0] + X[1,1].
        # So U_in[1] = [[0, 0], [1, 1]].
        # And so on.
        
        # Let's list them properly.
        # t=0: X00 + X11 -> [[1, 0], [0, 1]]
        # t=1: X10 + X11 -> [[0, 0], [1, 1]]
        # t=2: X00 -> [[1, 0], [0, 0]]
        # t=3: X11 -> [[0, 0], [0, 1]]
        # t=4: X00 + X01 -> [[1, 1], [0, 0]]
        # t=5: X10 - X00 -> [[-1, 0], [1, 0]]
        # t=6: X01 - X11 -> [[0, 1], [0, -1]]
    ]
    
    U_in = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    
    # V_in[t] for Y
    # t=0: Y00 + Y11 -> [[1, 0], [0, 1]]
    # t=1: Y00 -> [[1, 0], [0, 0]]
    # t=2: Y01 - Y11 -> [[0, 1], [0, -1]]
    # t=3: Y10 - Y00 -> [[-1, 0], [1, 0]]
    # t=4: Y11 -> [[0, 0], [0, 1]]
    # t=5: Y00 + Y01 -> [[1, 1], [0, 0]]
    # t=6: Y10 + Y11 -> [[0, 1], [1, 1]]
    
    V_in = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 1], [1, 1]])
    ]
    
    # W_in[t] for P
    # P00 = M0 + M3 - M4 + M6
    # P01 = M2 + M4
    # P10 = M1 + M3
    # P11 = M0 - M1 + M2 + M5
    
    # W_in[t][k, i] is coeff of P[k, i] in term t?
    # No, W_in[t] is the linear form for the output P.
    # For term t, the output is M_t.
    # M_t contributes to P with some coefficients.
    # But in the decomposition, we sum over t.
    # P = sum_t W_t M_t.
    # So W_in[t] should be the matrix such that M_t * W_in[t] gives the contribution to P?
    # No, the tensor decomposition is sum_r u_r v_r w_r.
    # For each term t, we have u_t, v_t, w_t.
    # w_t is the coefficient for P.
    # So W_in[t] is a 2x2 matrix where W_in[t][k, i] is the coefficient of P[k, i] in the expansion?
    # No, the term is u_t * v_t * w_t.
    # This term adds to the tensor T[i, j, k].
    # The contribution to P[i, k] is u_t[i, j] v_t[j, k] w_t[k, i]?
    # The identity is sum_t U_t[i, j] V_t[j, k] W_t[k, i] = 1.
    # So for the inner 2x2x2 multiplication, we need:
    # sum_t U_in[t][i, j] V_in[t][j, k] W_in[t][k, i] = 1.
    
    # Let's check t=0.
    # U0 = [[1,0],[0,1]], V0 = [[1,0],[0,1]], W0 = ?
    # M0 = (X00+X11)(Y00+Y11).
    # P00 gets +M0. P11 gets +M0.
    # So W0[0, 0] = 1, W0[1, 1] = 1. Others 0.
    # W0 = [[1, 0], [0, 1]].
    
    # t=1: M1 = (X10+X11)Y00.
    # P10 gets +M1. P11 gets -M1.
    # W1[0, 1] = 1? No, P10 is [1, 0]. So W1[0, 1] = 1.
    # P11 is [1, 1]. So W1[1, 1] = -1.
    # W1 = [[0, 1], [0, -1]].
    
    # t=2: M2 = X00(Y01-Y11).
    # P01 gets +M2. P11 gets +M2.
    # W2[1, 0] = 1, W2[1, 1] = 1.
    # W2 = [[0, 0], [1, 1]].
    
    # t=3: M3 = X11(Y10-Y00).
    # P00 gets +M3. P10 gets +M3.
    # W3[0, 0] = 1, W3[0, 1] = 1.
    # W3 = [[1, 1], [0, 0]].
    
    # t=4: M4 = (X00+X01)Y11.
    # P00 gets -M4. P01 gets +M4.
    # W4[0, 0] = -1, W4[1, 0] = 1.
    # W4 = [[-1, 0], [1, 0]].
    
    # t=5: M5 = (X10-X00)(Y00+Y01).
    # P11 gets +M5.
    # W5[1, 1] = 1.
    # W5 = [[0, 0], [0, 1]].
    
    # t=6: M6 = (X01-X11)(Y10+Y11).
    # P00 gets +M6.
    # W6[0, 0] = 1.
    # W6 = [[1, 0], [0, 0]].
    
    W_in = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    # Verify inner decomposition
    # for i in range(2):
    #   for j in range(2):
    #     for k in range(2):
    #       s = 0
    #       for t in range(7):
    #         s += U_in[t][i, j] * V_in[t][j, k] * W_in[t][k, i]
    #       if s != 1: print("Error")
    
    # Outer Strassen Coefficients
    # C_s[u, v] coeff for A_u,v
    # D_s[v, w] coeff for B_v,w
    # E_s[u, w] coeff for C_u,w
    
    # M0 = (A00+A11)(B00+B11)
    C0 = np.array([[1, 0], [0, 1]])
    D0 = np.array([[1, 0], [0, 1]])
    E0 = np.array([[1, 0], [0, 1]]) # C00 += M0, C11 += M0
    
    # M1 = (A10+A11)B00
    C1 = np.array([[0, 0], [1, 1]])
    D1 = np.array([[1, 0], [0, 0]])
    E1 = np.array([[0, 0], [1, -1]]) # C10 += M1, C11 -= M1
    
    # M2 = A00(B01-B11)
    C2 = np.array([[1, 0], [0, 0]])
    D2 = np.array([[0, 1], [0, -1]])
    E2 = np.array([[0, 1], [0, 1]]) # C01 += M2, C11 += M2
    
    # M3 = A11(B10-B00)
    C3 = np.array([[0, 0], [0, 1]])
    D3 = np.array([[-1, 0], [1, 0]])
    E3 = np.array([[1, 0], [1, 0]]) # C00 += M3, C10 += M3
    
    # M4 = (A00+A01)B11
    C4 = np.array([[1, 1], [0, 0]])
    D4 = np.array([[0, 0], [0, 1]])
    E4 = np.array([[-1, 1], [0, 0]]) # C00 -= M4, C01 += M4
    
    # M5 = (A10-A00)(B00+B01)
    C5 = np.array([[-1, 0], [1, 0]])
    D5 = np.array([[1, 1], [0, 0]])
    E5 = np.array([[0, 0], [0, 1]]) # C11 += M5
    
    # M6 = (A01-A11)(B10+B11)
    C6 = np.array([[0, 1], [0, -1]])
    D6 = np.array([[0, 1], [1, 1]])
    E6 = np.array([[1, 0], [0, 0]]) # C00 += M6
    
    Cs = [C0, C1, C2, C3, C4, C5, C6]
    Ds = [D0, D1, D2, D3, D4, D5, D6]
    Es = [E0, E1, E2, E3, E4, E5, E6]
    
    r_idx = 0
    for s in range(7):
        for part in range(2): # 0: L, 1: R
            for t in range(7):
                # Update U
                for u in range(2):
                    for v in range(2):
                        c_val = Cs[s][u, v]
                        if c_val != 0:
                            for xi in range(2):
                                for xj in range(2):
                                    u_val = U_in[t][xi, xj] * c_val
                                    if u_val != 0:
                                        i_global = 2 * u + xi
                                        j_global = 2 * v + xj
                                        U[i_global * m + j_global][r_idx] += u_val
                                        
                # Update V
                for v in range(2):
                    for w in range(2):
                        d_val = Ds[s][v, w]
                        if d_val != 0:
                            # Check if this block contributes to L or R part
                            # L part uses w=0, k_l in 0,1
                            # R part uses w=1, k_l in 0,1 (local in B_v,1)
                            # Y_s,L uses B_v,0. Y_s,R uses B_v,1.
                            if part == 0 and w == 0:
                                for yj in range(2):
                                    for yk in range(2):
                                        v_val = V_in[t][yj, yk] * d_val
                                        if v_val != 0:
                                            j_global = 2 * v + yj
                                            k_global = 4 * w + yk
                                            V[j_global * p + k_global][r_idx] += v_val
                            elif part == 1 and w == 1:
                                for yj in range(2):
                                    for yk in range(2):
                                        v_val = V_in[t][yj, yk] * d_val
                                        if v_val != 0:
                                            j_global = 2 * v + yj
                                            k_global = 4 * w + yk
                                            V[j_global * p + k_global][r_idx] += v_val
                                            
                # Update W
                for u in range(2):
                    for w in range(2):
                        e_val = Es[s][u, w]
                        if e_val != 0:
                            if part == 0 and w == 0:
                                for pi in range(2):
                                    for pk in range(2):
                                        w_val = W_in[t][pk, pi] * e_val
                                        if w_val != 0:
                                            i_global = 2 * u + pi
                                            k_global = 4 * w + pk
                                            W[k_global * n + i_global][r_idx] += w_val
                            elif part == 1 and w == 1:
                                for pi in range(2):
                                    for pk in range(2):
                                        w_val = W_in[t][pk, pi] * e_val
                                        if w_val != 0:
                                            i_global = 2 * u + pi
                                            k_global = 4 * w + pk
                                            W[k_global * n + i_global][r_idx] += w_val
                                            
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
