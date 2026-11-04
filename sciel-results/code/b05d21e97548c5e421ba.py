import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    def _build_4x4x4_rank56():
        """Build rank-56 decomposition for 4x4x4 matrix multiplication."""
        # A: 4x4, B: 4x4, C: 4x4
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # Split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where Axy is 2x2
        # B = [[B00, B01], [B10, B11]] where Bxy is 2x2
        # C = [[C00, C01], [C10, C11]] where Cxy is 2x2
        #
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B00+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U_terms = []
        V_terms = []
        W_terms = []
        
        # Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict from (bi,bj) to coeff
        # b_coeffs: dict from (bj,bk) to coeff
        # c_coeffs: dict from (ci,ck) to coeff (for reconstruction)
        
        strassen = [
            # (a_coeffs, b_coeffs)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}),            # M2
            ({(0,0):1}, {(1,0):1, (0,0):-1}),           # M3
            ({(1,1):1}, {(1,0):1, (0,0):-1}),           # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}),            # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # M6
            ({(0,1):1, (1,1):-1}, {(0,0):1, (1,1):1}),  # M7
        ]
        
        c_recon = [
            # (ci, ck, [(s, coeff), ...])
            (0,0, [(0,1), (3,1), (4,-1), (6,1)]),
            (0,1, [(2,1), (4,1)]),
            (1,0, [(1,1), (3,1)]),
            (1,1, [(0,1), (1,-1), (2,1), (5,1)]),
        ]
        
        for s, (a_coeff, b_coeff) in enumerate(strassen):
            # For each local (i,j,k) in 2x2x2 schoolbook
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # U: sum over (bi,bj) in a_coeff of a_coeff[(bi,bj)] * e_{(2*bi+i, 2*bj+j)}
                        u_vec = np.zeros(16)
                        for (bi, bj), c in a_coeff.items():
                            u_vec[(2*bi+i)*4 + (2*bj+j)] = c
                        
                        # V: sum over (bj,bk) in b_coeff of b_coeff[(bj,bk)] * e_{(2*bj+j, 2*bk+k)}
                        v_vec = np.zeros(16)
                        for (bj, bk), c in b_coeff.items():
                            v_vec[(2*bj+j)*4 + (2*bk+k)] = c
                        
                        # W: for each (ci,ck) in c_recon where s appears with coeff c_recon_val
                        # W = sum over (ci,ck) of c_recon_val * e_{(2*ci+i, 2*ck+k)}
                        w_vec = np.zeros(16)
                        for (ci, ck, s_list) in c_recon:
                            for (s_idx, c_val) in s_list:
                                if s_idx == s:
                                    w_vec[(2*ci+i)*4 + (2*ck+k)] = c_val
                        
                        U_terms.append(u_vec)
                        V_terms.append(v_vec)
                        W_terms.append(w_vec)
        
        U = np.column_stack(U_terms)
        V = np.column_stack(V_terms)
        W = np.column_stack(W_terms)
        return U, V, W
    
    def _build_4x4x1_rank16():
        """Build rank-16 decomposition for 4x4x1 matrix multiplication."""
        # A: 4x4, B: 4x1, C: 4x1
        # C[i,0] = sum_j A[i,j] * B[j,0]
        U = np.zeros((16, 16))
        V = np.zeros((4, 16))
        W = np.zeros((4, 16))
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j, col] = 1
                V[j*1+0, col] = 1
                W[0*4+i, col] = 1
                col += 1
        return U, V, W
    
    # Build the full 4x4x9 decomposition
    # First 4 columns: 4x4x4
    # Next 4 columns: 4x4x4
    # Last 1 column: 4x4x1
    
    U1, V1, W1 = _build_4x4x4_rank56()
    U2, V2, W2 = _build_4x4x4_rank56()
    U3, V3, W3 = _build_4x4x1_rank16()
    
    # For the second 4x4x4, we need to shift the B and W indices
    # B is 4x9, so the second 4x4x4 uses columns 4-7 of B
    # V2 should be shifted: V2_orig[j,k] -> V2_new[j, k+4] for k in 0..3
    V2_shifted = np.zeros((4*9, 56))
    for j in range(4):
        for k in range(4):
            V2_shifted[j*9 + (k+4), :] = V2[j*4 + k, :]
    
    # W2 should be shifted: W2_orig[k,i] -> W2_new[k+4, i] for k in 0..3
    W2_shifted = np.zeros((9*4, 56))
    for k in range(4):
        for i in range(4):
            W2_shifted[(k+4)*4 + i, :] = W2[k*4 + i, :]
    
    # For 4x4x1, B uses column 8, W uses column 8
    V3_shifted = np.zeros((4*9, 16))
    for j in range(4):
        V3_shifted[j*9 + 8, :] = V3[j, :]
    
    W3_shifted = np.zeros((9*4, 16))
    for i in range(4):
        W3_shifted[8*4 + i, :] = W3[i, :]
    
    U_full = np.hstack([U1, U2, U3])
    V_full = np.hstack([V1, V2_shifted, V3_shifted])
    W_full = np.hstack([W1, W2_shifted, W3_shifted])
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
