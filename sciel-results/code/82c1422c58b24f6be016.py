import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    N = n * m  # 12
    P = m * p  # 16
    Q = p * n  # 12

    def build_44_rank():
        U = np.zeros((N, 44))
        V = np.zeros((P, 44))
        W = np.zeros((Q, 44))
        
        # Local 3x2x2 factors (rank 11)
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10)
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1
        
        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1
        
        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1
        
        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1
        
        # Place blocks
        # A (3x4) split into A_L (3x2) and A_R (3x2)
        # B (4x4) split into 4 2x2 blocks: B11, B12, B21, B22
        # C (3x4) split into C_L (3x2) and C_R (3x2)
        
        # C_L = A_L B11 + A_R B21
        # C_R = A_L B12 + A_R B22
        
        blocks = [
            # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
            (0, 0, 0, 0, 0, 0), # A_L, B11 -> C_L
            (0, 2, 2, 0, 0, 0), # A_R, B21 -> C_L
            (0, 0, 0, 2, 0, 2), # A_L, B12 -> C_R
            (0, 2, 2, 2, 0, 2)  # A_R, B22 -> C_R
        ]
        
        for b_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
            offset = b_idx * 11
            
            # A indices: i (0..2), j (0..1)
            for i in range(3):
                for j in range(2):
                    la = i * 2 + j
                    ga = (ar + i) * 4 + (ac + j)
                    U[ga, offset:offset+11] += U_loc[la, :]
            
            # B indices: i (0..1), j (0..1)
            for i in range(2):
                for j in range(2):
                    lb = i * 2 + j
                    gb = (br + i) * 4 + (bc + j)
                    V[gb, offset:offset+11] += V_loc[lb, :]
            
            # C indices: i (0..2), j (0..1)
            for i in range(3):
                for j in range(2):
                    lc = i * 2 + j
                    gc = (cc + j) * 3 + (cr + i)
                    W[gc, offset:offset+11] += W_loc[lc, :]
        
        return U, V, W

    def build_48_rank():
        U = np.zeros((N, 48))
        V = np.zeros((P, 48))
        W = np.zeros((Q, 48))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return U, V, W

    def verify(U, V, W):
        # Check T = sum_r U_r V_r W_r
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor of size 12 x 16 x 12.
        # We can check a few random entries or all.
        # 12*16*12 = 2304 entries.
        # R is up to 48.
        # 2304 * 48 ~ 110k ops. Fast.
        
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Construct T from factors
        # T = sum_r U[:, r] V[:, r] W[:, r]
        # This is equivalent to einsum('ar,br,cr->abc', U, V, W)
        T_factored = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Construct T expected
        T_exp = np.zeros((N, P, Q))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_exp[a, b, c] = 1
        
        return np.allclose(T_factored, T_exp, atol=1e-9)

    U, V, W = build_44_rank()
    
    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        U, V, W = build_48_rank()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
