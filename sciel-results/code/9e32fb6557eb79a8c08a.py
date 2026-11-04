import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 189  # 27 (3x3 schoolbook) * 7 (2x2 Strassen)
    
    U = np.zeros((n * m, rank), dtype=int)  # 36 x 189
    V = np.zeros((m * p, rank), dtype=int)  # 36 x 189
    W = np.zeros((p * n, rank), dtype=int)  # 36 x 189
    
    # For 2x2 Strassen:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define Strassen coefficients for each of the 7 products
    # Each entry: list of (position_in_A, coeff), list of (position_in_B, coeff), list of (position_in_C, coeff)
    # Positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_defs = [
        # M1 = (a00+a11)*(b00+b11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a10+a11)*b00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a00*(b01-b11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = a11*(b10-b00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a00+a01)*b11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (a10-a00)*(b00+b01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a01-a11)*(b10+b11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # A block at (i,k): rows 2i,2i+1; cols 2k,2k+1
                    # Flat index for A[row,col] = row*6 + col
                    a_pos = [
                        2 * i * 6 + 2 * k,       # a00
                        2 * i * 6 + 2 * k + 1,   # a01
                        (2 * i + 1) * 6 + 2 * k,     # a10
                        (2 * i + 1) * 6 + 2 * k + 1, # a11
                    ]
                    
                    # B block at (k,j): rows 2k,2k+1; cols 2j,2j+1
                    # Flat index for B[row,col] = row*6 + col
                    b_pos = [
                        2 * k * 6 + 2 * j,       # b00
                        2 * k * 6 + 2 * j + 1,   # b01
                        (2 * k + 1) * 6 + 2 * j,     # b10
                        (2 * k + 1) * 6 + 2 * j + 1, # b11
                    ]
                    
                    # C block at (i,j): rows 2i,2i+1; cols 2j,2j+1
                    # Trace convention: C[r,c] is at W_flat[c*6 + r]
                    # So C[row,col] -> W_flat[col*6 + row]
                    c_pos = [
                        2 * j * 6 + 2 * i,       # C[2i, 2j]
                        (2 * j + 1) * 6 + 2 * i,     # C[2i, 2j+1]
                        2 * j * 6 + (2 * i + 1),     # C[2i+1, 2j]
                        (2 * j + 1) * 6 + (2 * i + 1), # C[2i+1, 2j+1]
                    ]
                    
                    a_coeffs, b_coeffs, c_coeffs = strassen_defs[s]
                    
                    for pos, coeff in a_coeffs:
                        U[a_pos[pos], idx] += coeff
                    for pos, coeff in b_coeffs:
                        V[b_pos[pos], idx] += coeff
                    for pos, coeff in c_coeffs:
                        W[c_pos[pos], idx] += coeff
                    
                    idx += 1
    
    assert idx == rank, "Index mismatch: {} vs {}".format(idx, rank)
    
    # Verify correctness
    np.random.seed(12345)
    A_test = np.random.randint(-3, 3, (n, m)).astype(float)
    B_test = np.random.randint(-3, 3, (m, p)).astype(float)
    C_expected = A_test @ B_test
    
    U_f = U.astype(float)
    V_f = V.astype(float)
    W_f = W.astype(float)
    
    U_3d = U_f.reshape(n, m, rank)
    V_3d = V_f.reshape(m, p, rank)
    W_3d = W_f.reshape(p, n, rank)
    
    prod = np.einsum('ikr,kjr->ijr', U_3d, V_3d)
    C_computed = np.einsum('ijr,jir->ij', prod, W_3d)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook (rank 216)
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s), dtype=int)
        V_s = np.zeros((m * p, rank_s), dtype=int)
        W_s = np.zeros((p * n, rank_s), dtype=int)
        idx_s = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_s[i * m + k, idx_s] = 1
                    V_s[k * p + j, idx_s] = 1
                    W_s[j * n + i, idx_s] = 1
                    idx_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
