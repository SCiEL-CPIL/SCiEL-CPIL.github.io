import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (24, R)
    
    # Two-level Strassen: 7 top-level terms, each with 11 sub-multiplications
    # Total rank: 7 * 11 = 77
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 x 2x2
    # A blocks indexed 0-3: (0,0), (0,1), (1,0), (1,1)
    # B blocks indexed 0-3: (0,0), (0,1), (1,0), (1,1)
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},   # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},   # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},   # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},   # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0},  # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}   # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},   # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1},  # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0},  # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},   # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},   # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}    # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0: C00 += M0, C11 += M0
        [(1, 0, 1), (1, 1, -1)],    # s=1: C10 += M1, C11 -= M1
        [(0, 1, 1), (1, 1, 1)],     # s=2: C01 += M2, C11 += M2
        [(0, 0, 1), (1, 0, 1)],     # s=3: C00 += M3, C10 += M3
        [(0, 0, -1), (0, 1, 1)],    # s=4: C00 -= M4, C01 += M4
        [(1, 1, 1)],                # s=5: C11 += M5
        [(0, 0, 1)]                 # s=6: C00 += M6
    ]
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # U part: combine X_s entries via inner_alpha
            # X_s[i,j] = sum_r alpha[s][r] * A[r//2, r%2][i,j]
            # inner_X[i,j] = sum_a alpha[si][a] * X_s[a//2, a%2]
            # => U coefficient for A[ar,br][i,j] = alpha[si][a] * alpha[s][r]
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += ca * cr
            
            # V part: combine Y_s entries via inner_beta
            # Y_s[i,j] = sum_c beta[s][c] * B[c//2, c%2][i,j]
            # inner_Y[i,j] = sum_b beta[si][b] * Y_s[b//2, b%2]
            # => V coefficient for B[ac,bc][i,j] = beta[si][b] * beta[s][c]
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += cb * cc
            
            # W part: output Z[i,j] = P_s[i,j] for j in {0,1}
            # Z[i,j] contributes to C[a,b][i,j] with coeff gamma*zgamma
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        # Z[u] = sum_v X_s[u,v] * Y_s[v,2]
        for u in range(2):
            for v in range(2):
                # U part: X_s[u,v] = sum_r alpha[s][r] * A[r//2, r%2][u,v]
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += cr
                
                # V part: Y_s[v,2] = sum_c beta[s][c] * B[c//2, c%2][v,2]
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += cc
                
                # W part: Z[u] contributes to C[a,b][u,2] with coeff gamma
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify: check that the decomposition is correct
    # For random A (4x4) and B (4x6), compute C = A*B
    # and compare with tensor reconstruction
    if time.time() - START_TIME < TIME_LIMIT - 10:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with a few random matrices
        for trial in range(3):
            A_test = np.random.randint(-5, 5, (n, m))
            B_test = np.random.randint(-5, 5, (m, p))
            C_expected = A_test @ B_test
            
            # Flatten A, B
            A_flat = A_test.flatten()  # row-major: (ar*2+u)*4 + (br*2+v)
            B_flat = B_test.flatten()  # row-major: (ac*2+v)*6 + (bc*3+w)
            
            # Compute tensor decomposition result
            # C[k*n+i, r] = sum_t U[i*m+k, t] * V[k*p+r, t] * W[r*n+i, t]
            # Wait, let me re-read the convention: W[k*n+i, r]
            # U[i*m+k, t], V[k*p+r, t], W[r*n+i, t]
            # C[i, r] = sum_k sum_t U[i*m+k, t] * V[k*p+r, t] * W[r*n+i, t]
            
            C_recon = np.zeros((n, p), dtype=float)
            for i in range(n):
                for r in range(p):
                    for k in range(m):
                        for t in range(R):
                            C_recon[i, r] += U_np[i * m + k, t] * V_np[k * p + r, t] * W_np[r * n + i, t]
            
            if not np.allclose(C_recon, C_expected, atol=1e-9):
                # Fallback: return schoolbook if verification fails
                rank = n * m * p
                U_f = np.zeros((n * m, rank))
                V_f = np.zeros((m * p, rank))
                W_f = np.zeros((p * n, rank))
                idx = 0
                for i in range(n):
                    for k in range(m):
                        for r in range(p):
                            U_f[i * m + k, idx] = 1
                            V_f[k * p + r, idx] = 1
                            W_f[r * n + i, idx] = 1
                            idx += 1
                return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
