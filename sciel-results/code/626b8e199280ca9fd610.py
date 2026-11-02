import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use a 4+2 block decomposition
    # A = [[A44, A42], [A24, A22]]
    # B = [[B44, B42], [B24, B22]]
    # C = [[A44*B44 + A42*B24, A44*B42 + A42*B22],
    #      [A24*B44 + A22*B24, A24*B42 + A22*B22]]
    
    # We need to compute 8 products:
    # 1. A44 * B44 (4x4 * 4x4) - use Laderman (rank 21)
    # 2. A42 * B24 (4x2 * 2x4) - use Strassen-like (rank 7)
    # 3. A24 * B44 (2x4 * 4x4) - use Strassen-like (rank 7)
    # 4. A44 * B42 (4x4 * 4x2) - use Strassen-like (rank 7)
    # 5. A24 * B42 (2x4 * 4x2) - use Strassen-like (rank 7)
    # 6. A42 * B22 (4x2 * 2x2) - use Strassen-like (rank 7)
    # 7. A22 * B24 (2x2 * 2x4) - use Strassen-like (rank 7)
    # 8. A22 * B22 (2x2 * 2x2) - use Strassen (rank 7)
    
    # Total rank: 21 + 7*7 = 70
    
    # Laderman's algorithm for 4x4 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A00+A10)(B11-B10)
    # M8 = (A01+A11)(B00-B01)
    
    # Wait, Laderman uses 21 multiplications, not 9.
    # Let's use the standard Laderman decomposition.
    
    # Actually, let's use a simpler approach: use Strassen for all blocks.
    # 6 = 3*2, so 3x3 blocks of 2x2.
    # Rank = 3^3 * 7 = 189 (current)
    
    # To improve, let's use a 4+2 decomposition with Laderman.
    # But implementing Laderman correctly is complex.
    
    # Alternative: Use the known rank-21 algorithm for 4x4 and pad to 6x6.
    # But that's not straightforward.
    
    # Let's stick with the block-Strassen approach but try to optimize it.
    # The current program already achieves 189.
    
    # Let's try a different approach: use the fact that 6 = 2*3 and use
    # a tensor product of 2x2 Strassen and 3x3 schoolbook.
    # Rank = 7 * 27 = 189 (same)
    
    # Or 3x3 Strassen-like and 2x2 schoolbook: 27 * 4 = 108? No, that's not valid.
    
    # Let's use the current block-Strassen approach which is known to work.
    
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Verify the decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        T = np.zeros((n, m, p))
        for r in range(U_mat.shape[1]):
            u_mat_2d = U_mat[:, r].reshape(n, m)
            v_mat_2d = V_mat[:, r].reshape(m, p)
            w_mat_2d = W_mat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat_2d, v_mat_2d, w_mat_2d)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=atol)
    
    if not verify_decomposition(U_bs, V_bs, W_bs):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_bs.tolist(), "V": V_bs.tolist(), "W": W_bs.tolist()}
