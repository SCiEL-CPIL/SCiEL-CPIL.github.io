"""Matrix multiplication 2x4 by 4x5: rank-36 exact tensor decomposition (fixed)."""
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    _start = time.time()
    _deadline = _start + 140.0

    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        """Strassen 2x2x2 with = assignment."""
        c = col_start
        # M1 = (A00+A11)*(B00+B11)
        U[i0 * m + j0][c] = 1
        U[(i0+1) * m + (j0+1)][c] = 1
        V[j0 * p + k0][c] = 1
        V[(j0+1) * p + (k0+1)][c] = 1
        W[k0 * n + i0][c] = 1
        W[(k0+1) * n + (i0+1)][c] = 1
        c += 1
        # M2 = (A10+A11)*B00
        U[(i0+1) * m + j0][c] = 1
        U[(i0+1) * m + (j0+1)][c] = 1
        V[j0 * p + k0][c] = 1
        W[k0 * n + (i0+1)][c] = 1
        W[(k0+1) * n + (i0+1)][c] = -1
        c += 1
        # M3 = A00*(B01-B11)
        U[i0 * m + j0][c] = 1
        V[j0 * p + (k0+1)][c] = 1
        V[(j0+1) * p + (k0+1)][c] = -1
        W[(k0+1) * n + i0][c] = 1
        W[(k0+1) * n + (i0+1)][c] = 1
        c += 1
        # M4 = A11*(B10-B00)
        U[(i0+1) * m + (j0+1)][c] = 1
        V[(j0+1) * p + k0][c] = 1
        V[j0 * p + k0][c] = -1
        W[k0 * n + i0][c] = 1
        W[k0 * n + (i0+1)][c] = 1
        c += 1
        # M5 = (A00+A01)*B11
        U[i0 * m + j0][c] = 1
        U[i0 * m + (j0+1)][c] = 1
        V[(j0+1) * p + (k0+1)][c] = 1
        W[k0 * n + i0][c] = -1
        W[(k0+1) * n + i0][c] = 1
        c += 1
        # M6 = (A10-A00)*(B00+B01)
        U[(i0+1) * m + j0][c] = 1
        U[i0 * m + j0][c] = -1
        V[j0 * p + k0][c] = 1
        V[j0 * p + (k0+1)][c] = 1
        W[(k0+1) * n + (i0+1)][c] = 1
        c += 1
        # M7 = (A01-A11)*(B10+B11)
        U[i0 * m + (j0+1)][c] = 1
        U[(i0+1) * m + (j0+1)][c] = -1
        V[(j0+1) * p + k0][c] = 1
        V[(j0+1) * p + (k0+1)][c] = 1
        W[k0 * n + i0][c] = 1
        c += 1
        return c

    def add_schoolbook_2x4x1(col_start):
        """Schoolbook 2x4x1: 8 terms."""
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[di * m + dj][c] = 1
                V[dj * p + 4][c] = 1
                W[4 * n + di][c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, col)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify correctness
    if time.time() < _deadline:
        try:
            import numpy as np
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0.0
                    for j in range(m):
                        for r in range(rank):
                            total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] * A_test[i,j] * B_test[j,k]
                    C_computed[i, k] = total
            
            max_err = np.max(np.abs(C_computed - C_expected))
            if max_err > 1e-10:
                # Fallback to pure schoolbook rank 40
                rank_fb = 40
                U_fb = [[0]*rank_fb for _ in range(n*m)]
                V_fb = [[0]*rank_fb for _ in range(m*p)]
                W_fb = [[0]*rank_fb for _ in range(p*n)]
                c = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i*m+j][c] = 1
                            V_fb[j*p+k][c] = 1
                            W_fb[k*n+i][c] = 1
                            c += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
