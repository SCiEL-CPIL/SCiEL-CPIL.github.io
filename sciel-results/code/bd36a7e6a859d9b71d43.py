import time
import numpy as np

t0 = time.time()
DEADLINE = 140.0

def construct():
    t_start = time.time()
    def alive():
        return (time.time() - t_start) < DEADLINE

    n, m, p = 3, 4, 6
    total = n * m * p  # 72 tensor entries

    # Build a rank-66 decomposition using a different block structure:
    # Use 3x2x3 blocks (one per i-pair, one per k-pair)
    # M(3,4,6) as blocks of M(3,2,3) * 2 (for k) and M(3,4,2) * 3 (for k)
    # Alternative: use M(1,4,3) rank 12 * 3 + M(2,4,3) rank ? * 3
    #
    # Simpler approach: schoolbook rank 72, then try to reduce
    # Or: use the known M(3,4,6) <= 63 (Larsen et al. style)
    #
    # For now, build a clean rank-66 using Strassen-based blocks
    # Structure: split m=4 into two m=2 blocks, k=6 into three k=2 blocks
    # M(3,4,6) = sum over k-blocks of M(3,4,2)
    # M(3,4,2) = M(3,2,2)*2 (split m into 2+2)
    # M(3,2,2) = M(1,2,2) + M(2,2,2) = 2 + 7 = 9
    # M(3,4,2) = 2 * 9 = 18
    # M(3,4,6) = 3 * 18 = 54
    #
    # That gives rank 54! Let me build this.

    R = 54
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    col = 0

    def emit(u_sp, v_sp, w_sp):
        nonlocal col
        if not alive():
            return
        for (ii, jj), uu in u_sp:
            U[ii * m + jj, col] = uu
        for (jj, kk), vv in v_sp:
            V[jj * p + kk, col] = vv
        for (kk, ii), ww in w_sp:
            W[kk * n + ii, col] = ww
        col += 1

    def m122(i0, j0, k0):
        """M(1,2,2): 1x2 matrix times 2x2 matrix, rank 2 (schoolbook)"""
        for kk in range(2):
            for jj in range(2):
                emit([(i0, j0+j0) if False else (i0, jj)], [1.0],
                     [(jj, kk)], [1.0],
                     [(kk, i0)], [1.0])

    # Wait, M(1,2,2): A is 1x2, B is 2x2, C is 1x2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Schoolbook: for each (j,k) pair: 2*2=4 terms. But rank can be 2.
    # Actually M(1,2,2) = rank 2: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # This is a sum of 2 rank-1 terms (over k):
    # Term 1: A[0,0] * B[0,:], contributes to C[0,:]
    # Term 2: A[0,1] * B[1,:], contributes to C[0,:]
    # So rank 2 for M(1,2,2).

    def m122_correct(i0, j0, k0):
        """M(1,2,2): rank 2"""
        # Term 1: uses A[0,0] and B[0,k] for k=0,1
        emit([(i0, j0)], [1.0],
             [(j0, k0), (j0, k0+1)], [1.0, 1.0],
             [(k0, i0), (k0+1, i0)], [1.0, 1.0])
        # Term 2: uses A[0,1] and B[1,k] for k=0,1
        emit([(i0, j0+1)], [1.0],
             [(j0+1, k0), (j0+1, k0+1)], [1.0, 1.0],
             [(k0, i0), (k0+1, i0)], [1.0, 1.0])

    def m222(i0, j0, k0):
        """M(2,2,2): Strassen's 7 multiplications"""
        # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
        emit([(i0,j0),(i0+1,j0+1)], [1.0,1.0],
             [(j0,k0),(j0+1,k0+1)], [1.0,1.0],
             [(k0,i0),(k0+1,i0+1)], [1.0,1.0])
        # m2 = (A10+A11)(B00) -> C10 += m2, C11 -= m2
        emit([(i0+1,j0),(i0+1,j0+1)], [1.0,1.0],
             [(j0,k0)], [1.0],
             [(k0,i0+1),(k0+1,i0+1)], [1.0,-1.0])
        # m3 = (A00)(B01-B11) -> C01 += m3, C11 += m3
        emit([(i0,j0)], [1.0],
             [(j0,k0+1),(j0+1,k0+1)], [1.0,-1.0],
             [(k0+1,i0),(k0+1,i0+1)], [1.0,1.0])
        # m4 = (A11)(B10-B00) -> C00 += m4, C10 += m4
        emit([(i0+1,j0+1)], [1.0],
             [(j0+1,k0),(j0,k0)], [1.0,-1.0],
             [(k0,i0),(k0,i0+1)], [1.0,1.0])
        # m5 = (A00+A01)(B11) -> C00 -= m5, C01 += m5
        emit([(i0,j0),(i0,j0+1)], [1.0,1.0],
             [(j0+1,k0+1)], [1.0],
             [(k0+1,i0),(k0+1,i0+1)], [-1.0,1.0])
        # m6 = (A10-A00)(B00+B01) -> C11 += m6
        emit([(i0+1,j0),(i0,j0)], [1.0,-1.0],
             [(j0,k0),(j0,k0+1)], [1.0,1.0],
             [(k0+1,i0+1)], [1.0])
        # m7 = (A01-A11)(B10+B11) -> C00 += m7
        emit([(i0,j0+1),(i0+1,j0+1)], [1.0,-1.0],
             [(j0+1,k0),(j0+1,k0+1)], [1.0,1.0],
             [(k0,i0)], [1.0])

    def m322(i0, j0, k0):
        """M(3,2,2): A is 3x2, B is 2x2, C is 3x2, rank 9 = M(1,2,2)+M(2,2,2)"""
        # Split rows: row 0 uses m122, rows 1-2 use m222
        m122_correct(i0, j0, k0)
        m222(i0+1, j0, k0)

    def m342(i0, j0, k0):
        """M(3,4,2): A is 3x4, B is 4x2, C is 3x2, rank 18"""
        # Split columns of A / rows of B into two groups of 2
        m322(i0, j0, k0)
        m322(i0, j0+2, k0)

    # Build M(3,4,6) as three M(3,4,2) blocks in k
    for k_block in range(3):
        if not alive():
            break
        k0 = k_block * 2
        m342(0, 0, k0)

    # Verify
    if alive() and col > 0:
        Uv = U[:, :col]
        Vv = V[:, :col]
        Wv = W[:, :col]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(col):
                        s += Uv[i*m+j, r] * Vv[j*p+k, r] * Wv[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if not ok:
            # Fallback: schoolbook rank 72
            R2 = 72
            U2 = np.zeros((n*m, R2), dtype=np.float64)
            V2 = np.zeros((m*p, R2), dtype=np.float64)
            W2 = np.zeros((p*n, R2), dtype=np.float64)
            c2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j, c2] = 1.0
                        V2[j*p+k, c2] = 1.0
                        W2[k*n+i, c2] = 1.0
                        c2 += 1
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return {"U": U[:, :col].tolist(), "V": V[:, :col].tolist(), "W": W[:, :col].tolist()}
