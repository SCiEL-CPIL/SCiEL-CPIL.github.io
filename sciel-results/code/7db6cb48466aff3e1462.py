import numpy as np
import time

def _build_block_strassen_6x6():
    """
    Build a rank-189 decomposition for 6x6 matrix multiplication using
    a 3x3 grid of 2x2 blocks, each multiplied via Strassen (rank 7).
    Total rank = 3^3 * 7 = 189.
    
    A is partitioned into 3x3 blocks of size 2x2:
    A[ib, kb] is rows ib*2..ib*2+1, cols kb*2..kb*2+1
    B[kb, jb] is rows kb*2..kb*2+1, cols jb*2..jb*2+1
    C[ib, jb] is rows ib*2..ib*2+1, cols jb*2..jb*2+1
    
    C[ib, jb] = sum_kb A[ib, kb] * B[kb, jb]
    
    For each (ib, kb, jb), we use Strassen on the 2x2 block multiplication.
    
    U index: i*m + j where i is row of A, j is col of A
    V index: j*p + k where j is row of B, k is col of B
    W index: k*n + i where k is col of C, i is row of C
    
    For block (ib, kb, jb):
    A block rows: ib*2, ib*2+1; cols: kb*2, kb*2+1
    B block rows: kb*2, kb*2+1; cols: jb*2, jb*2+1
    C block rows: ib*2, ib*2+1; cols: jb*2, jb*2+1
    
    U[i*m + j] for i in {ib*2, ib*2+1}, j in {kb*2, kb*2+1}
    V[j*p + k] for j in {kb*2, kb*2+1}, k in {jb*2, jb*2+1}
    W[k*n + i] for k in {jb*2, jb*2+1}, i in {ib*2, ib*2+1}
    """
    n = 6
    rank_total = 189
    U = np.zeros((n * n, rank_total))
    V = np.zeros((n * n, rank_total))
    W = np.zeros((n * n, rank_total))
    
    col = 0
    for ib in range(3):
        for kb in range(3):
            for jb in range(3):
                # Strassen 7 multiplications for A[ib,kb] * B[kb,jb]
                # M1 = (A00 + A11)(B00 + B11)
                U[(ib*2)*n + kb*2, col] = 1
                U[(ib*2+1)*n + (kb*2+1), col] = 1
                V[(kb*2)*n + jb*2, col] = 1
                V[(kb*2+1)*n + (jb*2+1), col] = 1
                W[(jb*2)*n + ib*2, col] = 1
                W[(jb*2+1)*n + (ib*2+1), col] = 1
                col += 1
                
                # M2 = (A10 + A11)B00
                U[(ib*2+1)*n + kb*2, col] = 1
                U[(ib*2+1)*n + (kb*2+1), col] = 1
                V[(kb*2)*n + jb*2, col] = 1
                W[(jb*2+1)*n + ib*2, col] = 1
                W[(jb*2+1)*n + (ib*2+1), col] = -1
                col += 1
                
                # M3 = A00(B10 - B11)
                U[(ib*2)*n + kb*2, col] = 1
                V[(kb*2)*n + (jb*2+1), col] = 1
                V[(kb*2+1)*n + (jb*2+1), col] = -1
                W[(jb*2+1)*n + ib*2, col] = 1
                W[(jb*2+1)*n + (ib*2+1), col] = 1
                col += 1
                
                # M4 = A11(B01 - B00)
                U[(ib*2+1)*n + (kb*2+1), col] = 1
                V[(kb*2+1)*n + (jb*2+1), col] = 1
                V[(kb*2)*n + jb*2, col] = -1
                W[(jb*2)*n + ib*2, col] = 1
                W[(jb*2+1)*n + ib*2, col] = 1
                col += 1
                
                # M5 = (A00 + A01)B11
                U[(ib*2)*n + kb*2, col] = 1
                U[(ib*2)*n + (kb*2+1), col] = 1
                V[(kb*2+1)*n + (jb*2+1), col] = 1
                W[(jb*2)*n + ib*2, col] = -1
                W[(jb*2+1)*n + ib*2, col] = 1
                col += 1
                
                # M6 = (A10 - A00)(B00 + B01)
                U[(ib*2+1)*n + kb*2, col] = 1
                U[(ib*2)*n + kb*2, col] = -1
                V[(kb*2)*n + jb*2, col] = 1
                V[(kb*2)*n + (jb*2+1), col] = 1
                W[(jb*2+1)*n + (ib*2+1), col] = 1
                col += 1
                
                # M7 = (A01 - A11)(B10 + B11)
                U[(ib*2)*n + (kb*2+1), col] = 1
                U[(ib*2+1)*n + (kb*2+1), col] = -1
                V[(kb*2+1)*n + (jb*2+1), col] = 1
                V[(kb*2)*n + (jb*2+1), col] = 1
                W[(jb*2)*n + ib*2, col] = 1
                col += 1
    
    return U, V, W

def _verify(n, m, p, U, V, W, tol=1e-6):
    """Verify the decomposition."""
    # Check diagonal entries (i,j,k) where i=i', j=j', k=k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check off-diagonal entries should be zero
    np.random.seed(42)
    for _ in range(200):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0
    
    U, V, W = _build_block_strassen_6x6()
    
    if _verify(6, 6, 6, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook 216
    U_sb = np.zeros((36, 216))
    V_sb = np.zeros((36, 216))
    W_sb = np.zeros((36, 216))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U_sb[i*6+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*6+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
