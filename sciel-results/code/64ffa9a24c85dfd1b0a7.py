"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _build_strassen_blocks(n, m, p, bn, bm, bp):
    """
    Build a block Strassen decomposition.
    n x m times m x p, with blocks of size bn x bm x bp.
    Uses Strassen 2x2 as base (so bn=bm=bp=2).
    """
    if n % bn != 0 or m % bm != 0 or p % bp != 0:
        return None
    if bn != 2 or bm != 2 or bp != 2:
        return None  # Only support 2x2 Strassen blocks
    
    ni = n // bn
    nj = m // bm
    nk = p // bp
    base_rank = 7
    total_rank = ni * nj * nk * base_rank
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    bU, bV, bW = _strassen_2x2()
    
    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                col_off = (ii * nj + jj) * nk * base_rank + kk * base_rank
                for r in range(base_rank):
                    for li in range(bn):
                        for lj in range(bm):
                            i = ii * bn + li
                            j = jj * bm + lj
                            val = bU[li * bm + lj][r]
                            if val != 0:
                                U[i * m + j, col_off + r] = val
                    for lj in range(bm):
                        for lk in range(bp):
                            j = jj * bm + lj
                            k = kk * bp + lk
                            val = bV[lj * bp + lk][r]
                            if val != 0:
                                V[j * p + k, col_off + r] = val
                    for lk in range(bp):
                        for li in range(bn):
                            k = kk * bp + lk
                            i = ii * bn + li
                            val = bW[lk * bn + li][r]
                            if val != 0:
                                W[k * n + i, col_off + r] = val
    
    # Convert to list of lists for interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    # Ensure integer values
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try block Strassen with 2x2 blocks
    if time.time() < deadline:
        result = _build_strassen_blocks(n, m, p, 2, 2, 2)
        if result is not None:
            block_rank = len(result["U"][0])
            if block_rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = block_rank
    
    return best
# EVOLVE-BLOCK-END
