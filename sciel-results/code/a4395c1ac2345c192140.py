"""Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: Build a block-Strassen decomposition by partitioning indices into
2x2 and 1x1 blocks, then applying Strassen's algorithm to each 2x2x2 sub-tensor.
This gives rank 108 (12*7 + 6*4), better than schoolbook rank 120.
Then attempt greedy column removal to reduce rank further.
"""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

try:
    import sympy
except Exception:
    sympy = None

DIMENSIONS = [4, 5, 6]
SEED = 42


def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2 matrix multiplication.
    Convention: C[i,k] = sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t]
    For 2x2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    Index: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # u: A11=1, A22=1 -> [1,0,0,1]
    # v: B11=1, B22=1 -> [1,0,0,1]
    # w: C11=1, C22=1 -> [1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A21+A22)B11
    # u: A21=1, A22=1 -> [0,0,1,1]
    # v: B11=1 -> [1,0,0,0]
    # w: C21=1 -> [0,0,1,0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3 = A11(B12-B22)
    # u: A11=1 -> [1,0,0,0]
    # v: B12=1, B22=-1 -> [0,1,0,-1]
    # w: C12=1 -> [0,1,0,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4 = A22(B21-B11)
    # u: A22=1 -> [0,0,0,1]
    # v: B21=1, B11=-1 -> [-1,0,1,0]
    # w: C11=1, C21=1 -> [1,0,1,0]
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # M5 = (A11+A12)B22
    # u: A11=1, A12=1 -> [1,1,0,0]
    # v: B22=1 -> [0,0,0,1]
    # w: C11=-1, C12=1 -> [-1,1,0,0]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A21-A11)(B11+B12)
    # u: A21=1, A11=-1 -> [-1,0,1,0]
    # v: B11=1, B12=1 -> [1,1,0,0]
    # w: C22=1 -> [0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A12-A22)(B21+B22)
    # u: A12=1, A22=-1 -> [0,1,0,-1]
    # v: B21=1, B22=1 -> [0,0,1,1]
    # w: C11=1 -> [1,0,0,0]
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms


def _build_block_strassen(n=4, m=5, p=6):
    """Build a block-Strassen decomposition for n x m * m x p.
    
    Partition indices into 2x2 and 1x1 blocks:
    - i in {0,1,2,3}: blocks {0,1} and {2,3}
    - j in {0,1,2,3,4}: blocks {0,1}, {2,3}, {4}
    - k in {0,1,2,3,4,5}: blocks {0,1}, {2,3}, {4,5}
    
    For each combination of (i_block, j_block, k_block):
    - If all blocks are size 2: use Strassen (rank 7)
    - If one block is size 1: use schoolbook (rank 4 for 2x1x2, etc.)
    
    Total: 12 * 7 + 6 * 4 = 84 + 24 = 108
    """
    R = 108
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    strassen = _strassen_2x2()
    col = 0
    
    # Define blocks
    i_blocks = [(0,1), (2,3)]  # size 2 each
    j_blocks = [(0,1), (2,3), (4,)]  # size 2, 2, 1
    k_blocks = [(0,1), (2,3), (4,5)]  # size 2 each
    
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                si = len(ib)  # 2
                sj = len(jb)  # 2 or 1
                sk = len(kb)  # 2
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen: rank 7
                    for t in range(7):
                        u_vec, v_vec, w_vec = strassen[t]
                        for di in range(2):
                            for dj in range(2):
                                i = ib[0] + di
                                j = jb[0] + dj
                                U[i * m + j][col] += u_vec[di * 2 + dj]
                        for dj in range(2):
                            for dk in range(2):
                                j = jb[0] + dj
                                k = kb[0] + dk
                                V[j * p + k][col] += v_vec[dj * 2 + dk]
                        for dk in range(2):
                            for di in range(2):
                                k = kb[0] + dk
                                i = ib[0] + di
                                W[k * n + i][col] += w_vec[dk * 2 + di]
                        col += 1
                elif si == 2 and sj == 1 and sk == 2:
                    # Schoolbook: rank 4 (2*1*2)
                    for di in range(2):
                        for dk in range(2):
                            i = ib[0] + di
                            j = jb[0]
                            k = kb[0] + dk
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                else:
                    # Should not happen for our partition
                    pass
    
    return U, V, W, R


def _build_schoolbook(n=4, m=5, p=6):
    """Build the schoolbook decomposition: rank = n*m*p = 120."""
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W, R


def _verify_numpy(U, V, W, n, m, p):
    """Verify the decomposition using numpy."""
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True


def _verify_sympy(U, V, W, n, m, p):
    """Verify the decomposition using sympy for exactness."""
    if sympy is None:
        return False
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = sympy.Integer(0)
                    for c in range(R):
                        wv = sympy.Integer(W[k * n + i][c])
                        uu = sympy.Integer(U[i * m + j][c])
                        vv = sympy.Integer(V[j2 * p + k][c])
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = sympy.Integer(1) if j == j2 else sympy.Integer(0)
                    if s != target:
                        return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing columns that are redundant."""
    R = len(U[0])
    changed = True
    while changed and time.time() < deadline - 30.0:
        changed = False
        R = len(U[0])
        for c in range(R):
            if time.time() > deadline - 30.0:
                break
            # Try removing column c
            keep = [x for x in range(R) if x != c]
            U2 = [[U[r][x] for x in keep] for r in range(n * m)]
            V2 = [[V[r][x] for x in keep] for r in range(m * p)]
            W2 = [[W[r][x] for x in keep] for r in range(p * n)]
            if _verify_numpy(U2, V2, W2, n, m, p):
                U, V, W = U2, V2, W2
                changed = True
                break
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    try:
        # Build block-Strassen decomposition (rank 108)
        U, V, W, R = _build_block_strassen(n, m, p)
        
        # Verify
        if sympy is not None and R <= 130:
            ok = _verify_sympy(U, V, W, n, m, p)
        else:
            ok = _verify_numpy(U, V, W, n, m, p)
        
        if not ok:
            # Fallback: pure schoolbook
            U, V, W, R = _build_schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
        
        # Try to reduce rank
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
        
        return {"U": U, "V": V, "W": W}
    
    except Exception:
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
