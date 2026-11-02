"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
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
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2: (a10+a11)b00
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = -1
    
    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5: (a00+a01)b11
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    W[1][6] = 1
    
    return U, V, W

def _block_decompose(n, m, p, block_n, block_m, block_p, base_U, base_V, base_W, block_rank):
    """
    Decompose n x m * m x p into blocks of size block_n x block_m * block_p.
    Each block uses a decomposition of rank block_rank.
    Total rank = (n/block_n) * (m/block_m) * (p/block_p) * block_rank
    """
    if n % block_n != 0 or m % block_m != 0 or p % block_p != 0:
        return None
    
    num_blocks_i = n // block_n
    num_blocks_j = m // block_m
    num_blocks_k = p // block_p
    total_rank = num_blocks_i * num_blocks_j * num_blocks_k * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for bi in range(num_blocks_i):
        for bj in range(num_blocks_j):
            for bk in range(num_blocks_k):
                block_col_offset = (bi * num_blocks_j + bj) * num_blocks_k * block_rank + bk * block_rank
                
                for r in range(block_rank):
                    # U: rows are (i, j) where i in [bi*block_n, (bi+1)*block_n), j in [bj*block_m, (bj+1)*block_m)
                    for li in range(block_n):
                        for lj in range(block_m):
                            i = bi * block_n + li
                            j = bj * block_m + lj
                            u_val = base_U[li * block_m + lj][r]
                            if u_val != 0:
                                U[i * m + j][block_col_offset + r] = u_val
                    
                    # V: rows are (j, k) where j in [bj*block_m, (bj+1)*block_m), k in [bk*block_p, (bk+1)*block_p)
                    for lj in range(block_m):
                        for lk in range(block_p):
                            j = bj * block_m + lj
                            k = bk * block_p + lk
                            v_val = base_V[lj * block_p + lk][r]
                            if v_val != 0:
                                V[j * p + k][block_col_offset + r] = v_val
                    
                    # W: rows are (k, i) where k in [bk*block_p, (bk+1)*block_p), i in [bi*block_n, (bi+1)*block_n)
                    for lk in range(block_p):
                        for li in range(block_n):
                            k = bk * block_p + lk
                            i = bi * block_n + li
                            w_val = base_W[lk * block_n + li][r]
                            if w_val != 0:
                                W[k * n + i][block_col_offset + r] = w_val
    
    return {"U": U, "V": V, "W": W}

def _try_4x4x8_decomposition(n, m, p):
    """
    Try to construct a decomposition for 4x4 x 4x8 using block methods.
    """
    results = []
    
    # Approach 1: 2x2x2 Strassen blocks
    # 4/2 * 4/2 * 8/2 * 7 = 2 * 2 * 4 * 7 = 112
    base_U, base_V, base_W = _strassen_2x2()
    res1 = _block_decompose(n, m, p, 2, 2, 2, base_U, base_V, base_W, 7)
    if res1 is not None:
        if _verify(res1["U"], res1["V"], res1["W"], n, m, p):
            results.append(res1)
    
    # Approach 2: 4x4x4 blocks (using schoolbook for the base, or if we had a better one)
    # For 4x4x4, schoolbook rank is 64.
    # 4/4 * 4/4 * 8/4 * 64 = 1 * 1 * 2 * 64 = 128 (same as schoolbook)
    # But if we had a rank 20 decomposition for 4x4x4, it would be 1*1*2*20 = 40.
    # Since we don't have a known rank 20 hardcoded here, let's stick to what we have.
    
    # Approach 3: Mixed blocks?
    # e.g., 2x4x4 blocks?
    # If we use 2x2x2 Strassen for the i and j dimensions, and schoolbook for k?
    # This is essentially the same as the block decomposition above if the block sizes align.
    
    # Approach 4: Try 1x1x1 (Schoolbook) just in case
    # rank = 128
    
    if not results:
        return _schoolbook(n, m, p)
    
    # Return the one with the lowest rank
    best = min(results, key=lambda x: len(x["U"][0]))
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try block decomposition with Strassen 2x2 blocks
    if time.time() < deadline:
        try:
            candidate = _try_4x4x8_decomposition(n, m, p)
            cand_rank = len(candidate["U"][0])
            if cand_rank < best_rank:
                # Verify again to be safe
                if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                    best = candidate
                    best_rank = cand_rank
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
