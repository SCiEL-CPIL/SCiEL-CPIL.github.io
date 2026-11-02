"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank):
        return False
    if Va.shape != (m*p, rank):
        return False
    if Wa.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
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

def _strassen_4x4_rank20():
    """
    Construct a rank-20 decomposition for 4x4 x 4x4 matrix multiplication.
    Returns None if not implemented.
    """
    return None

def _block_decompose_2x2(n, m, p):
    """
    Decompose n x m * m x p into 2x2x2 Strassen blocks.
    Total rank = (n/2) * (m/2) * (p/2) * 7
    """
    if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
        return None
    
    num_i = n // 2
    num_j = m // 2
    num_k = p // 2
    total_rank = num_i * num_j * num_k * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    base_U, base_V, base_W = _strassen_2x2()
    
    for bi in range(num_i):
        for bj in range(num_j):
            for bk in range(num_k):
                off = (bi * num_j + bj) * num_k * 7 + bk * 7
                
                for r in range(7):
                    for li in range(2):
                        for lj in range(2):
                            i = bi * 2 + li
                            j = bj * 2 + lj
                            val = base_U[li*2 + lj][r]
                            if val != 0:
                                U[i*m + j][off + r] = val
                    
                    for lj in range(2):
                        for lk in range(2):
                            j = bj * 2 + lj
                            k = bk * 2 + lk
                            val = base_V[lj*2 + lk][r]
                            if val != 0:
                                V[j*p + k][off + r] = val
                    
                    for lk in range(2):
                        for li in range(2):
                            k = bk * 2 + lk
                            i = bi * 2 + li
                            val = base_W[lk*2 + li][r]
                            if val != 0:
                                W[k*n + i][off + r] = val
    
    return {"U": U, "V": V, "W": W}

def _block_decompose_4x4(n, m, p):
    """
    Decompose n x m * m x p into 4x4x4 blocks, each using 2x2 Strassen sub-blocks.
    Each 4x4x4 block uses 8 2x2x2 Strassen decompositions (rank 56).
    Total rank = (n/4) * (m/4) * (p/4) * 56
    """
    if n % 4 != 0 or m % 4 != 0 or p % 4 != 0:
        return None
    
    num_i = n // 4
    num_j = m // 4
    num_k = p // 4
    total_rank = num_i * num_j * num_k * 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    base_U, base_V, base_W = _strassen_2x2()
    
    for bi in range(num_i):
        for bj in range(num_j):
            for bk in range(num_k):
                block_off = (bi * num_j + bj) * num_k * 56 + bk * 56
                
                # Within each 4x4x4 block, there are 8 2x2x2 sub-blocks
                for si in range(2):
                    for sj in range(2):
                        for sk in range(2):
                            sub_off = block_off + (si*2 + sj)*2*7 + sk*7
                            
                            for r in range(7):
                                for li in range(2):
                                    for lj in range(2):
                                        i = bi*4 + si*2 + li
                                        j = bj*4 + sj*2 + lj
                                        val = base_U[li*2 + lj][r]
                                        if val != 0:
                                            U[i*m + j][sub_off + r] = val
                                
                                for lj in range(2):
                                    for lk in range(2):
                                        j = bj*4 + sj*2 + lj
                                        k = bk*4 + sk*2 + lk
                                        val = base_V[lj*2 + lk][r]
                                        if val != 0:
                                            V[j*p + k][sub_off + r] = val
                                
                                for lk in range(2):
                                    for li in range(2):
                                        k = bk*4 + sk*2 + lk
                                        i = bi*4 + si*2 + li
                                        val = base_W[lk*2 + li][r]
                                        if val != 0:
                                            W[k*n + i][sub_off + r] = val
    
    return {"U": U, "V": V, "W": W}

def _decompose_4x4x8_rank32():
    """
    Construct a rank-32 decomposition for 4x4 x 4x8 matrix multiplication.
    
    Strategy: Partition the 4x4x8 tensor into two 4x4x4 blocks.
    For each 4x4x4 block, use a rank-16 decomposition based on
    the Kronecker product structure: (2x2 Strassen) x (2x1 identity).
    
    Specifically, view 4x4x4 as a Kronecker product of:
    - A 2x2x2 multiplication (Strassen rank 7)
    - A 2x2x2 multiplication (identity rank 4)
    Total rank per 4x4x4 block = 7 * 4 = 28
    
    But we can do better by sharing. Actually, let's use a different approach:
    View 4x4x8 as 4x4x4 + 4x4x4 (two blocks in the k-dimension).
    
    For each 4x4x4 block, use 2x2 Strassen blocks: rank 56 per block.
    Total: 2 * 56 = 112. Not good.
    
    Alternative: Use the fact that 4x4x8 can be seen as a single tensor.
    Let's try a direct construction with rank 32.
    
    Actually, the simplest correct approach: use 2x2 Strassen blocks.
    For 4x4x8: (4/2)*(4/2)*(8/2)*7 = 2*2*4*7 = 112.
    
    Let me try a different partition: 4x4x8 = (2x2) x (2x2) x (4x2) in some sense.
    
    For now, return None and fall back to the block decomposition.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try 2x2x2 Strassen blocks
    if time.time() < deadline:
        result_2x2 = _block_decompose_2x2(n, m, p)
        if result_2x2 is not None:
            rank_2x2 = len(result_2x2["U"][0])
            if rank_2x2 < best_rank:
                if _verify(result_2x2["U"], result_2x2["V"], result_2x2["W"], n, m, p):
                    best = result_2x2
                    best_rank = rank_2x2
    
    # Try 4x4x4 blocks (using 2x2 Strassen sub-blocks)
    if time.time() < deadline:
        result_4x4 = _block_decompose_4x4(n, m, p)
        if result_4x4 is not None:
            rank_4x4 = len(result_4x4["U"][0])
            if rank_4x4 < best_rank:
                if _verify(result_4x4["U"], result_4x4["V"], result_4x4["W"], n, m, p):
                    best = result_4x4
                    best_rank = rank_4x4
    
    # Try rank-32 decomposition for 4x4x8
    if time.time() < deadline:
        result_32 = _decompose_4x4x8_rank32()
        if result_32 is not None:
            rank_32 = len(result_32["U"][0])
            if rank_32 < best_rank:
                if _verify(result_32["U"], result_32["V"], result_32["W"], n, m, p):
                    best = result_32
                    best_rank = rank_32
    
    return best
# EVOLVE-BLOCK-END
