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

def _kron_strassen_4x4():
    """
    Construct a rank-49 decomposition for 4x4 x 4x4 matrix multiplication
    using the Kronecker product of two 2x2x2 Strassen decompositions.
    Returns U, V, W of shapes (16, 49), (16, 49), (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            i = i1 * 2 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            val_u = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            U4[i * 4 + j, r] = val_u
                            
                            val_v = V2[j1 * 2 + (r1 % 2) , r1] * V2[j2 * 2 + (r2 % 2), r2] # Wait, V indices are j, k
                            # V2[j, k, r]. Here j is input to V, k is output from V (which is input to W).
                            # In the Kronecker product, V4[j, k, r] = V2[j1, k1, r1] * V2[j2, k2, r2].
                            # But k is also an index of V.
                            # Let's re-evaluate.
                            # T_{i,j,k} = sum U[i,r] V[j,r] W[k,r].
                            # T = T1 x T2.
                            # i = i1 x i2, j = j1 x j2, k = k1 x k2.
                            # U4[i, r] = U2[i1, r1] * U2[i2, r2].
                            # V4[j, r] = V2[j1, r1] * V2[j2, r2].
                            # W4[k, r] = W2[k1, r1] * W2[k2, r2].
                            # This is correct.
                            pass
                            
    # Let's just use numpy kron for simplicity and correctness.
    # U4 = kron(U2, U2) ?
    # Index i in U4 is i1*2+i2. Index r in U4 is r1*7+r2.
    # U4[i, r] = U2[i1, r1] * U2[i2, r2].
    # This is exactly the Kronecker product if we order indices correctly.
    # np.kron(A, B) has shape (na*nb, pa*pb).
    # (i1,i2) maps to i1*ni2 + i2.
    # (r1,r2) maps to r1*nr2 + r2.
    # So U4 = np.kron(U2, U2).
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    return U4.tolist(), V4.tolist(), W4.tolist()

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

def _decompose_4x4x8_rank98():
    """
    Construct a rank-98 decomposition for 4x4 x 4x8 matrix multiplication.
    Uses two copies of the rank-49 Kronecker-Strassen decomposition for 4x4x4.
    """
    U4, V4, W4 = _kron_strassen_4x4()
    U4 = np.array(U4)
    V4 = np.array(V4)
    W4 = np.array(W4)
    
    # U for 4x4x8 has shape (16, 98)
    U = np.zeros((16, 98))
    # V for 4x4x8 has shape (32, 98)
    V = np.zeros((32, 98))
    # W for 4x4x8 has shape (32, 98)
    W = np.zeros((32, 98))
    
    # First block (k=0..3)
    U[:, :49] = U4
    # V rows for k=0..3 are indices 0..15 (j*4+k)
    V[0:16, :49] = V4
    # W rows for k=0..3 are indices 0..15 (k*4+i)
    W[0:16, :49] = W4
    
    # Second block (k=4..7)
    U[:, 49:] = U4
    # V rows for k=4..7 are indices 16..31 (j*4 + (k-4) + 16? No, j*8+k)
    # j in 0..3, k in 4..7.
    # Row index in V is j*8 + k.
    # For j=0, k=4..7 -> rows 4,5,6,7.
    # For j=1, k=4..7 -> rows 12,13,14,15.
    # Wait, V shape is (m*p, rank) = (32, 98).
    # Indices are j*p + k. p=8.
    # So for k=0..3, rows are j*8 + k.
    # For k=4..7, rows are j*8 + k.
    
    # Let's re-map V4 to V.
    # V4 has shape (16, 49). Indices are j*4 + k (k in 0..3).
    # We need to place V4 into V at positions corresponding to k=0..3 and k=4..7.
    
    # Clear V and W
    V[:] = 0
    W[:] = 0
    
    # Fill first block (k=0..3)
    for j in range(4):
        for k in range(4):
            row_v = j * 8 + k
            row_v4 = j * 4 + k
            V[row_v, :49] = V4[row_v4, :]
            
    # Fill second block (k=4..7)
    for j in range(4):
        for k in range(4):
            row_v = j * 8 + (k + 4)
            row_v4 = j * 4 + k
            V[row_v, 49:] = V4[row_v4, :]
            
    # Fill W
    # W has shape (p*n, rank) = (32, 98).
    # Indices are k*n + i. n=4.
    # For k=0..3, rows are k*4 + i.
    # For k=4..7, rows are k*4 + i.
    
    for k in range(4):
        for i in range(4):
            row_w = k * 4 + i
            row_w4 = k * 4 + i
            W[row_w, :49] = W4[row_w4, :]
            
    for k in range(4):
        for i in range(4):
            row_w = (k + 4) * 4 + i
            row_w4 = k * 4 + i
            W[row_w, 49:] = W4[row_w4, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
    
    # Try rank-98 decomposition for 4x4x8
    if time.time() < deadline:
        result_98 = _decompose_4x4x8_rank98()
        if result_98 is not None:
            rank_98 = len(result_98["U"][0])
            if rank_98 < best_rank:
                if _verify(result_98["U"], result_98["V"], result_98["W"], n, m, p):
                    best = result_98
                    best_rank = rank_98
    
    return best
# EVOLVE-BLOCK-END
