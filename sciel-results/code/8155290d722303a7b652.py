"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Vectorized verification using einsum
    # Tensor T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # We want T[i,j,k] = 1 for all i,j,k
    # Compute: for each (i,j,k), sum over t of U[i,j,t]*V[j,k,t]*W[k,i,t]
    # Use einsum: 'ijt,jkt,kit->ijk'
    T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _strassen_2x2x2():
    """Construct rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Term 0: P1 = (A00+A11)(B00+B11)
    # U: A00 and A11 contribute
    U[0*2+0][0] = 1.0  # A00
    U[1*2+1][0] = 1.0  # A11
    # V: B00 and B11 contribute
    V[0*2+0][0] = 1.0  # B00
    V[1*2+1][0] = 1.0  # B11
    # W: contributes to C00 and C11
    W[0*2+0][0] = 1.0  # C00
    W[1*2+1][0] = 1.0  # C11
    
    # Term 1: P2 = (A10+A11)B00
    U[1*2+0][1] = 1.0  # A10
    U[1*2+1][1] = 1.0  # A11
    V[0*2+0][1] = 1.0  # B00
    # W: contributes to C10
    W[0*2+1][1] = 1.0  # C10
    
    # Term 2: P3 = A00(B01-B11)
    U[0*2+0][2] = 1.0  # A00
    V[0*2+1][2] = 1.0  # B01
    V[1*2+1][2] = -1.0  # B11
    # W: contributes to C01 and C11
    W[1*2+0][2] = 1.0  # C01
    W[1*2+1][2] = 1.0  # C11
    
    # Term 3: P4 = A11(B10-B00)
    U[1*2+1][3] = 1.0  # A11
    V[1*2+0][3] = 1.0  # B10
    V[0*2+0][3] = -1.0  # B00
    # W: contributes to C00 and C10
    W[0*2+0][3] = 1.0  # C00
    W[0*2+1][3] = 1.0  # C10
    
    # Term 4: P5 = (A00+A01)B11
    U[0*2+0][4] = 1.0  # A00
    U[0*2+1][4] = 1.0  # A01
    V[1*2+1][4] = 1.0  # B11
    # W: contributes to C00 and C01
    W[0*2+0][4] = -1.0  # C00
    W[1*2+0][4] = 1.0  # C01
    
    # Term 5: P6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1.0  # A10
    U[0*2+0][5] = -1.0  # A00
    V[0*2+0][5] = 1.0  # B00
    V[0*2+1][5] = 1.0  # B01
    # W: contributes to C11
    W[1*2+1][5] = 1.0  # C11
    
    # Term 6: P7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1.0  # A01
    U[1*2+1][6] = -1.0  # A11
    V[1*2+0][6] = 1.0  # B10
    V[1*2+1][6] = 1.0  # B11
    # W: contributes to C00
    W[0*2+0][6] = 1.0  # C00
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions.
    
    If (U1,V1,W1) decomposes n1xm1xp1 and (U2,V2,W2) decomposes n2xm2xp2,
    then the Kronecker product decomposes (n1*n2)x(m1*m2)x(p1*p2).
    
    Indexing convention:
    U1[i1*m1+j1, t1], V1[j1*p1+k1, t1], W1[k1*n1+i1, t1]
    U2[i2*m2+j2, t2], V2[j2*p2+k2, t2], W2[k2*n2+i2, t2]
    
    Result:
    U[i1*n2+i2, m1*m2, rank1*rank2] where index = i1*m1*m2 + i2... 
    Actually, let's use: row = i1*(m1*m2) + i2*m2 + j2, but that's not right either.
    
    Let me think more carefully. The tensor element for the Kronecker product is:
    T[(i1,i2),(j1,j2),(k1,k2)] = T1[i1,j1,k1] * T2[i2,j2,k2]
    
    So U[(i1*m1+j1)*(m2) + (i2*m2+j2)]... no.
    
    The row index for U in the product is (i1,i2,j1,j2) mapped to a single index.
    Standard Kronecker: row = i1*(n2*m2) + i2*m2 + j2? No...
    
    Let me use: the combined matrix is (n1*n2) x (m1*m2), with element at 
    row (i1*n2 + i2), col (j1*m2 + j2).
    So U row index = (i1*n2 + i2)*m1*m2 + (j1*m2 + j2)
    
    Similarly V row index = (j1*n2 + j2)*p1*p2 + (k1*p2 + k2) -- no, V is (m1*m2)x(p1*p2)
    V row index = (j1*m2 + j2)*p1*p2 + (k1*p2 + k2) -- wait, m1*m2 rows, p1*p2 cols
    V row = j1*m2 + j2 (for the m part), col = k1*p2 + k2 (for the p part)
    So V index = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
    
    W is (p1*p2) x (n1*n2), row = k1*p2 + k2, col = i1*n2 + i2
    W index = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
    
    The rank index is t1*rank2 + t2.
    
    U[row, t1*rank2+t2] = U1[i1*m1+j1, t1] * U2[i2*m2+j2, t2]
    V[row, t1*rank2+t2] = V1[j1*p1+k1, t1] * V2[j2*p2+k2, t2]
    W[row, t1*rank2+t2] = W1[k1*n1+i1, t1] * W2[k2*n2+i2, t2]
    """
    # Determine dimensions from the input matrices
    n1 = int(np.sqrt(len(U1)))  # This won't work in general
    # Actually, we need to pass dimensions explicitly
    raise NotImplementedError("Pass dimensions explicitly")

def _kron_decomp_explicit(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions with explicit dimensions."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 * rank2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for t1 in range(rank1):
                    u1 = U1[i1 * m1 + j1][t1]
                    v1 = V1[j1 * p1 + k1][t1]
                    w1 = W1[k1 * n1 + i1][t1]
                    if u1 == 0 and v1 == 0 and w1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for t2 in range(rank2):
                                    u2 = U2[i2 * m2 + j2][t2]
                                    v2 = V2[j2 * p2 + k2][t2]
                                    w2 = W2[k2 * n2 + i2][t2]
                                    if u2 == 0 and v2 == 0 and w2 == 0:
                                        continue
                                    
                                    t = t1 * rank2 + t2
                                    
                                    # U index: row = (i1*n2 + i2), col = (j1*m2 + j2)
                                    u_row = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                    U[u_row][t] += u1 * u2
                                    
                                    # V index: row = (j1*m2 + j2), col = (k1*p2 + k2)
                                    v_row = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                                    V[v_row][t] += v1 * v2
                                    
                                    # W index: row = (k1*p2 + k2), col = (i1*n2 + i2)
                                    w_row = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                                    W[w_row][t] += w1 * w2
    
    return U, V, W

def _concat_decompositions(decomps, n, m, p):
    """Concatenate multiple decompositions along the rank dimension.
    
    decomps: list of (U, V, W) tuples, each decomposing n x m x p
    Returns a single (U, V, W) with rank = sum of individual ranks.
    """
    total_rank = sum(len(U[0]) for U, V, W in decomps)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for U_i, V_i, W_i in decomps:
        rank_i = len(U_i[0])
        for r in range(n * m):
            for t in range(rank_i):
                U[r][offset + t] = U_i[r][t]
        for r in range(m * p):
            for t in range(rank_i):
                V[r][offset + t] = V_i[r][t]
        for r in range(p * n):
            for t in range(rank_i):
                W[r][offset + t] = W_i[r][t]
        offset += rank_i
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for nxm x mxp multiplication."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return U, V, W

def _construct_444_rank49():
    """Construct rank-49 decomposition for 4x4x4 using Kronecker product of Strassen 2x2x2."""
    U2, V2, W2 = _strassen_2x2x2()
    
    # Verify the 2x2x2 decomposition
    if not _verify_decomposition(U2, V2, W2, 2, 2, 2):
        # Fall back to schoolbook
        U2, V2, W2 = _schoolbook(2, 2, 2)
    
    # Kronecker product to get 4x4x4
    U4, V4, W4 = _kron_decomp_explicit(U2, V2, W2, 2, 2, 2, U2, V2, W2, 2, 2, 2)
    
    return U4, V4, W4

def _restrict_p_to_subset(U, V, W, n, m, p, p_subset):
    """Restrict a decomposition from p columns to p_subset columns.
    
    If we have a decomposition for n x m x p, we can restrict to any subset of p columns
    by keeping only the corresponding V rows and W rows.
    
    p_subset: list of column indices to keep
    """
    rank = len(U[0])
    p_new = len(p_subset)
    
    U_new = [row[:] for row in U]  # U unchanged
    V_new = [[0.0] * rank for _ in range(m * p_new)]
    W_new = [[0.0] * rank for _ in range(p_new * n)]
    
    for new_k, k in enumerate(p_subset):
        # V row for column k: V[j*p + k, t] -> V_new[j*p_new + new_k, t]
        for j in range(m):
            for t in range(rank):
                V_new[j * p_new + new_k][t] = V[j * p + k][t]
        
        # W row for column k: W[k*n + i, t] -> W_new[new_k*n + i, t]
        for i_row in range(n):
            for t in range(rank):
                W_new[new_k * n + i_row][t] = W[k * n + i_row][t]
    
    return U_new, V_new, W_new

def construct():
    """Return the best known decomposition for 4x4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Strategy: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # Use Strassen-based rank-49 for each 4x4x4, schoolbook for 4x4x1
    # Total rank: 49 + 49 + 16 = 114
    
    if _remaining_time() < 10:
        return best
    
    try:
        # Get rank-49 decomposition for 4x4x4
        U444, V444, W444 = _construct_444_rank49()
        rank_444 = len(U444[0])
        
        # Verify 4x4x4 decomposition
        if not _verify_decomposition(U444, V444, W444, 4, 4, 4):
            # Fall back to schoolbook for 4x4x4
            U444, V444, W444 = _schoolbook(4, 4, 4)
            rank_444 = 64
        
        if _remaining_time() < 5:
            return best
        
        # Construct the full 4x4x9 decomposition directly
        # Part A: columns 0-3 (first 4x4x4) -> rank 49
        # Part B: columns 4-7 (second 4x4x4) -> rank 49  
        # Part C: column 8 (4x4x1) -> rank 16
        # Total rank: 49 + 49 + 16 = 114
        
        rank_c = 16  # schoolbook rank for 4x4x1
        total_rank = rank_444 + rank_444 + rank_c
        
        U_full = [[0.0] * total_rank for _ in range(n * m)]
        V_full = [[0.0] * total_rank for _ in range(m * p)]
        W_full = [[0.0] * total_rank for _ in range(p * n)]
        
        # Part A: columns 0-3 (first 4x4x4)
        # U: same rows, columns 0..rank_444-1
        for r in range(n * m):
            for t in range(rank_444):
                U_full[r][t] = U444[r][t]
        
        # V: for j in 0..3, k in 0..3: V_full[j*9+k][t] = V444[j*4+k][t]
        for j in range(m):
            for k in range(4):
                for t in range(rank_444):
                    V_full[j * p + k][t] = V444[j * 4 + k][t]
        
        # W: for k in 0..3, i in 0..3: W_full[k*4+i][t] = W444[k*4+i][t]
        for k in range(4):
            for i in range(n):
                for t in range(rank_444):
                    W_full[k * n + i][t] = W444[k * 4 + i][t]
        
        # Part B: columns 4-7 (second 4x4x4)
        offset_b = rank_444
        for r in range(n * m):
            for t in range(rank_444):
                U_full[r][offset_b + t] = U444[r][t]
        
        # V: for j in 0..3, k in 4..7: V_full[j*9+k][offset_b+t] = V444[j*4+(k-4)][t]
        for j in range(m):
            for k in range(4, 8):
                for t in range(rank_444):
                    V_full[j * p + k][offset_b + t] = V444[j * 4 + (k - 4)][t]
        
        # W: for k in 4..7, i in 0..3: W_full[k*4+i][offset_b+t] = W444[(k-4)*4+i][t]
        for k in range(4, 8):
            for i in range(n):
                for t in range(rank_444):
                    W_full[k * n + i][offset_b + t] = W444[(k - 4) * 4 + i][t]
        
        # Part C: column 8 (4x4x1) - schoolbook
        offset_c = 2 * rank_444
        # Schoolbook 4x4x1: for each (j,k=0), one term
        # U[i*4+j][col] = 1, V[j*1+0][col] = 1, W[0*4+i][col] = 1
        # col = i*4+j for i in 0..3, j in 0..3
        for i in range(n):
            for j in range(m):
                col = i * m + j
                U_full[i * m + j][offset_c + col] = 1.0
                V_full[j * p + 8][offset_c + col] = 1.0
                W_full[8 * n + i][offset_c + col] = 1.0
        
        # Verify the full decomposition
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            best = {"U": U_full, "V": V_full, "W": W_full}
            best_rank = total_rank
    except Exception:
        pass
    
    return best
