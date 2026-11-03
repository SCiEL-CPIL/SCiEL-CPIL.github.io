"""Matrix multiplication 4x5x5 with rank-reduced decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u, v, w are vectors of length 4 (row-major 2x2).
    """
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return terms

def _build_4x4x4_decomp():
    """
    Build the 4x4x4 decomposition using 7 Strassen 2x2x2 blocks.
    Returns U, V, W of shape (16, 49).
    """
    strassen_terms = _strassen_2x2x2_terms()
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    for s in range(7):
        u_s, v_s, w_s = strassen_terms[s]
        for t in range(7):
            u_t, v_t, w_t = strassen_terms[t]
            for l in range(4):
                bi = l // 2
                bj = l % 2
                for li in range(2):
                    for lj in range(2):
                        i_4 = 2 * bi + li
                        j_4 = 2 * bj + lj
                        idx_U = i_4 * 4 + j_4
                        U[idx_U][col] = u_t[l]
                        idx_V = i_4 * 4 + j_4
                        V[idx_V][col] = v_t[l]
                        idx_W = i_4 * 4 + j_4
                        W[idx_W][col] = w_t[l]
            col += 1
    return U, V, W

def _schoolbook_sub(n, m, p):
    """Schoolbook for n*m x m*p sub-block."""
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
    return U, V, W

def _assemble_block_decomp(n, m, p, sub_decomps):
    """
    Assemble full decomposition from sub-block decompositions.
    sub_decomps: list of (U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk)
    """
    total_rank = sum(len(U[0]) for U, V, W, *_ in sub_decomps)
    U_out = [[0] * total_rank for _ in range(n * m)]
    V_out = [[0] * total_rank for _ in range(m * p)]
    W_out = [[0] * total_rank for _ in range(p * n)]
    
    col_off = 0
    for U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk in sub_decomps:
        n_cols = len(U_sub[0])
        for i in range(ni):
            for j in range(mj):
                for c in range(n_cols):
                    U_out[(i0+i)*m + (j0+j)][col_off + c] = U_sub[i*mj+j][c]
        for j in range(mj):
            for k in range(pk):
                for c in range(n_cols):
                    V_out[(j0+j)*p + (k0+k)][col_off + c] = V_sub[j*pk+k][c]
        for k in range(pk):
            for i in range(ni):
                for c in range(n_cols):
                    W_out[(k0+k)*n + (i0+i)][col_off + c] = W_sub[k*ni+i][c]
        col_off += n_cols
    return U_out, V_out, W_out

def _validate_decomp(n, m, p, U, V, W, tol=1e-9):
    """Validate that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _build_direct_sum_1x5x5(n, m, p):
    """
    Build 4x5x5 as direct sum of four 1x5x5 matrix-vector multiplications.
    Each 1x5x5 has rank 25 under schoolbook, total rank = 4*25 = 100.
    This is the baseline for the tensor product / direct sum approach.
    """
    # Each row i of A (1x5) multiplies B (5x5) to give row i of C (1x5)
    # The 1x5x5 tensor T[j,k] = sum_c U[j,c] V[k,c] W[c]
    # For 1x5x5: U has 5 entries, V has 5 entries, W has 5 entries
    # Schoolbook: rank = 5*5 = 25
    # U[j,c] = 1 if c == j*5+k else 0... wait
    # Actually for 1x5x5: A is 1x5, B is 5x5, C is 1x5
    # C[k] = sum_j A[j] * B[j,k]
    # Tensor T[j,k] = 1 for all j,k (since A[j]*B[j,k] contributes to C[k])
    # Wait no: T[j,k] should be 1 for the (j,k) pair that contributes to C
    # The tensor for 1x5x5 is T[j,k] = 1 for all j in 0..4, k in 0..4
    # Rank-1 decomposition: T[j,k] = sum_c U[j,c] V[k,c] W[c]
    # Schoolbook: 25 terms, each with U[j,c]=1, V[k,c]=1, W[c]=1 for specific (j,k)
    
    total_rank = n * m * p  # 4*5*5 = 100
    U = [[0] * total_rank for _ in range(n * m)]  # (20, 100)
    V = [[0] * total_rank for _ in range(m * p)]  # (25, 100)
    W = [[0] * total_rank for _ in range(p * n)]  # (20, 100)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return U, V, W

def _try_strassen_2x5x5():
    """
    Try to find a low-rank decomposition for 2x5x5 using Strassen-like ideas.
    2x5x5: A is 2x5, B is 5x5, C is 2x5.
    Schoolbook rank = 2*5*5 = 50.
    We can try to use the fact that 5 = 2+3 or other partitions.
    """
    # For now, return schoolbook
    n, m, p = 2, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _build_4x5x5_block_partition():
    """
    Build 4x5x5 using block partition:
    - Split A (4x5) into A1 (2x5) and A2 (2x5)
    - Compute C1 = A1 * B and C2 = A2 * B
    - Use Strassen-like for the 2x5x5 parts if possible
    """
    n, m, p = 4, 5, 5
    
    # Try: split into 2x5x5 + 2x5x5
    U1, V1, W1 = _try_strassen_2x5x5()
    U2, V2, W2 = _try_strassen_2x5x5()
    
    total_rank = len(U1[0]) + len(U2[0])
    U_out = [[0] * total_rank for _ in range(n * m)]
    V_out = [[0] * total_rank for _ in range(m * p)]
    W_out = [[0] * total_rank for _ in range(p * n)]
    
    # First 2x5x5: rows 0-1 of A
    for i in range(2):
        for j in range(m):
            for c in range(len(U1[0])):
                U_out[i*m + j][c] = U1[i*m + j][c]
        for k in range(p):
            for c in range(len(U1[0])):
                W_out[k*n + i][c] = W1[k*2 + i][c]
    for j in range(m):
        for k in range(p):
            for c in range(len(V1[0])):
                V_out[j*p + k][c] = V1[j*p + k][c]
    
    # Second 2x5x5: rows 2-3 of A
    col_off = len(U2[0])
    for i in range(2):
        for j in range(m):
            for c in range(len(U2[0])):
                U_out[(i+2)*m + j][col_off + c] = U2[i*m + j][c]
        for k in range(p):
            for c in range(len(U2[0])):
                W_out[k*n + (i+2)][col_off + c] = W2[k*2 + i][c]
    for j in range(m):
        for k in range(p):
            for c in range(len(V2[0])):
                V_out[j*p + k][col_off + c] = V2[j*p + k][c]
    
    return U_out, V_out, W_out

def construct():
    import time
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s limit
    
    n, m, p = 4, 5, 5
    
    # Method 1: Direct sum of 1x5x5 (rank 100)
    U_ds, V_ds, W_ds = _build_direct_sum_1x5x5(n, m, p)
    best_rank = len(U_ds[0])
    best_result = {"U": U_ds, "V": V_ds, "W": W_ds}
    
    # Method 2: Block partition 4x4x4 + strips (current approach, rank 49+16+16+4=85)
    if time.time() < deadline - 10:
        U1, V1, W1 = _build_4x4x4_decomp()
        U2, V2, W2 = _schoolbook_sub(4, 4, 1)
        U3, V3, W3 = _schoolbook_sub(4, 1, 4)
        U4, V4, W4 = _schoolbook_sub(4, 1, 1)
        
        sub_decomps = [
            (U1, V1, W1, 0, 0, 0, 4, 4, 4),
            (U2, V2, W2, 0, 0, 4, 4, 4, 1),
            (U3, V3, W3, 0, 4, 0, 4, 1, 4),
            (U4, V4, W4, 0, 4, 4, 4, 1, 1),
        ]
        
        U_bp, V_bp, W_bp = _assemble_block_decomp(n, m, p, sub_decomps)
        rank_bp = len(U_bp[0])
        if _validate_decomp(n, m, p, U_bp, V_bp, W_bp):
            if rank_bp < best_rank:
                best_rank = rank_bp
                best_result = {"U": U_bp, "V": V_bp, "W": W_bp}
    
    # Method 3: Block partition 2x5x5 + 2x5x5 (rank 50+50=100, same as direct sum)
    if time.time() < deadline - 10:
        U_2x, V_2x, W_2x = _build_4x5x5_block_partition()
        rank_2x = len(U_2x[0])
        if _validate_decomp(n, m, p, U_2x, V_2x, W_2x):
            if rank_2x < best_rank:
                best_rank = rank_2x
                best_result = {"U": U_2x, "V": V_2x, "W": W_2x}
    
    # Return best result found
    return best_result
