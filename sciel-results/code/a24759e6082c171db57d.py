"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 matrix multiplication."""
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    if not _verify_decomposition(2, 2, 2, U.tolist(), V.tolist(), W.tolist()):
        return None
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _place_block(U_full, V_full, W_full, U_b, V_b, W_b, i_off, j_off, k_off, n, m, p, col_start):
    """Place a block decomposition into the full matrices."""
    rank_b = U_b.shape[1]
    
    # Determine block dimensions
    found = False
    for nb in range(1, 10):
        for mb in range(1, 10):
            for pb in range(1, 10):
                if U_b.shape[0] == nb * mb and V_b.shape[0] == mb * pb and W_b.shape[0] == pb * nb:
                    for t in range(rank_b):
                        for ii in range(nb):
                            for jj in range(mb):
                                full_u_row = (i_off + ii) * m + (j_off + jj)
                                U_full[full_u_row, col_start + t] = U_b[ii * mb + jj, t]
                        
                        for jj in range(mb):
                            for kk in range(pb):
                                full_v_row = (j_off + jj) * p + (k_off + kk)
                                V_full[full_v_row, col_start + t] = V_b[jj * pb + kk, t]
                        
                        for kk in range(pb):
                            for ii in range(nb):
                                full_w_row = (k_off + kk) * n + (i_off + ii)
                                W_full[full_w_row, col_start + t] = W_b[kk * nb + ii, t]
                    found = True
                    return col_start + rank_b
    if not found:
        raise ValueError("Could not determine block dimensions")

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    sU, sV, sW = _strassen_2x2()
    c1U, c1V, c1W = _schoolbook_nm_p(2, 3, 2)
    c2U, c2V, c2W = _schoolbook_nm_p(3, 2, 3)
    bU, bV, bW = _schoolbook_nm_p(3, 3, 3)
    
    total_rank = sU.shape[1] + c1U.shape[1] + c2U.shape[1] + bU.shape[1]
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col = 0
    col = _place_block(U_full, V_full, W_full, sU, sV, sW, 0, 0, 0, n, m, p, col)
    col = _place_block(U_full, V_full, W_full, c1U, c1V, c1W, 0, 2, 0, n, m, p, col)
    col = _place_block(U_full, V_full, W_full, c2U, c2V, c2W, 2, 0, 2, n, m, p, col)
    col = _place_block(U_full, V_full, W_full, bU, bV, bW, 2, 2, 2, n, m, p, col)
    
    if not _verify_decomposition(n, m, p, U_full.tolist(), V_full.tolist(), W_full.tolist()):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
