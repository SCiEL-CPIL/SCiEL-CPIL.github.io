import numpy as np
import time

def _verify_solution(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1 for all i,j,k.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    # Reshape to 3D tensors
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute the contraction
    # S[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
    try:
        S = np.einsum('ijs,jks,ksi->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(S, np.ones((n, m, p)), atol=tol)
    except Exception:
        return False

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, s], V[j*2+k, s], W[k*2+i, s]
    """
    n, m, p = 2, 2, 2
    R = 7
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen's 7 factors
    # Local indices:
    # U: 0:a00, 1:a01, 2:a10, 3:a11
    # V: 0:b00, 1:b01, 2:b10, 3:b11
    # W: 0:c00, 1:c01, 2:c10, 3:c11
    
    factors = [
        # 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # 2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # 3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # 4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # 5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # 6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # 7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    for s, (u, v, w) in enumerate(factors):
        U[:, s] = u
        V[:, s] = v
        W[:, s] = w
        
    return U, V, W

def _get_strassen_248():
    """
    Construct 2x4x8 matrix multiplication decomposition using 8 Strassen 2x2x2 blocks.
    Rank: 8 * 7 = 56.
    U: (2*4, 56) = (8, 56)
    V: (4*8, 56) = (32, 56)
    W: (8*2, 56) = (16, 56)
    """
    n, m, p = 2, 4, 8
    R = 56
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    U_base, V_base, W_base = _get_strassen_222()
    
    # We place 8 blocks of 2x2x2.
    # Block indices: (ib, jb, kb) where ib in 0..0, jb in 0..1, kb in 0..3
    # Total rank per block is 7.
    # Block offset in rank: (ib * 2 + jb) * 4 + kb, then * 7
    
    for ib in range(n // 2):
        for jb in range(m // 2):
            for kb in range(p // 2):
                # Rank offset for this block
                block_idx = (ib * (m//2) + jb) * (p//2) + kb
                r_start = block_idx * 7
                r_end = r_start + 7
                
                # Row/Col offsets in global matrices
                i_start = ib * 2
                j_start = jb * 2
                k_start = kb * 2
                
                # Fill U
                for s in range(7):
                    # U local indices: 0:00, 1:01, 2:10, 3:11
                    # Global i: i_start, i_start+1
                    # Global j: j_start, j_start+1
                    U[i_start*4 + j_start, r_start+s] = U_base[0, s]
                    U[i_start*4 + j_start+1, r_start+s] = U_base[1, s]
                    U[(i_start+1)*4 + j_start, r_start+s] = U_base[2, s]
                    U[(i_start+1)*4 + j_start+1, r_start+s] = U_base[3, s]
                    
                    # Fill V
                    # V local: 0:00, 1:01, 2:10, 3:11
                    # Global j: j_start, j_start+1
                    # Global k: k_start, k_start+1
                    V[j_start*8 + k_start, r_start+s] = V_base[0, s]
                    V[j_start*8 + k_start+1, r_start+s] = V_base[1, s]
                    V[(j_start+1)*8 + k_start, r_start+s] = V_base[2, s]
                    V[(j_start+1)*8 + k_start+1, r_start+s] = V_base[3, s]
                    
                    # Fill W
                    # W local: 0:00, 1:01, 2:10, 3:11
                    # Global k: k_start, k_start+1
                    # Global i: i_start, i_start+1
                    # W index: k*n + i
                    W[k_start*2 + i_start, r_start+s] = W_base[0, s]
                    W[k_start*2 + i_start+1, r_start+s] = W_base[1, s]
                    W[(k_start+1)*2 + i_start, r_start+s] = W_base[2, s]
                    W[(k_start+1)*2 + i_start+1, r_start+s] = W_base[3, s]
                    
    return U, V, W

def _get_schoolbook_148():
    """
    Construct 1x4x8 matrix multiplication decomposition (Schoolbook).
    Rank: 1*4*8 = 32.
    U: (1*4, 32) = (4, 32)
    V: (4*8, 32) = (32, 32)
    W: (8*1, 32) = (8, 32)
    """
    n, m, p = 1, 4, 8
    R = 32
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for j in range(m):
        for k in range(p):
            s = j * p + k
            U[j, s] = 1.0
            V[j*p + k, s] = 1.0
            W[k*n + 0, s] = 1.0 # i=0
            
    return U, V, W

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 140
    
    # 1. Get the 2x4x8 Strassen-based block (Rank 56)
    U_248, V_248, W_248 = _get_strassen_248()
    
    # 2. Get the 1x4x8 Schoolbook block (Rank 32)
    U_148, V_148, W_148 = _get_schoolbook_148()
    
    # 3. Combine into 3x4x8
    # Total Rank = 56 + 32 = 88
    rank_total = 88
    
    U_total = np.zeros((n*m, rank_total)) # (12, 88)
    V_total = np.zeros((m*p, rank_total)) # (32, 88)
    W_total = np.zeros((p*n, rank_total)) # (24, 88)
    
    # Copy 2x4x8 part
    # U rows: 0..7 (i=0,1)
    # V rows: 0..31 (all j, k)
    # W rows: 0..15 (i=0,1)
    U_total[:8, :56] = U_248
    V_total[:, :56] = V_248
    W_total[:16, :56] = W_248
    
    # Copy 1x4x8 part
    # U rows: 8..11 (i=2)
    # V rows: 0..31 (all j, k) -- Note: V is shared, so we just add to the same rows but different columns
    # W rows: 16..23 (i=2)
    U_total[8:, 56:] = U_148
    V_total[:, 56:] = V_148
    W_total[16:, 56:] = W_148
    
    # 4. Verify
    if _verify_solution(U_total, V_total, W_total, n, m, p):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # Fallback: Schoolbook 3x4x8 (Rank 96)
    rank = 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = i * m * p + j * p + k
                U[i*m + j, s] = 1
                V[j*p + k, s] = 1
                W[k*n + i, s] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
