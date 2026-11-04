"""Rewritten program: search for a low-rank exact tensor decomposition of 4x5 * 5x5."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
n, m, p = DIMENSIONS
N_ROWS = n * m  # 20
M_ROWS = m * p  # 25
P_ROWS = p * n  # 20

def _build_tensor():
    T = np.zeros((N_ROWS, M_ROWS, P_ROWS))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

T_target = _build_tensor()

def _schoolbook_decomp(n, m, p):
    """Returns U, V, W for n x m x p schoolbook multiplication."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1.0
                V[v, col] = 1.0
                W[w, col] = 1.0
                col += 1
    return U, V, W

def _strassen_decomp_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen multiplication."""
    # Rank 7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: 00, 01, 10, 11
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
    
    # M2 = A22(B21-B11)
    U[3, 1] = 1
    V[2, 1] = 1; V[0, 1] = -1
    W[3, 1] = 1 # C22 += M2
    W[2, 1] = 1 # C21 += M2
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12 += M3
    W[3, 2] = -1 # C22 -= M3
    
    # M4 = (A11+A12)B22
    U[0, 3] = 1; U[1, 3] = 1
    V[3, 3] = 1
    W[0, 3] = 1 # C11 += M4
    W[2, 3] = 1 # C21 += M4
    
    # M5 = (A21+A22)B11
    U[2, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1
    W[0, 4] = -1 # C11 -= M5
    W[1, 4] = 1 # C12 += M5
    
    # M6 = (A12-A22)(B21+B22)
    U[1, 5] = 1; U[3, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1 # C22 += M6
    
    # M7 = (A11-A21)(B11+B12)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1 # C11 += M7
    
    return U, V, W

def _combine_decompositions(sub_decomps, offsets):
    """
    Combines a list of (U, V, W) decompositions into global U, V, W.
    offsets: list of (i_start, j_start, k_start, n_loc, m_loc, p_loc)
    """
    total_rank = sum(U.shape[1] for U, V, W in sub_decomps)
    U_glob = np.zeros((N_ROWS, total_rank))
    V_glob = np.zeros((M_ROWS, total_rank))
    W_glob = np.zeros((P_ROWS, total_rank))
    
    col_offset = 0
    for (U, V, W), (i_start, j_start, k_start, n_loc, m_loc, p_loc) in zip(sub_decomps, offsets):
        rank_loc = U.shape[1]
        # Map local indices to global
        for i_loc in range(n_loc):
            for j_loc in range(m_loc):
                r_loc = i_loc * m_loc + j_loc
                r_glob = (i_start + i_loc) * m + (j_start + j_loc)
                U_glob[r_glob, col_offset:col_offset+rank_loc] = U[r_loc, :]
                
        for j_loc in range(m_loc):
            for k_loc in range(p_loc):
                r_loc = j_loc * p_loc + k_loc
                r_glob = (j_start + j_loc) * p + (k_start + k_loc)
                V_glob[r_glob, col_offset:col_offset+rank_loc] = V[r_loc, :]
                
        for k_loc in range(p_loc):
            for i_loc in range(n_loc):
                r_loc = k_loc * n_loc + i_loc
                r_glob = (k_start + k_loc) * n + (i_start + i_loc)
                W_glob[r_glob, col_offset:col_offset+rank_loc] = W[r_loc, :]
                
        col_offset += rank_loc
        
    return U_glob, V_glob, W_glob

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Sub-problems for Top block (i in 0,1)
    # A_top: 2x5, B: 5x5
    # Split B cols 0,1 and 2,3,4. Split B rows 0,1 and 2,3,4.
    
    decomp_top = []
    offsets_top = []
    
    # 1. A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # i_start=0, j_start=0, k_start=0, n=2, m=2, p=2
    U, V, W = _strassen_decomp_2x2x2()
    decomp_top.append((U, V, W))
    offsets_top.append((0, 0, 0, 2, 2, 2))
    
    # 2. A12 (2x3) * B21 (3x2) -> C11 (2x2)
    # i_start=0, j_start=2, k_start=0, n=2, m=3, p=2
    U, V, W = _schoolbook_decomp(2, 3, 2)
    decomp_top.append((U, V, W))
    offsets_top.append((0, 2, 0, 2, 3, 2))
    
    # 3. A11 (2x2) * B12 (2x3) -> C12 (2x3)
    # i_start=0, j_start=0, k_start=2, n=2, m=2, p=3
    U, V, W = _schoolbook_decomp(2, 2, 3)
    decomp_top.append((U, V, W))
    offsets_top.append((0, 0, 2, 2, 2, 3))
    
    # 4. A12 (2x3) * B22 (3x3) -> C12 (2x3)
    # i_start=0, j_start=2, k_start=2, n=2, m=3, p=3
    U, V, W = _schoolbook_decomp(2, 3, 3)
    decomp_top.append((U, V, W))
    offsets_top.append((0, 2, 2, 2, 3, 3))
    
    # Sub-problems for Bottom block (i in 2,3)
    # Same structure, but i_start=2
    
    decomp_bot = []
    offsets_bot = []
    
    # 1. A21 (2x2) * B11 (2x2) -> C21 (2x2)
    # i_start=2, j_start=0, k_start=0, n=2, m=2, p=2
    U, V, W = _strassen_decomp_2x2x2()
    decomp_bot.append((U, V, W))
    offsets_bot.append((2, 0, 0, 2, 2, 2))
    
    # 2. A22 (2x3) * B21 (3x2) -> C21 (2x2)
    # i_start=2, j_start=2, k_start=0, n=2, m=3, p=2
    U, V, W = _schoolbook_decomp(2, 3, 2)
    decomp_bot.append((U, V, W))
    offsets_bot.append((2, 2, 0, 2, 3, 2))
    
    # 3. A21 (2x2) * B12 (2x3) -> C22 (2x3)
    # i_start=2, j_start=0, k_start=2, n=2, m=2, p=3
    U, V, W = _schoolbook_decomp(2, 2, 3)
    decomp_bot.append((U, V, W))
    offsets_bot.append((2, 0, 2, 2, 2, 3))
    
    # 4. A22 (2x3) * B22 (3x3) -> C22 (2x3)
    # i_start=2, j_start=2, k_start=2, n=2, m=3, p=3
    U, V, W = _schoolbook_decomp(2, 3, 3)
    decomp_bot.append((U, V, W))
    offsets_bot.append((2, 2, 2, 2, 3, 3))
    
    # Combine
    U_glob, V_glob, W_glob = _combine_decompositions(decomp_top, offsets_top)
    
    # Append bottom
    U_bot, V_bot, W_bot = _combine_decompositions(decomp_bot, offsets_bot)
    
    U_final = np.hstack([U_glob, U_bot])
    V_final = np.hstack([V_glob, V_bot])
    W_final = np.hstack([W_glob, W_bot])
    
    # Verify
    T_recon = np.einsum('ur,vr,wr->uvw', U_final, V_final, W_final)
    if not np.allclose(T_recon, T_target, atol=1e-10):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((N_ROWS, rank_sb))
        V_sb = np.zeros((M_ROWS, rank_sb))
        W_sb = np.zeros((P_ROWS, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U_sb[u, col] = 1.0
                    V_sb[v, col] = 1.0
                    W_sb[w, col] = 1.0
                    col += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
        
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
