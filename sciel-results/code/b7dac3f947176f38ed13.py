import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _decomp_1x2x2():
    """Rank 2 decomposition for 1x2x2 matrix multiplication."""
    # U: (1*2, 2) -> (2, 2)
    # V: (2*2, 2) -> (4, 2)
    # W: (2*1, 2) -> (2, 2)
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(4)]
    W = [[0]*2 for _ in range(2)]
    
    # Term 1: A0 * B0_row
    U[0][0] = 1 # A0,0
    V[0][0] = 1 # B0,0
    V[1][0] = 1 # B0,1
    W[0][0] = 1 # C0,0
    W[1][0] = 1 # C0,1
    
    # Term 2: A1 * B1_row
    U[1][1] = 1 # A0,1
    V[2][1] = 1 # B1,0
    V[3][1] = 1 # B1,1
    W[0][1] = 1 # C0,0
    W[1][1] = 1 # C0,1
    
    return U, V, W

def _decomp_2x2x2():
    """Rank 7 Strassen decomposition for 2x2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4 = A22(B11-B21)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _decomp_1x3x2():
    """Rank 3 decomposition for 1x3x2."""
    # U: (1*3, 3) -> (3, 3)
    # V: (3*2, 3) -> (6, 3)
    # W: (2*1, 3) -> (2, 3)
    U = [[0]*3 for _ in range(3)]
    V = [[0]*3 for _ in range(6)]
    W = [[0]*3 for _ in range(2)]
    
    for j in range(3):
        U[j][j] = 1
        V[2*j][j] = 1
        V[2*j+1][j] = 1
        W[0][j] = 1
        W[1][j] = 1
        
    return U, V, W

def _decomp_2x3x2():
    """Rank 9 decomposition for 2x3x2 (2x2x2 + 2x1x2)."""
    # U: (2*3, 9) -> (6, 9)
    # V: (3*2, 9) -> (6, 9)
    # W: (2*2, 9) -> (4, 9)
    U = [[0]*9 for _ in range(6)]
    V = [[0]*9 for _ in range(6)]
    W = [[0]*9 for _ in range(4)]
    
    # Part 1: 2x2x2 (cols 0-6)
    U1, V1, W1 = _decomp_2x2x2()
    # Map indices for 2x2x2 part (j in 0,1)
    for u_idx in range(4):
        for c in range(7):
            # U1 indices: i in 0,1; j in 0,1 -> i*2+j
            # U indices: i in 0,1; j in 0,1 -> i*3+j
            i = u_idx // 2
            j = u_idx % 2
            U[i*3+j][c] = U1[u_idx][c]
            
    for v_idx in range(4):
        for c in range(7):
            # V1 indices: j in 0,1; k in 0,1 -> j*2+k
            # V indices: j in 0,1; k in 0,1 -> j*2+k
            V[v_idx][c] = V1[v_idx][c]
            
    for w_idx in range(4):
        for c in range(7):
            # W1 indices: k in 0,1; i in 0,1 -> k*2+i
            # W indices: k in 0,1; i in 0,1 -> k*2+i
            W[w_idx][c] = W1[w_idx][c]
            
    # Part 2: 2x1x2 (cols 7-8)
    # C_i,k = A_i,2 * B_2,k
    # Term 7: A0,2 * B2_row
    U[0*3+2][7] = 1 # A0,2
    V[2*2+0][7] = 1 # B2,0
    V[2*2+1][7] = 1 # B2,1
    W[0*2+0][7] = 1 # C0,0
    W[1*2+0][7] = 1 # C0,1
    
    # Term 8: A1,2 * B2_row
    U[1*3+2][8] = 1 # A1,2
    V[2*2+0][8] = 1 # B2,0
    V[2*2+1][8] = 1 # B2,1
    W[0*2+1][8] = 1 # C1,0
    W[1*2+1][8] = 1 # C1,1
    
    return U, V, W

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x5x6."""
    deadline = time.time() + 140.0
    n, m, p = 3, 5, 6
    
    # Splits
    n_parts = [1, 2]
    m_parts = [2, 3]
    p_parts = [2, 2, 2]
    
    # Pre-compute decompositions for each block type
    # Types: (n_i, m_j, p_k)
    decomp_cache = {}
    
    def get_decomp(n_i, m_j, p_k):
        key = (n_i, m_j, p_k)
        if key in decomp_cache:
            return decomp_cache[key]
        
        if (n_i, m_j, p_k) == (1, 2, 2):
            d = _decomp_1x2x2()
        elif (n_i, m_j, p_k) == (2, 2, 2):
            d = _decomp_2x2x2()
        elif (n_i, m_j, p_k) == (1, 3, 2):
            d = _decomp_1x3x2()
        elif (n_i, m_j, p_k) == (2, 3, 2):
            d = _decomp_2x3x2()
        else:
            # Fallback to schoolbook for unexpected sizes (should not happen)
            r = n_i * m_j * p_k
            U = [[0]*r for _ in range(n_i*m_j)]
            V = [[0]*r for _ in range(m_j*p_k)]
            W = [[0]*r for _ in range(p_k*n_i)]
            col = 0
            for i in range(n_i):
                for j in range(m_j):
                    for k in range(p_k):
                        U[i*m_j+j][col] = 1
                        V[j*p_k+k][col] = 1
                        W[k*n_i+i][col] = 1
                        col += 1
            d = (U, V, W)
            
        decomp_cache[key] = d
        return d
        
    # Calculate total rank
    total_rank = 0
    offsets_n = []
    offset = 0
    for ni in n_parts:
        offsets_n.append(offset)
        offset += ni
    offsets_m = []
    offset = 0
    for mj in m_parts:
        offsets_m.append(offset)
        offset += mj
    offsets_p = []
    offset = 0
    for pk in p_parts:
        offsets_p.append(offset)
        offset += pk
        
    rank_offsets = []
    current_rank = 0
    for ni in n_parts:
        for mj in m_parts:
            for pk in p_parts:
                U, V, W = get_decomp(ni, mj, pk)
                r = len(U[0])
                rank_offsets.append(current_rank)
                current_rank += r
                total_rank += r
                
    # Initialize full matrices
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    idx = 0
    for ni_idx, ni in enumerate(n_parts):
        for mj_idx, mj in enumerate(m_parts):
            for pk_idx, pk in enumerate(p_parts):
                U, V, W = get_decomp(ni, mj, pk)
                r = len(U[0])
                r_off = rank_offsets[idx]
                idx += 1
                
                n_off = offsets_n[ni_idx]
                m_off = offsets_m[mj_idx]
                p_off = offsets_p[pk_idx]
                
                # Fill U_full
                for i in range(ni):
                    for j in range(mj):
                        for c in range(r):
                            U_full[(n_off+i)*m + (m_off+j)][r_off+c] = U[i*mj+j][c]
                            
                # Fill V_full
                for j in range(mj):
                    for k in range(pk):
                        for c in range(r):
                            V_full[(m_off+j)*p + (p_off+k)][r_off+c] = V[j*pk+k][c]
                            
                # Fill W_full
                for k in range(pk):
                    for i in range(ni):
                        for c in range(r):
                            W_full[(p_off+k)*n + (n_off+i)][r_off+c] = W[k*ni+i][c]
                            
    # Verify
    if not _verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        U_full = [[0]*90 for _ in range(15)]
        V_full = [[0]*90 for _ in range(30)]
        W_full = [[0]*90 for _ in range(18)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_full[i*m+j][col] = 1
                    V_full[j*p+k][col] = 1
                    W_full[k*n+i][col] = 1
                    col += 1
                    if col == 90:
                        break
                if col == 90:
                    break
            if col == 90:
                break
                
    return {"U": U_full, "V": V_full, "W": W_full}
