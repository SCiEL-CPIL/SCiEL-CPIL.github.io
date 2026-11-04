"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback: rank = n*m*p = 120
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def strassen2_decomposition():
        """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        return U, V, W

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def try_decomposition(n_bounds, m_bounds, p_bounds, use_strassen2=True):
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline:
            return
        
        # Calculate total rank
        total_rank = 0
        for bi in range(len(n_bounds)-1):
            for bj in range(len(m_bounds)-1):
                for bk in range(len(p_bounds)-1):
                    ns = n_bounds[bi+1] - n_bounds[bi]
                    ms = m_bounds[bj+1] - m_bounds[bj]
                    ps = p_bounds[bk+1] - p_bounds[bk]
                    if use_strassen2 and ns == 2 and ms == 2 and ps == 2:
                        total_rank += 7
                    else:
                        total_rank += ns * ms * ps
        
        if total_rank >= best_rank:
            return
        
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col_offset = 0
        U2, V2, W2 = strassen2_decomposition()
        
        for bi in range(len(n_bounds)-1):
            for bj in range(len(m_bounds)-1):
                for bk in range(len(p_bounds)-1):
                    ni = n_bounds[bi]
                    nj = n_bounds[bi+1]
                    mi = m_bounds[bj]
                    mj = m_bounds[bj+1]
                    pi = p_bounds[bk]
                    pj = p_bounds[bk+1]
                    
                    n_size = nj - ni
                    m_size = mj - mi
                    p_size = pj - pi
                    
                    if use_strassen2 and n_size == 2 and m_size == 2 and p_size == 2:
                        for r in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U_full[(ni+ii)*m + (mi+jj)][col_offset + r] = U2[ii*2+jj][r]
                            for jj in range(2):
                                for kk in range(2):
                                    V_full[(mi+jj)*p + (pi+kk)][col_offset + r] = V2[jj*2+kk][r]
                            for kk in range(2):
                                for ii in range(2):
                                    W_full[(pi+kk)*n + (ni+ii)][col_offset + r] = W2[kk*2+ii][r]
                        col_offset += 7
                    else:
                        for ii in range(n_size):
                            for jj in range(m_size):
                                for kk in range(p_size):
                                    U_full[(ni+ii)*m + (mi+jj)][col_offset] = 1
                                    V_full[(mi+jj)*p + (pi+kk)][col_offset] = 1
                                    W_full[(pi+kk)*n + (ni+ii)][col_offset] = 1
                                    col_offset += 1
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(total_rank):
                        s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
            return
    
    # Try various block decompositions
    # n=4, m=5, p=6
    
    # Option 1: 2+2, 2+2+1, 2+2+2
    try_decomposition([0,2,4], [0,2,4,5], [0,2,4,6])
    
    # Option 2: 2+2, 1+2+2, 2+2+2  
    try_decomposition([0,2,4], [0,1,3,5], [0,2,4,6])
    
    # Option 3: 2+2, 2+1+2, 2+2+2
    try_decomposition([0,2,4], [0,2,3,5], [0,2,4,6])
    
    # Option 4: 2+2, 1+1+3, 2+2+2
    try_decomposition([0,2,4], [0,1,2,5], [0,2,4,6])
    
    # Option 5: 1+3, 2+2+1, 2+2+2
    try_decomposition([0,1,4], [0,2,4,5], [0,2,4,6])
    
    # Option 6: 1+3, 1+2+2, 2+2+2
    try_decomposition([0,1,4], [0,1,3,5], [0,2,4,6])
    
    # Option 7: 2+2, 2+2+1, 1+2+3
    try_decomposition([0,2,4], [0,2,4,5], [0,1,3,6])
    
    # Option 8: 2+2, 2+2+1, 3+3
    try_decomposition([0,2,4], [0,2,4,5], [0,3,6])
    
    # Option 9: 2+2, 1+1+1+2, 2+2+2
    try_decomposition([0,2,4], [0,1,2,3,5], [0,2,4,6])
    
    # Option 10: 2+2, 2+1+1+1, 2+2+2
    try_decomposition([0,2,4], [0,2,3,4,5], [0,2,4,6])
    
    # Additional options with more granular splits
    # Option 11: 2+2, 2+2+1, 2+2+1+1
    try_decomposition([0,2,4], [0,2,4,5], [0,2,4,5,6])
    
    # Option 12: 2+2, 1+2+2, 2+2+1+1
    try_decomposition([0,2,4], [0,1,3,5], [0,2,4,5,6])
    
    # Option 13: 2+2, 2+1+2, 2+2+1+1
    try_decomposition([0,2,4], [0,2,3,5], [0,2,4,5,6])
    
    # Option 14: 1+3, 2+2+1, 2+2+1+1
    try_decomposition([0,1,4], [0,2,4,5], [0,2,4,5,6])
    
    # Option 15: 2+2, 2+2+1, 1+3+2
    try_decomposition([0,2,4], [0,2,4,5], [0,1,4,6])
    
    # Option 16: 2+2, 2+2+1, 2+1+3
    try_decomposition([0,2,4], [0,2,4,5], [0,2,3,6])
    
    # Option 17: 2+2, 2+2+1, 1+2+3
    try_decomposition([0,2,4], [0,2,4,5], [0,1,3,6])
    
    # Option 18: 2+2, 2+2+1, 3+2+1
    try_decomposition([0,2,4], [0,2,4,5], [0,3,5,6])
    
    # Option 19: 2+2, 2+2+1, 2+3+1
    try_decomposition([0,2,4], [0,2,4,5], [0,2,5,6])
    
    # Option 20: 2+2, 2+2+1, 1+1+4
    try_decomposition([0,2,4], [0,2,4,5], [0,1,2,6])
    
    if best_rank < schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return best_result
# EVOLVE-BLOCK-END
