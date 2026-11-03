"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def _4x4x4_strassen_full():
        """4x4x4 matrix mult with rank 56 using Strassen on 2x2 blocks."""
        U = [[0]*56 for _ in range(16)]
        V = [[0]*56 for _ in range(16)]  
        W = [[0]*56 for _ in range(16)]
        
        def strassen_block(u_off, v_off, w_off, m_off):
            U[u_off+0][m_off+0] += 1; U[u_off+3][m_off+0] += 1
            V[v_off+0][m_off+0] += 1; V[v_off+3][m_off+0] += 1
            W[w_off+0][m_off+0] += 1; W[w_off+3][m_off+0] += 1
            U[u_off+2][m_off+1] += 1; U[u_off+3][m_off+1] += 1
            V[v_off+0][m_off+1] += 1
            W[w_off+2][m_off+1] += 1
            U[u_off+0][m_off+2] += 1
            V[v_off+1][m_off+2] += 1; V[v_off+3][m_off+2] -= 1
            W[w_off+1][m_off+2] += 1; W[w_off+3][m_off+2] += 1
            U[u_off+3][m_off+3] += 1
            V[v_off+2][m_off+3] += 1; V[v_off+0][m_off+3] -= 1
            W[w_off+0][m_off+3] += 1; W[w_off+2][m_off+3] += 1
            U[u_off+0][m_off+4] += 1; U[u_off+1][m_off+4] += 1
            V[v_off+3][m_off+4] += 1
            W[w_off+0][m_off+4] -= 1; W[w_off+3][m_off+4] += 1
            U[u_off+2][m_off+5] += 1; U[u_off+0][m_off+5] -= 1
            V[v_off+0][m_off+5] += 1; V[v_off+1][m_off+5] += 1
            W[w_off+3][m_off+5] += 1
            U[u_off+1][m_off+6] += 1; U[u_off+3][m_off+6] -= 1
            V[v_off+2][m_off+6] += 1; V[v_off+3][m_off+6] += 1
            W[w_off+1][m_off+6] += 1; W[w_off+3][m_off+6] -= 1
        
        strassen_block(0, 0, 0, 0)
        strassen_block(2, 8, 0, 7)
        strassen_block(0, 4, 4, 14)
        strassen_block(2, 12, 4, 21)
        strassen_block(8, 0, 8, 28)
        strassen_block(10, 8, 8, 35)
        strassen_block(8, 4, 12, 42)
        strassen_block(10, 12, 12, 49)
        
        return U, V, W
    
    U44, V44, W44 = _4x4x4_strassen_full()
    
    total_rank = 56 + 56 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    for i in range(16):
        for c in range(56):
            U[i][c] = U44[i][c]
            V[i][c] = V44[i][c]
            W[i][c] = W44[i][c]
    
    for i in range(16):
        for c in range(56):
            U[i][56+c] = U44[i][c]
            V[i][56+c] = V44[i][c]
            W[i+16][56+c] = W44[i][c]
    
    for i in range(4):
        for j in range(4):
            c = 112 + i*4 + j
            U[i*4+j][c] = 1
            V[j*9+8][c] = 1
            W[32+i][c] = 1
    
    return {"U": U, "V": V, "W": W}
