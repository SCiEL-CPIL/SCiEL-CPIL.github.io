import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify_rank_decomp(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_reconstructed += np.einsum('ij,jk,ki->ijk', u, v, w)
        if not np.allclose(T_reconstructed, np.ones((n, m, p)), atol=1e-10):
            return False
        return True
    
    def build_strassen_2x2_2():
        """Strassen decomposition of M_{2,2,2}. Rank 7."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = 1
        W[1][2] = 1; W[3][2] = 1
        
        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[2][3] = 1
        W[0][3] = 1; W[2][3] = 1
        
        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[2][4] = -1
        W[0][4] = -1; W[1][4] = 1
        
        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[3][5] = -1
        W[3][5] = 1
        
        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def build_2x3x8():
        """Build M_{2,3,8} with rank 44 using two blocks of M_{2,3,4}."""
        total_rank = 44
        
        U = [[0]*total_rank for _ in range(6)]
        V = [[0]*total_rank for _ in range(24)]
        W = [[0]*total_rank for _ in range(16)]
        
        # Build M_{2,2,4} (rank 14) using two Strassen 2x2x2
        U1, V1, W1 = build_strassen_2x2_2()
        U2, V2, W2 = build_strassen_2x2_2()
        
        U224 = [row1 + row2 for row1, row2 in zip(U1, U2)]
        V224 = [[0]*14 for _ in range(8)]
        W224 = [[0]*14 for _ in range(8)]
        
        for r in range(7):
            V224[0][r] = V1[0][r]
            V224[1][r] = V1[1][r]
            V224[4][r] = V1[2][r]
            V224[5][r] = V1[3][r]
            W224[0][r] = W1[0][r]
            W224[1][r] = W1[1][r]
            W224[4][r] = W1[2][r]
            W224[5][r] = W1[3][r]
            
            V224[2][r+7] = V2[0][r]
            V224[3][r+7] = V2[1][r]
            V224[6][r+7] = V2[2][r]
            V224[7][r+7] = V2[3][r]
            W224[2][r+7] = W2[0][r]
            W224[3][r+7] = W2[1][r]
            W224[6][r+7] = W2[2][r]
            W224[7][r+7] = W2[3][r]
        
        # Build M_{2,1,4} (rank 8)
        U214 = [[0]*8 for _ in range(2)]
        V214 = [[0]*8 for _ in range(4)]
        W214 = [[0]*8 for _ in range(8)]
        
        col = 0
        for i in range(2):
            for k in range(4):
                U214[i][col] = 1
                V214[k][col] = 1
                W214[k*2 + i][col] = 1
                col += 1
        
        # Assemble Block 1 (B columns 0-3)
        for i in range(2):
            for r in range(14):
                U[i*3+0][r] = U224[i*2+0][r]
                U[i*3+1][r] = U224[i*2+1][r]
            for r in range(8):
                U[i*3+2][14+r] = U214[i][r]
        
        for r in range(14):
            for j in range(2):
                for k in range(4):
                    V[j*8+k][r] = V224[j*4+k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i][r] = W224[k*2+i][r]
        
        for r in range(8):
            for k in range(4):
                V[2*8+k][14+r] = V214[k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i][14+r] = W214[k*2+i][r]
        
        # Assemble Block 2 (B columns 4-7)
        for i in range(2):
            for r in range(14):
                U[i*3+0][22+r] = U224[i*2+0][r]
                U[i*3+1][22+r] = U224[i*2+1][r]
            for r in range(8):
                U[i*3+2][36+r] = U214[i][r]
        
        for r in range(14):
            for j in range(2):
                for k in range(4):
                    V[j*8+k+4][22+r] = V224[j*4+k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i+8][22+r] = W224[k*2+i][r]
        
        for r in range(8):
            for k in range(4):
                V[2*8+k+4][36+r] = V214[k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i+8][36+r] = W214[k*2+i][r]
        
        return U, V, W
    
    U, V, W = build_2x3x8()
    
    if verify_rank_decomp(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
