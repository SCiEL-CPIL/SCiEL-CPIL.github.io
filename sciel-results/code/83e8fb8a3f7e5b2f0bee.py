import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 140
    n, m, p = 2, 3, 8
    
    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    def build_2x2x4_rank13():
        """Rank-13 decomposition of M_{2,2,4}."""
        rank = 13
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[5][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[4][1] = 1
        
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = 1
        W[1][2] = 1; W[5][2] = 1
        
        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[4][3] = 1
        W[0][3] = 1; W[4][3] = 1
        
        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[4][4] = -1
        W[0][4] = -1; W[1][4] = 1
        
        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[5][5] = -1
        W[5][5] = 1
        
        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[5][6] = 1
        W[0][6] = 1
        
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[6][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[6][8] = 1
        
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = 1
        W[5][9] = 1; W[7][9] = 1
        
        U[0][10] = 1; U[2][10] = -1
        V[3][10] = 1; V[6][10] = 1
        W[4][10] = 1; W[6][10] = 1
        
        U[1][11] = 1; U[2][11] = 1
        V[2][11] = 1; V[6][11] = -1
        W[4][11] = -1; W[5][11] = 1
        
        U[1][12] = -1; U[3][12] = 1
        V[3][12] = 1; V[7][12] = -1
        W[7][12] = 1
        
        return U, V, W
    
    def build_1plus2_split():
        """Split m=3 as 1+2, use rank-13 M_{2,2,4} for the 2x2 part."""
        n, m, p = 2, 3, 8
        rank1 = 16
        rank2 = 26
        total_rank = rank1 + rank2
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        col = 0
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        U224, V224, W224 = build_2x2x4_rank13()
        
        for r in range(13):
            for i in range(2):
                U[i*3+1][col+r] = U224[i*2+0][r]
                U[i*3+2][col+r] = U224[i*2+1][r]
            for j_local in range(2):
                for k_local in range(4):
                    V[(1+j_local)*8 + k_local][col+r] = V224[j_local*4+k_local][r]
            for k_local in range(4):
                for i in range(2):
                    W[k_local*2+i][col+r] = W224[k_local*2+i][r]
        col += 13
        
        for r in range(13):
            for i in range(2):
                U[i*3+1][col+r] = U224[i*2+0][r]
                U[i*3+2][col+r] = U224[i*2+1][r]
            for j_local in range(2):
                for k_local in range(4):
                    V[(1+j_local)*8 + (4+k_local)][col+r] = V224[j_local*4+k_local][r]
            for k_local in range(4):
                for i in range(2):
                    W[(4+k_local)*2+i][col+r] = W224[k_local*2+i][r]
        col += 13
        
        return U, V, W
    
    def build_strassen_split():
        """Split m=3 as 1+2, use Strassen for 2x2x2 blocks."""
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        for block in range(4):
            b00 = 1*p + block*2
            b01 = 1*p + block*2 + 1
            b10 = 2*p + block*2
            b11 = 2*p + block*2 + 1
            
            c00 = (block*2)*n + 0
            c01 = (block*2+1)*n + 0
            c10 = (block*2)*n + 1
            c11 = (block*2+1)*n + 1
            
            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[c00][col] = 1; W[c11][col] = 1
            col += 1
            
            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1; W[c11][col] = -1
            col += 1
            
            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[c01][col] = 1; W[c11][col] = 1
            col += 1
            
            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[c00][col] = 1; W[c10][col] = 1
            col += 1
            
            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1; W[c01][col] = 1
            col += 1
            
            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[c11][col] = 1
            col += 1
            
            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[c00][col] = 1
            col += 1
        
        return U, V, W
    
    def build_2plus1_split():
        """Split m=3 as 2+1, use Strassen for 2x2x2 blocks."""
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        
        # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x8) - 4 Strassen blocks
        for block in range(4):
            b00 = 0*p + block*2
            b01 = 0*p + block*2 + 1
            b10 = 1*p + block*2
            b11 = 1*p + block*2 + 1
            
            c00 = (block*2)*n + 0
            c01 = (block*2+1)*n + 0
            c10 = (block*2)*n + 1
            c11 = (block*2+1)*n + 1
            
            U[0][col] = 1; U[3][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[c00][col] = 1; W[c11][col] = 1
            col += 1
            
            U[2][col] = 1; U[3][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1; W[c11][col] = -1
            col += 1
            
            U[0][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[c01][col] = 1; W[c11][col] = 1
            col += 1
            
            U[3][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[c00][col] = 1; W[c10][col] = 1
            col += 1
            
            U[0][col] = 1; U[1][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1; W[c01][col] = 1
            col += 1
            
            U[2][col] = 1; U[0][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[c11][col] = 1
            col += 1
            
            U[1][col] = 1; U[3][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[c00][col] = 1
            col += 1
        
        # Part 2: A[:,2] (2x1) * B[2,:] (1x8) - schoolbook
        for i in range(n):
            for k in range(p):
                U[i*m + 2][col] = 1
                V[2*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        return U, V, W
    
    best_result = None
    best_rank = float('inf')
    
    candidates = [
        ("1+2 split (rank 42)", build_1plus2_split, 42),
        ("1+2 Strassen (rank 44)", build_strassen_split, 44),
        ("2+1 Strassen (rank 44)", build_2plus1_split, 44),
    ]
    
    for name, builder, expected_rank in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W = builder()
            if verify(U, V, W, n, m, p) and expected_rank < best_rank:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = expected_rank
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    rank_fallback = n*m*p
    U_f = [[0]*rank_fallback for _ in range(n*m)]
    V_f = [[0]*rank_fallback for _ in range(m*p)]
    W_f = [[0]*rank_fallback for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i*m+j][col] = 1
                V_f[j*p+k][col] = 1
                W_f[k*n+i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
