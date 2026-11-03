import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create trivial decomposition
    def trivial(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Helper to create Strassen 2x2x2 decomposition
    def strassen_222():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    def embed(U_sub, V_sub, W_sub, i_off, j_off, k_off, n_glob, m_glob, p_glob):
        n_sub, m_sub = U_sub.shape[0] // m_sub, U_sub.shape[0] % m_sub # This is wrong, need n_sub, m_sub
        # Actually, we know the shapes.
        # U_sub: (n_sub * m_sub, rank)
        # V_sub: (m_sub * p_sub, rank)
        # W_sub: (p_sub * n_sub, rank)
        # We need to pass n_sub, m_sub, p_sub
        pass

    # Let's do it manually for the specific case to avoid complex embedding logic errors
    # We need:
    # 1. M_217 (j=0)
    # 2. M_227 (j=1,2)
    #    M_227 = M_223 (k=0,1,2) + M_224 (k=3,4,5,6)
    #    M_223 = M_221 (k=0) + M_222 (k=1,2)
    #    M_224 = M_222 (k=3,4) + M_222 (k=5,6)
    
    U_full = []
    V_full = []
    W_full = []
    
    # 1. M_217 (Trivial)
    U_217, V_217, W_217 = trivial(2, 1, 7)
    # Embed j=0
    # U indices: i*3 + 0
    # V indices: 0*7 + k
    # W indices: k*2 + i
    U_part = np.zeros((6, 14))
    V_part = np.zeros((21, 14))
    W_part = np.zeros((14, 14))
    for r in range(14):
        for i in range(2):
            for k in range(7):
                # Trivial has U[i*1+0, r]=1, V[0*7+k, r]=1, W[k*2+i, r]=1
                # But trivial(2,1,7) U is 2x14. U[i, r]=1.
                # Global U idx: i*3 + 0
                U_part[i*3 + 0, r] = U_217[i, r]
                V_part[0*7 + k, r] = V_217[0*7 + k, r]
                W_part[k*2 + i, r] = W_217[k*2 + i, r]
    U_full.append(U_part)
    V_full.append(V_part)
    W_full.append(W_part)
    
    # 2. M_227
    # 2a. M_223 (j=1,2; k=0,1,2)
    # M_223 = M_221 (k=0) + M_222 (k=1,2)
    
    # 2a1. M_221 (j=1,2; k=0)
    U_221, V_221, W_221 = trivial(2, 2, 1) # Rank 4
    # Embed j_off=1, k_off=0
    U_part = np.zeros((6, 4))
    V_part = np.zeros((21, 4))
    W_part = np.zeros((14, 4))
    for r in range(4):
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U_part[i*3 + (1+j), r] = U_221[i*2 + j, r]
                    V_part[(1+j)*7 + (0+k), r] = V_221[j*1 + k, r]
                    W_part[(0+k)*2 + i, r] = W_221[k*2 + i, r]
    U_full.append(U_part)
    V_full.append(V_part)
    W_full.append(W_part)
    
    # 2a2. M_222 (j=1,2; k=1,2)
    U_222, V_222, W_222 = strassen_222() # Rank 7
    # Embed j_off=1, k_off=1
    U_part = np.zeros((6, 7))
    V_part = np.zeros((21, 7))
    W_part = np.zeros((14, 7))
    for r in range(7):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_part[i*3 + (1+j), r] = U_222[i*2 + j, r]
                    V_part[(1+j)*7 + (1+k), r] = V_222[j*2 + k, r]
                    W_part[(1+k)*2 + i, r] = W_222[k*2 + i, r]
    U_full.append(U_part)
    V_full.append(V_part)
    W_full.append(W_part)
    
    # 2b. M_224 (j=1,2; k=3,4,5,6)
    # M_224 = M_222 (k=3,4) + M_222 (k=5,6)
    
    # 2b1. M_222 (j=1,2; k=3,4)
    U_222, V_222, W_222 = strassen_222()
    U_part = np.zeros((6, 7))
    V_part = np.zeros((21, 7))
    W_part = np.zeros((14, 7))
    for r in range(7):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_part[i*3 + (1+j), r] = U_222[i*2 + j, r]
                    V_part[(1+j)*7 + (3+k), r] = V_222[j*2 + k, r]
                    W_part[(3+k)*2 + i, r] = W_222[k*2 + i, r]
    U_full.append(U_part)
    V_full.append(V_part)
    W_full.append(W_part)
    
    # 2b2. M_222 (j=1,2; k=5,6)
    U_222, V_222, W_222 = strassen_222()
    U_part = np.zeros((6, 7))
    V_part = np.zeros((21, 7))
    W_part = np.zeros((14, 7))
    for r in range(7):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_part[i*3 + (1+j), r] = U_222[i*2 + j, r]
                    V_part[(1+j)*7 + (5+k), r] = V_222[j*2 + k, r]
                    W_part[(5+k)*2 + i, r] = W_222[k*2 + i, r]
    U_full.append(U_part)
    V_full.append(V_part)
    W_full.append(W_part)
    
    U_final = np.hstack(U_full)
    V_final = np.hstack(V_full)
    W_final = np.hstack(W_full)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
