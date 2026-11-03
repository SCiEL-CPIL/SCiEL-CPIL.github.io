import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # We will construct a decomposition of rank 84
    # Split into 4x4x4 (rank 56) and 4x4x2 (rank 28)
    
    # Helper: Strassen 2x2x2
    # Indices for 2x2x2: i in 0,1; j in 0,1; k in 0,1
    # U size 4x7, V size 4x7, W size 4x7
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # Wait, C00 = M1...
        # Let's trace C indices.
        # C00 gets M1, M4, -M5, M7
        # C01 gets M3, M5
        # C10 gets M2, M4
        # C11 gets M1, -M2, M3, M6
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C10
        W[3, 1] = -1 # C11
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = 1
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C00
        W[1, 4] = 1  # C01
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C11
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C00
        
        return U, V, W

    U_222, V_222, W_222 = strassen_222()

    # Build 4x4x4 using 8 blocks of 2x2x2
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Block (bi, bj, bk) where bi, bj, bk in {0,1}
    # A block is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block is at rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # C block is at rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    
    rank_444 = 0
    U_444_list = []
    V_444_list = []
    W_444_list = []
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Extract U, V, W for this block
                # U indices: i in 2*bi..2*bi+1, j in 2*bj..2*bj+1
                # V indices: j in 2*bj..2*bj+1, k in 2*bk..2*bk+1
                # W indices: k in 2*bk..2*bk+1, i in 2*bi..2*bi+1
                
                U_block = np.zeros((16, 7))
                V_block = np.zeros((16, 7))
                W_block = np.zeros((16, 7))
                
                for r in range(7):
                    for ii in range(2):
                        for jj in range(2):
                            u_val = U_222[ii, jj, r] # Wait, U_222 is 4x7. Index is i*2+j?
                            # In strassen_222, U is 4x7.
                            # Index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                            u_idx = ii * 2 + jj
                            U_block[(2*bi + ii) * 4 + (2*bj + jj), r] = U_222[u_idx, r]
                            
                            v_idx = jj * 2 + (bk*2 + 0) # No, V is j*p+k.
                            # In 2x2x2, V is 4x7. Index j*2+k.
                            # Here j is local 0..1, k is local 0..1.
                            # Global j = 2*bj + jj. Global k = 2*bk + kk.
                            # V index = j*4 + k? No, m=2, p=2. V size 4.
                            # In 4x4x4, m=4, p=4. V size 16.
                            # V index = j*4 + k.
                            
                            # Let's re-do the mapping carefully.
                            
                # This manual mapping is tedious and error prone.
                # Let's use a generic tiling function.
                
                pass

    # Generic Tiling Function
    def tile_decomp(U_small, V_small, W_small, n_small, m_small, p_small, n_big, m_big, p_big):
        """
        Tiles a decomposition of n_small x m_small x p_small into n_big x m_big x p_big.
        Assumes n_big is multiple of n_small, etc.
        U_small: (n_small*m_small, rank)
        V_small: (m_small*p_small, rank)
        W_small: (p_small*n_small, rank)
        """
        rank_small = U_small.shape[1]
        rank_big = rank_small * (n_big//n_small) * (m_big//m_small) * (p_big//p_small)
        
        U_big = np.zeros((n_big*m_big, rank_big))
        V_big = np.zeros((m_big*p_big, rank_big))
        W_big = np.zeros((p_big*n_big, rank_big))
        
        col = 0
        for i_block in range(n_big//n_small):
            for j_block in range(m_big//m_small):
                for k_block in range(p_big//p_small):
                    # Map indices
                    # U: i in n_small, j in m_small -> global i, j
                    # V: j in m_small, k in p_small -> global j, k
                    # W: k in p_small, i in n_small -> global k, i
                    
                    for r in range(rank_small):
                        for i in range(n_small):
                            for j in range(m_small):
                                u_val = U_small[i*m_small + j, r]
                                if u_val != 0:
                                    U_big[(i_block*n_small + i)*m_big + (j_block*m_small + j), col] = u_val
                            
                            for k in range(p_small):
                                v_val = V_small[j*p_small + k, r]
                                if v_val != 0:
                                    V_big[(j_block*m_small + j)*p_big + (k_block*p_small + k), col] = v_val
                            
                            for k in range(p_small):
                                w_val = W_small[k*n_small + i, r]
                                if w_val != 0:
                                    W_big[(k_block*p_small + k)*n_big + (i_block*n_small + i), col] = w_val
                                    
                    col += rank_small
                    
        return U_big, V_big, W_big

    # 1. 4x4x4 part
    U_444, V_444, W_444 = tile_decomp(U_222, V_222, W_222, 2, 2, 2, 4, 4, 4)
    
    # 2. 4x4x2 part
    # This is not a simple tiling of 2x2x2 into 4x4x2?
    # 4x4x2 can be tiled from 2x2x1? No, we don't have 2x2x1.
    # We can tile 2x2x2 into 4x4x2?
    # n=4, m=4, p=2.
    # n_small=2, m_small=2, p_small=2?
    # If p_small=2, then p_big=2 is 1 block.
    # n_big=4 is 2 blocks.
    # m_big=4 is 2 blocks.
    # So we tile 2x2x2 into 4x4x2 by repeating in n and m, but only 1 block in p.
    
    U_442, V_442, W_442 = tile_decomp(U_222, V_222, W_222, 2, 2, 2, 4, 4, 2)
    
    # Combine
    # U: 16 x 56 (from 444) and 16 x 28 (from 442)
    # V: 16 x 56 (from 444, p=4) and 8 x 28 (from 442, p=2)
    #    Full V is 24 x 84.
    #    V_444 occupies k=0..3.
    #    V_442 occupies k=4..5.
    # W: 16 x 56 (from 444, p=4) and 8 x 28 (from 442, p=2)
    #    Full W is 24 x 84.
    #    W_444 occupies k=0..3.
    #    W_442 occupies k=4..5.
    
    U_full = np.hstack([U_444, U_442])
    
    V_full = np.zeros((24, 84))
    # V_444 is 16x56. Place in rows 0..15 (k=0..3).
    V_full[0:16, 0:56] = V_444
    # V_442 is 8x28. Place in rows 16..23 (k=4..5).
    # Wait, V index is j*p+k.
    # For 4x4x4: j in 0..3, k in 0..3. Index j*4+k.
    # For 4x4x2: j in 0..3, k in 0..1. Index j*2+k?
    # No, in the full problem, p=6.
    # V index is j*6+k.
    # For k=0..3 (from 444 part): j*6+k.
    # For k=4..5 (from 442 part): j*6+k.
    
    # My tile_decomp produces V with p_small stride.
    # For 444: p_small=2, p_big=4. V index j*4+k.
    # We need to map j*4+k to j*6+k for k<4.
    # For 442: p_small=2, p_big=2. V index j*2+k.
    # We need to map j*2+k to j*6+(k+4) for k<2.
    
    # Let's re-map V_444 and V_442 to the full V.
    
    # V_444 (16x56)
    for j in range(4):
        for k in range(4):
            for r in range(56):
                val = V_444[j*4+k, r]
                if val != 0:
                    V_full[j*6+k, r] = val
                    
    # V_442 (8x28)
    for j in range(4):
        for k in range(2):
            for r in range(28):
                val = V_442[j*2+k, r]
                if val != 0:
                    V_full[j*6+(k+4), 56+r] = val

    # W_444 (16x56)
    W_full = np.zeros((24, 84))
    for k in range(4):
        for i in range(4):
            for r in range(56):
                val = W_444[k*4+i, r]
                if val != 0:
                    W_full[k*4+i, r] = val # W index is k*n+i. n=4.
                    
    # W_442 (8x28)
    for k in range(2):
        for i in range(4):
            for r in range(28):
                val = W_442[k*2+i, r]
                if val != 0:
                    W_full[(k+4)*4+i, 56+r] = val

    return {"U": U_full, "V": V_full, "W": W_full}

