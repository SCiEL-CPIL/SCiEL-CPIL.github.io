import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will decompose the tensor into blocks.
    # Block 1: i in [0,2], j in [0,2], k in [0,2] -> 3x3x3
    # Block 2: i in [0,2], j in [3,4], k in [0,2] -> 3x2x3
    # Block 3: i in [0,2], j in [0,2], k in [3,6] -> 3x3x4
    # Block 4: i in [0,2], j in [3,4], k in [3,6] -> 3x2x4
    
    # For each block, we find a decomposition.
    # We use a helper function to find a low-rank decomposition of a small multiplication tensor.
    
    def get_decomp(n_sub, m_sub, p_sub):
        # Returns U, V, W for n_sub x m_sub x p_sub multiplication
        # U: (n_sub*m_sub, rank)
        # V: (m_sub*p_sub, rank)
        # W: (p_sub*n_sub, rank)
        
        # Try to find a low rank decomposition using a greedy approach
        # Target: exact decomposition with rational entries
        # For small sizes, we can try to find a basis.
        
        # Schoolbook
        rank_sb = n_sub * m_sub * p_sub
        U_sb = np.zeros((n_sub * m_sub, rank_sb))
        V_sb = np.zeros((m_sub * p_sub, rank_sb))
        W_sb = np.zeros((p_sub * n_sub, rank_sb))
        
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_sb[i * m_sub + j, col] = 1.0
                    V_sb[j * p_sub + k, col] = 1.0
                    W_sb[k * n_sub + i, col] = 1.0
                    col += 1
        
        # If the block is 3x3x3, try to find a better decomposition
        if n_sub == 3 and m_sub == 3 and p_sub == 3:
            # Try greedy search for rank < 27
            # We'll skip the complex search for now and just use schoolbook if no hard-coded solution is available
            # But we can try a simple Strassen-like embedding for 2x2x2 inside 3x3x3?
            # For now, return schoolbook to ensure correctness.
            # In a future iteration, we might hard-code the 23-rank solution.
            pass
            
        return U_sb, V_sb, W_sb, rank_sb

    # Get decompositions for blocks
    U1, V1, W1, r1 = get_decomp(3, 3, 3)
    U2, V2, W2, r2 = get_decomp(3, 2, 3)
    U3, V3, W3, r3 = get_decomp(3, 3, 4)
    U4, V4, W4, r4 = get_decomp(3, 2, 4)
    
    total_rank = r1 + r2 + r3 + r4
    
    # Assemble full U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1: i 0-2, j 0-2, k 0-2
    # U indices: i*5 + j. j in 0-2.
    # V indices: j*7 + k. j in 0-2, k in 0-2.
    # W indices: k*3 + i. k in 0-2, i in 0-2.
    
    col_offset = 0
    for r in range(r1):
        for i in range(3):
            for j in range(3):
                U[i * m + j, col_offset + r] = U1[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V[j * p + k, col_offset + r] = V1[j * 3 + k, r]
        for k in range(3):
            for i in range(3):
                W[k * n + i, col_offset + r] = W1[k * 3 + i, r]
    col_offset += r1
    
    # Block 2: i 0-2, j 3-4, k 0-2
    # U indices: i*5 + j. j in 3-4.
    # V indices: j*7 + k. j in 3-4, k in 0-2.
    # W indices: k*3 + i. k in 0-2, i in 0-2.
    
    for r in range(r2):
        for i in range(3):
            for j in range(2):
                U[i * m + (3 + j), col_offset + r] = U2[i * 2 + j, r]
        for j in range(2):
            for k in range(3):
                V[(3 + j) * p + k, col_offset + r] = V2[j * 3 + k, r]
        for k in range(3):
            for i in range(3):
                W[k * n + i, col_offset + r] = W2[k * 3 + i, r]
    col_offset += r2
    
    # Block 3: i 0-2, j 0-2, k 3-6
    # U indices: i*5 + j. j in 0-2.
    # V indices: j*7 + k. j in 0-2, k in 3-6.
    # W indices: k*3 + i. k in 3-6, i in 0-2.
    
    for r in range(r3):
        for i in range(3):
            for j in range(3):
                U[i * m + j, col_offset + r] = U3[i * 3 + j, r]
        for j in range(3):
            for k in range(4):
                V[j * p + (3 + k), col_offset + r] = V3[j * 4 + k, r]
        for k in range(4):
            for i in range(3):
                W[(3 + k) * n + i, col_offset + r] = W3[k * 3 + i, r]
    col_offset += r3
    
    # Block 4: i 0-2, j 3-4, k 3-6
    # U indices: i*5 + j. j in 3-4.
    # V indices: j*7 + k. j in 3-4, k in 3-6.
    # W indices: k*3 + i. k in 3-6, i in 0-2.
    
    for r in range(r4):
        for i in range(3):
            for j in range(2):
                U[i * m + (3 + j), col_offset + r] = U4[i * 2 + j, r]
        for j in range(2):
            for k in range(4):
                V[(3 + j) * p + (3 + k), col_offset + r] = V4[j * 4 + k, r]
        for k in range(4):
            for i in range(3):
                W[(3 + k) * n + i, col_offset + r] = W4[k * 3 + i, r]
    col_offset += r4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
