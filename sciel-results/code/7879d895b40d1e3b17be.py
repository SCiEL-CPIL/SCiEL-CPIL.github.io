import numpy as np

DIMENSIONS = [3, 4, 8]

def get_strassen_2x2x2():
    # Returns U, V, W for 2x2x2 with rank 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1 # A11, A22
    V[0, 0] = 1; V[3, 0] = 1 # B11, B22
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22 (Wait, W is k*n+i)
    # W index: k*2+i.
    # C11 is i=0, k=0 -> 0.
    # C22 is i=1, k=1 -> 1*2+1=3.
    # C12 is i=0, k=1 -> 1*2+0=2.
    # C21 is i=1, k=0 -> 0*2+1=1.
    
    # Let's map W indices carefully.
    # W[k*2+i]
    # M1 contributes to C11 and C22.
    # C11 (i=0, k=0) -> W[0]
    # C22 (i=1, k=1) -> W[3]
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21 (i=1, j=0) -> U[1*2+0]=U[2]
    # A22 (i=1, j=1) -> U[1*2+1]=U[3]
    # B11 (j=0, k=0) -> V[0*2+0]=V[0]
    # C21 (i=1, k=0) -> W[0*2+1]=W[1]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = A11(B21-B22)
    # A11 -> U[0]
    # B21 (j=1, k=0) -> V[1*2+0]=V[2]
    # B22 (j=1, k=1) -> V[1*2+1]=V[3]
    # C12 (i=0, k=1) -> W[1*2+0]=W[2]
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A22(B12-B11)
    # A22 -> U[3]
    # B12 (j=0, k=1) -> V[0*2+1]=V[1]
    # B11 (j=0, k=0) -> V[0]
    # C21 (i=1, k=0) -> W[1]
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11 -> U[0]
    # A12 (i=0, j=1) -> U[0*2+1]=U[1]
    # B22 -> V[3]
    # C12 (i=0, k=1) -> W[2]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21 -> U[2]
    # A11 -> U[0]
    # B11 -> V[0]
    # B12 -> V[1]
    # C22 (i=1, k=1) -> W[3]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12 -> U[1]
    # A22 -> U[3]
    # B21 -> V[2]
    # B22 -> V[3]
    # C11 (i=0, k=0) -> W[0]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W for the full problem
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    
    # Split m=4 into two m=2 blocks.
    # Block 1: j=0,1. Block 2: j=2,3.
    # For each block, we solve 3x2x8.
    
    # We will accumulate U, V, W columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    for block_idx in range(2):
        # Solve 3x2x8 for this block
        # Split n=3 into n=2 and n=1.
        
        # Part 1: 2x2x8
        # Split p=8 into four p=2 blocks.
        for p_block in range(4):
            # Get Strassen factors for 2x2x2
            Us, Vs, Ws = get_strassen_2x2x2()
            
            # Map indices to full problem
            # n=2 (i=0,1), m=2 (j=0,1 relative to block), p=2 (k=0,1 relative to block)
            
            # U mapping:
            # Full U index: i * 4 + j_full
            # j_full = block_idx * 2 + j_rel
            # Us row index: i * 2 + j_rel
            
            U_block = np.zeros((12, 7)) # 3*4
            for i in range(2):
                for j_rel in range(2):
                    j_full = block_idx * 2 + j_rel
                    u_idx = i * 4 + j_full
                    u_s_idx = i * 2 + j_rel
                    U_block[u_idx, :] = Us[u_s_idx, :]
            
            # V mapping:
            # Full V index: j_full * 8 + k_full
            # k_full = p_block * 2 + k_rel
            # Vs row index: j_rel * 2 + k_rel
            
            V_block = np.zeros((32, 7)) # 4*8
            for j_rel in range(2):
                j_full = block_idx * 2 + j_rel
                for k_rel in range(2):
                    k_full = p_block * 2 + k_rel
                    v_idx = j_full * 8 + k_full
                    v_s_idx = j_rel * 2 + k_rel
                    V_block[v_idx, :] = Vs[v_s_idx, :]
            
            # W mapping:
            # Full W index: k_full * 3 + i
            # i is 0,1 for this part.
            # Ws row index: k_rel * 2 + i
            
            W_block = np.zeros((24, 7)) # 8*3
            for k_rel in range(2):
                k_full = p_block * 2 + k_rel
                for i in range(2):
                    w_idx = k_full * 3 + i
                    w_s_idx = k_rel * 2 + i
                    W_block[w_idx, :] = Ws[w_s_idx, :]
            
            U_cols.append(U_block)
            V_cols.append(V_block)
            W_cols.append(W_block)
        
        # Part 2: 1x2x8
        # Schoolbook for i=2 (the third row)
        # n=1 (i=2), m=2 (j=0,1 relative to block), p=8
        # Rank 16
        U_part = np.zeros((12, 16))
        V_part = np.zeros((32, 16))
        W_part = np.zeros((24, 16))
        
        col = 0
        for j_rel in range(2):
            j_full = block_idx * 2 + j_rel
            for k_full in range(8):
                # U: i=2, j=j_full
                u_idx = 2 * 4 + j_full
                U_part[u_idx, col] = 1
                
                # V: j=j_full, k=k_full
                v_idx = j_full * 8 + k_full
                V_part[v_idx, col] = 1
                
                # W: k=k_full, i=2
                w_idx = k_full * 3 + 2
                W_part[w_idx, col] = 1
                
                col += 1
        
        U_cols.append(U_part)
        V_cols.append(V_part)
        W_cols.append(W_part)
    
    U = np.hstack(U_cols)
    V = np.hstack(V_cols)
    W = np.hstack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
