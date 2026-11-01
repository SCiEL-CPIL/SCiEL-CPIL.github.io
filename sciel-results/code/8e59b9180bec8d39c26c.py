"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    
    Strategy:
    Split the inner dimension m=3 into m1=2 and m2=1.
    The multiplication is A (2x3) * B (3x7) = C (2x7).
    C = A[:, :2] * B[:2, :] + A[:, 2:] * B[2:, :]
    
    Part 1: A1 (2x2) * B1 (2x7).
    We can further split the output dimension p=7 into p1=2, p2=2, p3=2, p4=1.
    B1 = [B11 (2x2), B12 (2x2), B13 (2x2), B14 (2x1)].
    C1 = [C11 (2x2), C12 (2x2), C13 (2x2), C14 (2x1)].
    
    For each 2x2 block multiplication (e.g., C11 = A1 * B11), we use the Strassen algorithm (rank 7).
    For the 2x1 block multiplication (C14 = A1 * B14), we use schoolbook (rank 4).
    
    Part 2: A2 (2x1) * B2 (1x7).
    This is a vector-matrix product. We use schoolbook (rank 14).
    
    Total Rank = 3 * 7 (Strassen for 2x2x2) + 1 * 4 (Schoolbook for 2x2x1) + 14 (Schoolbook for 2x1x7)
             = 21 + 4 + 14 = 39.
    """
    n, m, p = DIMENSIONS
    
    # Initialize factor matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the decomposition block by block and concatenate the columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a block decomposition
    # block_U: vector of length n*m_block (indices i*m_block+j)
    # block_V: vector of length m_block*p_block
    # block_W: vector of length p_block*n
    # Note: The global indices need to be mapped.
    
    def add_block(n_block, m_block, p_block, U_block, V_block, W_block, offset_i, offset_j, offset_k):
        """
        Add a block decomposition to the global lists.
        U_block is a list of tuples (idx, val) or similar? 
        Actually, let's just create dense vectors for the block and place them.
        """
        rank_block = len(U_block)
        
        # Create global U, V, W vectors for this block
        U_global = [0.0] * (n * m)
        V_global = [0.0] * (m * p)
        W_global = [0.0] * (p * n)
        
        for r in range(rank_block):
            # U indices: i in [0, n_block), j in [0, m_block)
            # Global i = offset_i + i, Global j = offset_j + j
            # Index in U_global: (offset_i + i) * m + (offset_j + j)
            
            # V indices: j in [0, m_block), k in [0, p_block)
            # Global j = offset_j + j, Global k = offset_k + k
            # Index in V_global: (offset_j + j) * p + (offset_k + k)
            
            # W indices: k in [0, p_block), i in [0, n_block)
            # Global k = offset_k + k, Global i = offset_i + i
            # Index in W_global: (offset_k + k) * n + (offset_i + i)
            
            # Extract values from block factors
            # U_block[r] is a vector of length n_block * m_block
            # V_block[r] is a vector of length m_block * p_block
            # W_block[r] is a vector of length p_block * n_block
            
            for i in range(n_block):
                for j in range(m_block):
                    u_val = U_block[r][i * m_block + j]
                    if u_val != 0:
                        U_global[(offset_i + i) * m + (offset_j + j)] = u_val
                        
            for j in range(m_block):
                for k in range(p_block):
                    v_val = V_block[r][j * p_block + k]
                    if v_val != 0:
                        V_global[(offset_j + j) * p + (offset_k + k)] = v_val
                        
            for k in range(p_block):
                for i in range(n_block):
                    w_val = W_block[r][k * n_block + i]
                    if w_val != 0:
                        W_global[(offset_k + k) * n + (offset_i + i)] = w_val
                        
        U_cols.append(U_global)
        V_cols.append(V_global)
        W_cols.append(W_global)

    # --- Part 1: A1 (2x2) * B1 (2x7) ---
    # Split p=7 into 2, 2, 2, 1
    # Offsets:
    # i_offset = 0, j_offset = 0
    # k_offsets: 0, 2, 4, 6
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2():
        U = []
        V = []
        W = []
        
        # M1 = (a+d)(e+h)
        # U: a=1, d=1 -> [1,0,0,1] (indices: 00, 01, 10, 11)
        # V: e=1, h=1 -> [1,0,0,1]
        # W: c11=1, c22=1 -> [1,0,0,1]
        U.append([1, 0, 0, 1])
        V.append([1, 0, 0, 1])
        W.append([1, 0, 0, 1])
        
        # M2 = (c+d)e
        # U: c=1, d=1 -> [0,0,1,1]
        # V: e=1 -> [1,0,0,0]
        # W: c21=1 -> [0,0,1,0]
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([0, 0, 1, 0])
        
        # M3 = a(f-h)
        # U: a=1 -> [1,0,0,0]
        # V: f=1, h=-1 -> [0,1,0,-1]
        # W: c12=1 -> [0,1,0,0]
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        W.append([0, 1, 0, 0])
        
        # M4 = d(g-e)
        # U: d=1 -> [0,0,0,1]
        # V: g=1, e=-1 -> [-1,0,1,0]
        # W: c11=1, c21=1 -> [1,0,1,0]
        U.append([0, 0, 0, 1])
        V.append([-1, 0, 1, 0])
        W.append([1, 0, 1, 0])
        
        # M5 = (a+b)h
        # U: a=1, b=1 -> [1,1,0,0]
        # V: h=1 -> [0,0,0,1]
        # W: c12=1, c11=-1 -> [-1,1,0,0]
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([-1, 1, 0, 0])
        
        # M6 = (c-a)(e+f)
        # U: c=1, a=-1 -> [-1,0,1,0]
        # V: e=1, f=1 -> [1,1,0,0]
        # W: c22=1 -> [0,0,0,1]
        U.append([-1, 0, 1, 0])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 0, 1])
        
        # M7 = (b-d)(g+h)
        # U: b=1, d=-1 -> [0,1,0,-1]
        # V: g=1, h=1 -> [0,0,1,1]
        # W: c11=1, c22=-1 -> [1,0,0,-1]
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        W.append([1, 0, 0, -1])
        
        return U, V, W

    # Schoolbook for 2x2x1
    # A (2x2) * v (2x1) -> w (2x1)
    # w0 = a*v0 + b*v1
    # w1 = c*v0 + d*v1
    # Rank 4
    def schoolbook_2x2x1():
        U = []
        V = []
        W = []
        
        # a*v0
        U.append([1, 0, 0, 0])
        V.append([1, 0])
        W.append([1, 0])
        
        # b*v1
        U.append([0, 1, 0, 0])
        V.append([0, 1])
        W.append([1, 0])
        
        # c*v0
        U.append([0, 0, 1, 0])
        V.append([1, 0])
        W.append([0, 1])
        
        # d*v1
        U.append([0, 0, 0, 1])
        V.append([0, 1])
        W.append([0, 1])
        
        return U, V, W

    # Schoolbook for 2x1x7
    # A (2x1) * B (1x7) -> C (2x7)
    # c_i,k = a_i * b_k
    # Rank 14
    def schoolbook_2x1x7():
        U = []
        V = []
        W = []
        for i in range(2):
            for k in range(7):
                u = [0.0] * 2
                u[i] = 1.0
                v = [0.0] * 7
                v[k] = 1.0
                w = [0.0] * 14 # 7*2
                w[k * 2 + i] = 1.0
                U.append(u)
                V.append(v)
                W.append(w)
        return U, V, W

    # Add the three Strassen blocks (2x2x2)
    # k_offsets: 0, 2, 4
    for k_off in [0, 2, 4]:
        U_s, V_s, W_s = strassen_2x2x2()
        add_block(2, 2, 2, U_s, V_s, W_s, 0, 0, k_off)
        
    # Add the one Schoolbook 2x2x1 block
    # k_offset: 6
    U_sb, V_sb, W_sb = schoolbook_2x2x1()
    add_block(2, 2, 1, U_sb, V_sb, W_sb, 0, 0, 6)
    
    # Add the Schoolbook 2x1x7 block
    # j_offset: 2, i_offset: 0, k_offset: 0
    U_s2, V_s2, W_s2 = schoolbook_2x1x7()
    add_block(2, 1, 7, U_s2, V_s2, W_s2, 0, 2, 0)
    
    # Concatenate
    U = np.array(U_cols).T # Shape (rank, n*m) -> (n*m, rank)
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Convert to lists of lists for the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
