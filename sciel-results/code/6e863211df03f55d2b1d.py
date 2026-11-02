import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Helper to create basic decompositions
    def get_basic(n, m, p):
        # Returns U, V, W for M_{n,m,p} schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    def get_strassen_222():
        # Rank 7 decomposition for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices:
        # A: 0,1,2,3 (00,01,10,11)
        # B: 0,1,2,3 (00,01,10,11)
        # C: 0,1,2,3 (00,01,10,11)
        
        # Strassen
        # m1 = a00 * (b01 + b11)
        # m2 = (a00 + a10) * b00
        # m3 = (a01 + a11) * b10
        # m4 = a11 * (b00 - b10)
        # m5 = (a00 + a01) * (b10 - b11)
        # m6 = (a10 - a00) * (b00 + b01)
        # m7 = (a11 - a10) * (b10 + b11)
        
        # U (A factors)
        U[0, 0] = 1 # a00
        U[1, 0] = 1 # a01
        U[2, 0] = 1 # a10
        U[3, 0] = 1 # a11
        
        U[0, 1] = 1
        U[2, 1] = 1
        
        U[1, 2] = 1
        U[3, 2] = 1
        
        U[3, 3] = 1
        
        U[0, 4] = 1
        U[1, 4] = 1
        
        U[2, 5] = 1
        U[0, 5] = -1
        
        U[3, 6] = 1
        U[2, 6] = -1
        
        # V (B factors)
        # b01 + b11 -> V[1,0]=1, V[3,0]=1
        V[1, 0] = 1
        V[3, 0] = 1
        
        # b00 -> V[0,1]=1
        V[0, 1] = 1
        
        # b10 -> V[2,2]=1
        V[2, 2] = 1
        
        # b00 - b10 -> V[0,3]=1, V[2,3]=-1
        V[0, 3] = 1
        V[2, 3] = -1
        
        # b10 - b11 -> V[2,4]=1, V[3,4]=-1
        V[2, 4] = 1
        V[3, 4] = -1
        
        # b00 + b01 -> V[0,5]=1, V[1,5]=1
        V[0, 5] = 1
        V[1, 5] = 1
        
        # b10 + b11 -> V[2,6]=1, V[3,6]=1
        V[2, 6] = 1
        V[3, 6] = 1
        
        # W (C factors)
        # c00 = m1 - m4 + m5 + m6 + m7
        W[0, 0] = 1
        W[0, 3] = -1
        W[0, 4] = 1
        W[0, 5] = 1
        W[0, 6] = 1
        
        # c01 = m3 + m5
        W[1, 2] = 1
        W[1, 4] = 1
        
        # c10 = m2 + m4 - m6 - m7
        W[2, 1] = 1
        W[2, 3] = 1
        W[2, 5] = -1
        W[2, 6] = -1
        
        # c11 = m1 - m2 + m3 + m7
        W[3, 0] = 1
        W[3, 1] = -1
        W[3, 2] = 1
        W[3, 6] = 1
        
        return U, V, W

    # Function to embed a decomposition into a larger block
    # n, m, p are the dimensions of the sub-block
    # U, V, W are the decomposition of the sub-block
    # offset_i, offset_j, offset_k are the starting indices in the parent matrix
    def embed(U, V, W, n, m, p, off_i, off_j, off_k):
        # Parent dimensions are handled by the caller's accumulation
        # We just return the local indices mapped to global
        # This is tricky to do with just arrays.
        # Better to build the global arrays directly.
        pass

    # I will build the global arrays directly by iterating through the decomposition tree.
    
    global_U = []
    global_V = []
    global_W = []
    
    def add_decomp(U, V, W, n, m, p, off_i, off_j, off_k):
        # U is n*m x r
        # V is m*p x r
        # W is p*n x r
        # off_i, off_j, off_k are offsets for rows of A, cols of A (rows of B), rows of B (cols of C)
        
        # A indices: i in [0, n), j in [0, m)
        # Global A row: off_i + i, col: off_j + j
        # B indices: j in [0, m), k in [0, p)
        # Global B row: off_j + j, col: off_k + k
        # C indices: k in [0, p), i in [0, n)
        # Global C row: off_k + k, col: off_i + i
        
        for c in range(U.shape[1]):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            
            # Create full columns
            U_full = [0] * (3 * 5) # n*m = 15
            V_full = [0] * (5 * 5) # m*p = 25
            W_full = [0] * (5 * 3) # p*n = 15
            
            for i in range(n):
                for j in range(m):
                    U_full[off_i * 5 + off_j + i * 5 + j] = u_col[i * m + j]
                    
            for j in range(m):
                for k in range(p):
                    V_full[off_j * 5 + off_k + j * 5 + k] = v_col[j * p + k]
                    
            for k in range(p):
                for i in range(n):
                    W_full[off_k * 3 + off_i + k * 3 + i] = w_col[k * n + i]
                    
            global_U.append(U_full)
            global_V.append(V_full)
            global_W.append(W_full)
            
    # 1. Right block: M_{3,1,5}
    # n=3, m=1, p=5
    # off_i=0, off_j=4, off_k=0
    U_r, V_r, W_r = get_basic(3, 1, 5)
    add_decomp(U_r, V_r, W_r, 3, 1, 5, 0, 4, 0)
    
    # 2. Left block: M_{3,4,5}
    # Split into two M_{3,2,5}
    
    # 2a. Left-Left: M_{3,2,5}
    # off_i=0, off_j=0, off_k=0
    # Split p=5 into 3,2
    
    # 2a-i. LL-Left: M_{3,2,3}
    # n=3, m=2, p=3
    # off_i=0, off_j=0, off_k=0
    # Split n=3 into 2,1
    
    # 2a-i-A. LLL-Left: M_{2,2,3}
    # n=2, m=2, p=3
    # off_i=0, off_j=0, off_k=0
    # Split p=3 into 2,1
    
    # 2a-i-A-i. LLLL-Left: M_{2,2,2}
    # n=2, m=2, p=2
    # off_i=0, off_j=0, off_k=0
    U_s, V_s, W_s = get_strassen_222()
    add_decomp(U_s, V_s, W_s, 2, 2, 2, 0, 0, 0)
    
    # 2a-i-A-ii. LLLL-Right: M_{2,2,1}
    # n=2, m=2, p=1
    # off_i=0, off_j=0, off_k=2
    U_b, V_b, W_b = get_basic(2, 2, 1)
    add_decomp(U_b, V_b, W_b, 2, 2, 1, 0, 0, 2)
    
    # 2a-i-B. LLL-Right: M_{1,2,3}
    # n=1, m=2, p=3
    # off_i=2, off_j=0, off_k=0
    U_b, V_b, W_b = get_basic(1, 2, 3)
    add_decomp(U_b, V_b, W_b, 1, 2, 3, 2, 0, 0)
    
    # 2a-ii. LL-Right: M_{3,2,2}
    # n=3, m=2, p=2
    # off_i=0, off_j=0, off_k=3
    # Split n=3 into 2,1
    
    # 2a-ii-A. LLR-Left: M_{2,2,2}
    # n=2, m=2, p=2
    # off_i=0, off_j=0, off_k=3
    U_s, V_s, W_s = get_strassen_222()
    add_decomp(U_s, V_s, W_s, 2, 2, 2, 0, 0, 3)
    
    # 2a-ii-B. LLR-Right: M_{1,2,2}
    # n=1, m=2, p=2
    # off_i=2, off_j=0, off_k=3
    U_b, V_b, W_b = get_basic(1, 2, 2)
    add_decomp(U_b, V_b, W_b, 1, 2, 2, 2, 0, 3)
    
    # 2b. Left-Right: M_{3,2,5}
    # off_i=0, off_j=2, off_k=0
    # Same structure as 2a, but off_j=2
    
    # 2b-i. LR-Left: M_{3,2,3}
    # n=3, m=2, p=3
    # off_i=0, off_j=2, off_k=0
    # Split n=3 into 2,1
    
    # 2b-i-A. LRL-Left: M_{2,2,3}
    # n=2, m=2, p=3
    # off_i=0, off_j=2, off_k=0
    # Split p=3 into 2,1
    
    # 2b-i-A-i. LRLR-Left: M_{2,2,2}
    U_s, V_s, W_s = get_strassen_222()
    add_decomp(U_s, V_s, W_s, 2, 2, 2, 0, 2, 0)
    
    # 2b-i-A-ii. LRLR-Right: M_{2,2,1}
    U_b, V_b, W_b = get_basic(2, 2, 1)
    add_decomp(U_b, V_b, W_b, 2, 2, 1, 0, 2, 2)
    
    # 2b-i-B. LRL-Right: M_{1,2,3}
    U_b, V_b, W_b = get_basic(1, 2, 3)
    add_decomp(U_b, V_b, W_b, 1, 2, 3, 2, 2, 0)
    
    # 2b-ii. LR-Right: M_{3,2,2}
    # n=3, m=2, p=2
    # off_i=0, off_j=2, off_k=3
    # Split n=3 into 2,1
    
    # 2b-ii-A. LRR-Left: M_{2,2,2}
    U_s, V_s, W_s = get_strassen_222()
    add_decomp(U_s, V_s, W_s, 2, 2, 2, 0, 2, 3)
    
    # 2b-ii-B. LRR-Right: M_{1,2,2}
    U_b, V_b, W_b = get_basic(1, 2, 2)
    add_decomp(U_b, V_b, W_b, 1, 2, 2, 2, 2, 3)
    
    # Convert lists to arrays
    U_arr = np.array(global_U).T
    V_arr = np.array(global_V).T
    W_arr = np.array(global_W).T
    
    # Check rank
    rank = U_arr.shape[1]
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}

