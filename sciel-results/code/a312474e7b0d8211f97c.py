import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS
    
    # Splits
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    # Accumulators for U, V, W
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We don't know the final rank yet, so we use lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomp(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset, U, V, W):
        """
        Adds the decomposition for a sub-problem of size n_sub x m_sub x p_sub
        located at offsets i_offset, j_offset, k_offset in the global matrix.
        U, V, W are the local decomposition matrices for this sub-problem.
        """
        # Local indices
        # U_local: (n_sub * m_sub, rank_sub)
        # V_local: (m_sub * p_sub, rank_sub)
        # W_local: (p_sub * n_sub, rank_sub)
        
        rank_sub = U.shape[1]
        
        for c in range(rank_sub):
            # Construct global column vectors
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U
            for i in range(n_sub):
                for j in range(m_sub):
                    u_val = U[i * m_sub + j, c]
                    if u_val != 0:
                        global_i = i_offset + i
                        global_j = j_offset + j
                        u_col[global_i * m + global_j] = u_val
            
            # Fill V
            for j in range(m_sub):
                for k in range(p_sub):
                    v_val = V[j * p_sub + k, c]
                    if v_val != 0:
                        global_j = j_offset + j
                        global_k = k_offset + k
                        v_col[global_j * p + global_k] = v_val
            
            # Fill W
            for k in range(p_sub):
                for i in range(n_sub):
                    w_val = W[k * n_sub + i, c]
                    if w_val != 0:
                        global_k = k_offset + k
                        global_i = i_offset + i
                        w_col[global_k * n + global_i] = w_val
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
        
        return rank_sub

    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 using Strassen
        # n=2, m=2, p=2
        # Rank 7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # Term 1
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # C11, C12, C21, C22 (with signs)
        # Wait, W is for C.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 multiplications
        # 1: (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1 # C00
        W[3,0]=1 # C11
        
        # 2: (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1 # C10
        W[3,1]=-1 # C11
        
        # 3: A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1 # C01
        W[3,2]=1 # C11
        
        # 4: A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1 # C00
        W[2,3]=1 # C10
        
        # 5: (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1 # C00
        W[1,4]=1 # C01
        
        # 6: (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1 # C11
        
        # 7: (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1 # C00
        W[1,6]=-1 # C01
        
        return U, V, W

    def get_schoolbook(n_sub, m_sub, p_sub):
        rank = n_sub * m_sub * p_sub
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j, col] = 1
                    V[j * p_sub + k, col] = 1
                    W[k * n_sub + i, col] = 1
                    col += 1
        return U, V, W

    def get_decomp(n_sub, m_sub, p_sub):
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            return get_strassen_2x2x2()
        else:
            return get_schoolbook(n_sub, m_sub, p_sub)

    # Iterate over m-blocks
    j_offset = 0
    for j_idx, m_sub in enumerate(m_splits):
        # For each m-block, we need to compute M_{3, m_sub, 5}
        # We split n=3 into [2, 1]
        n_sub_splits = [2, 1]
        i_offset = 0
        for n_sub in n_sub_splits:
            # For each n-block, we have M_{n_sub, m_sub, 5}
            # We split p=5 into [2, 2, 1]
            k_offset = 0
            for p_sub in p_splits:
                if time.time() > deadline:
                    # Fallback to schoolbook if time runs out (unlikely)
                    pass 
                
                U, V, W = get_decomp(n_sub, m_sub, p_sub)
                add_decomp(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset, U, V, W)
                k_offset += p_sub
            i_offset += n_sub
        j_offset += m_sub

    # Convert lists to lists of lists (as required by interface)
    U_final = [col for col in U_cols]
    V_final = [col for col in V_cols]
    W_final = [col for col in W_cols]
    
    return {"U": U_final, "V": V_final, "W": W_final}
