import numpy as np

def construct():
    """
    Exact tensor decomposition for 3x4x5 matrix multiplication with rank 56.
    
    Strategy: Split middle dimension m=4 into two 2-dim parts.
    For each part, split output p=5 into 2+2+1.
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    Total rank = 2 * (7+4+7+4+4+2) = 56.
    
    Index conventions:
    U: (12, 56) - A[i,j] at index i*4+j
    V: (20, 56) - B[j,k] at index j*5+k  
    W: (15, 56) - C[i,k] at index k*3+i (trace convention)
    """
    n, m, p = 3, 4, 5
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_decomposition(u_map, v_map, w_map, U_blk, V_blk, W_blk):
        """
        u_map: dict mapping local row idx -> global U index
        v_map: dict mapping local row idx -> global V index  
        w_map: dict mapping local row idx -> global W index
        """
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            for local_i, global_i in u_map.items():
                u_vec[global_i] = U_blk[local_i, r]
            U_rows.append(u_vec)
            
            v_vec = np.zeros(m * p)
            for local_j, global_j in v_map.items():
                v_vec[global_j] = V_blk[local_j, r]
            V_rows.append(v_vec)
            
            w_vec = np.zeros(p * n)
            for local_k, global_k in w_map.items():
                w_vec[global_k] = W_blk[local_k, r]
            W_rows.append(w_vec)
    
    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        # M3 = A00(B01-B11)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # M4 = A11(B10-B00)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    # Schoolbook 1x2x2 (rank 4)
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        U[0,2]=1; V[1,2]=1; W[1,2]=1
        U[1,3]=1; V[3,3]=1; W[1,3]=1
        return U, V, W
    
    # Schoolbook 2x2x1 (rank 4)
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W
    
    # Schoolbook 1x2x1 (rank 2)
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W
    
    # Process each of the two middle-dimension parts (j=0,1 and j=2,3)
    for mid_part in range(2):
        j_offset = mid_part * 2  # 0 or 2
        
        # For each output column group
        # Cols 0,1 (2x2 block)
        # Cols 2,3 (2x2 block)  
        # Col 4 (1x1 block)
        
        # --- 2x2x2 block for cols 0,1 ---
        U_blk, V_blk, W_blk = strassen_2x2x2()
        # A rows 0,1 (top 2x2): U indices for A[i, j_offset], A[i, j_offset+1]
        u_map = {0: 0*4 + j_offset, 1: 0*4 + j_offset+1, 2: 1*4 + j_offset, 3: 1*4 + j_offset+1}
        # B rows j_offset, j_offset+1; cols 0,1
        v_map = {0: j_offset*5 + 0, 1: j_offset*5 + 1, 2: (j_offset+1)*5 + 0, 3: (j_offset+1)*5 + 1}
        # C rows 0,1; cols 0,1 -> W index k*3+i
        w_map = {0: 0*3+0, 1: 1*3+0, 2: 0*3+1, 3: 1*3+1}
        add_decomposition(u_map, v_map, w_map, U_blk, V_blk, W_blk)
        
        # A row 2 (bottom 1x2): U indices for A[2, j_offset], A[2, j_offset+1]
        U_blk, V_blk, W_blk = schoolbook_1x2x2()
        u_map = {0: 2*4 + j_offset, 1: 2*4 + j_offset+1}
        v_map = {0: j_offset*5 + 0, 1: j_offset*5 + 1, 2: (j_offset+1)*5 + 0, 3: (j_offset+1)*5 + 1}
        w_map = {0: 0*3+2, 1: 1*3+2}
        add_decomposition(u_map, v_map, w_map, U_blk, V_blk, W_blk)
        
        # --- 2x2x2 block for cols 2,3 ---
        U_blk, V_blk, W_blk = strassen_2x2x2()
        u_map = {0: 0*4 + j_offset, 1: 0*4 + j_offset+1, 2: 1*4 + j_offset, 3: 1*4 + j_offset+1}
        v_map = {0: j_offset*5 + 2, 1: j_offset*5 + 3, 2: (j_offset+1)*5 + 2, 3: (j_offset+1)*5 + 3}
        w_map = {0: 2*3+0, 1: 3*3+0, 2: 2*3+1, 3: 3*3+1}
        add_decomposition(u_map, v_map, w_map, U_blk, V_blk, W_blk)
        
        U_blk, V_blk, W_blk = schoolbook_1x2x2()
        u_map = {0: 2*4 + j_offset, 1: 2*4 + j_offset+1}
        v_map = {0: j_offset*5 + 2, 1: j_offset*5 + 3, 2: (j_offset+1)*5 + 2, 3: (j_offset+1)*5 + 3}
        w_map = {0: 2*3+2, 1: 3*3+2}
        add_decomposition(u_map, v_map, w_map, U_blk, V_blk, W_blk)
        
        # --- 2x2x1 block for col 4 ---
        U_blk, V_blk, W_blk = schoolbook_2x2x1()
        u_map = {0: 0*4 + j_offset, 1: 0*4 + j_offset+1, 2: 1*4 + j_offset, 3: 1*4 + j_offset+1}
        v_map = {0: j_offset*5 + 4, 1: (j_offset+1)*5 + 4}
        w_map = {0: 4*3+0, 1: 4*3+1}
        add_decomposition(u_map, v_map, w_map, U_blk, V_blk, W_blk)
        
        U_blk, V_blk, W_blk = schoolbook_1x2x1()
        u_map = {0: 2*4 + j_offset, 1: 2*4 + j_offset+1}
        v_map = {0: j_offset*5 + 4, 1: (j_offset+1)*5 + 4}
        w_map = {0: 4*3+2}
        add_decomposition(u_map, v_map, w_map, U_blk, V_blk, W_blk)
    
    U = np.array(U_rows).T
    V = np.array(V_rows).T
    W = np.array(W_rows).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
