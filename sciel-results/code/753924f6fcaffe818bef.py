import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 5, 5
    
    # Step 1: Construct rank-49 decomposition for M_{4,4,4} using Kronecker product of Strassen's M_{2,2,2}
    # Strassen's 7 tensors for 2x2x2
    # u, v, w are 2x2 matrices
    # We represent them as 4x4 matrices for the Kronecker product step later? 
    # No, we generate the 49 terms directly.
    
    # Strassen tensors (u_r, v_r, w_r) for 2x2 matrices
    # Indexing: 0,1
    E00 = np.array([[1, 0], [0, 0]])
    E01 = np.array([[0, 1], [0, 0]])
    E10 = np.array([[0, 0], [1, 0]])
    E11 = np.array([[0, 0], [0, 1]])
    I2 = np.array([[1, 0], [0, 1]])
    
    # u_r, v_r, w_r
    # 1
    u1 = E00; v1 = E00; w1 = E00 + E11
    # 2
    u2 = E11; v2 = E11; w2 = E10 + E11
    # 3
    u3 = I2; v3 = I2; w3 = E00 - E11
    # 4
    u4 = E00 + E01; v4 = E11; w4 = E01 + E11
    # 5
    u5 = E10 + E11; v5 = E00; w5 = E00 - E10
    # 6
    u6 = E01 + E11; v6 = E00; w6 = E11 - E01
    # 7
    u7 = E00 + E10; v7 = E11; w7 = E10 - E01
    
    strassen_tensors = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Generate 49 tensors for 4x4x4
    # u, v, w will be 4x4 matrices
    tensors_444 = []
    for i in range(7):
        for j in range(7):
            ui, vi, wi = strassen_tensors[i]
            uj, vj, wj = strassen_tensors[j]
            # Kronecker product
            U_kron = np.kron(ui, uj)
            V_kron = np.kron(vi, vj)
            W_kron = np.kron(wi, wj)
            tensors_444.append((U_kron, V_kron, W_kron))
            
    # Step 2: Construct schoolbook decompositions for the remaining blocks
    # We need M_{4,1,4}, M_{4,4,1}, M_{4,1,1}
    # M_{4,1,4}: A is 4x1, B is 1x4, C is 4x4
    # Rank is 4*1*4 = 16
    # Terms: E_{i,1} (in A), E_{1,k} (in B), E_{i,k} (in C)
    # A is 4x1. Index i=0..3, j=0.
    # B is 1x4. Index j=0, k=0..3.
    # C is 4x4. Index i=0..3, k=0..3.
    
    tensors_414 = []
    for i in range(4):
        for k in range(4):
            # A part: 4x1 matrix with 1 at (i,0)
            U_part = np.zeros((4, 1))
            U_part[i, 0] = 1
            # B part: 1x4 matrix with 1 at (0,k)
            V_part = np.zeros((1, 4))
            V_part[0, k] = 1
            # C part: 4x4 matrix with 1 at (i,k)
            W_part = np.zeros((4, 4))
            W_part[i, k] = 1
            tensors_414.append((U_part, V_part, W_part))
            
    # M_{4,4,1}: A is 4x4, B is 4x1, C is 4x1
    # Rank is 4*4*1 = 16
    # Terms: E_{i,j} (in A), E_{j,0} (in B), E_{i,0} (in C)
    
    tensors_441 = []
    for i in range(4):
        for j in range(4):
            U_part = np.zeros((4, 4))
            U_part[i, j] = 1
            V_part = np.zeros((4, 1))
            V_part[j, 0] = 1
            W_part = np.zeros((4, 1))
            W_part[i, 0] = 1
            tensors_441.append((U_part, V_part, W_part))
            
    # M_{4,1,1}: A is 4x1, B is 1x1, C is 4x1
    # Rank is 4*1*1 = 4
    # Terms: E_{i,0} (in A), E_{0,0} (in B), E_{i,0} (in C)
    
    tensors_411 = []
    for i in range(4):
        U_part = np.zeros((4, 1))
        U_part[i, 0] = 1
        V_part = np.zeros((1, 1))
        V_part[0, 0] = 1
        W_part = np.zeros((4, 1))
        W_part[i, 0] = 1
        tensors_411.append((U_part, V_part, W_part))
        
    # Step 3: Assemble the full 4x5x5 decomposition
    # A is 4x5. Split into A_44 (4x4) and A_41 (4x1).
    # B is 5x5. Split into B_44 (4x4), B_41 (4x1), B_14 (1x4), B_11 (1x1).
    # C is 4x5. Split into C_44 (4x4) and C_41 (4x1).
    
    # The total product is:
    # C_44 = A_44 B_44 + A_41 B_14
    # C_41 = A_44 B_41 + A_41 B_11
    
    # We need to map the local tensors to the global 4x5, 5x5, 4x5 spaces.
    
    U_rows = 4 * 5 # 20
    V_rows = 5 * 5 # 25
    W_rows = 4 * 5 # 20
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_tensor(U_local, V_local, W_local, U_offset, V_offset, W_offset):
        """
        U_local: 4x4 or 4x1
        V_local: 4x4 or 4x1 or 1x4 or 1x1
        W_local: 4x4 or 4x1
        
        Offsets determine where in the global matrix these blocks sit.
        A is 4x5. Rows 0-3, Cols 0-4.
        A_44 is at (0:4, 0:4). A_41 is at (0:4, 4:5).
        B is 5x5. Rows 0-4, Cols 0-4.
        B_44 is at (0:4, 0:4). B_41 is at (0:4, 4:5).
        B_14 is at (4:5, 0:4). B_11 is at (4:5, 4:5).
        C is 4x5. Rows 0-3, Cols 0-4.
        C_44 is at (0:4, 0:4). C_41 is at (0:4, 4:5).
        """
        U_global = np.zeros((20, 1))
        V_global = np.zeros((25, 1))
        W_global = np.zeros((20, 1))
        
        # Flatten local matrices and place them
        # U_local is (4, 4) or (4, 1)
        # We need to place it in the 4x5 matrix A.
        # U_offset is (row_start, col_start) for the block in A.
        # But U_local is already the correct shape for the block.
        # We just need to flatten it and put it in the right spot in U_global.
        
        # Flatten U_local (row-major)
        U_flat = U_local.flatten()
        # Calculate index in U_global
        # U_global index = i*5 + j
        # Block starts at U_offset
        # U_local index = i_local * width_local + j_local
        # i = U_offset[0] + i_local
        # j = U_offset[1] + j_local
        
        h_u, w_u = U_local.shape
        for i_l in range(h_u):
            for j_l in range(w_u):
                i_g = U_offset[0] + i_l
                j_g = U_offset[1] + j_l
                idx = i_g * 5 + j_g
                U_global[idx, 0] = U_flat[i_l * w_u + j_l]
                
        # V_local is (4,4), (4,1), (1,4), (1,1)
        h_v, w_v = V_local.shape
        V_flat = V_local.flatten()
        for i_l in range(h_v):
            for j_l in range(w_v):
                i_g = V_offset[0] + i_l
                j_g = V_offset[1] + j_l
                idx = i_g * 5 + j_g
                V_global[idx, 0] = V_flat[i_l * w_v + j_l]
                
        # W_local is (4,4) or (4,1)
        h_w, w_w = W_local.shape
        W_flat = W_local.flatten()
        for i_l in range(h_w):
            for j_l in range(w_w):
                i_g = W_offset[0] + i_l
                j_g = W_offset[1] + j_l
                idx = i_g * 5 + j_g
                W_global[idx, 0] = W_flat[i_l * w_w + j_l]
                
        U_list.append(U_global.flatten())
        V_list.append(V_global.flatten())
        W_list.append(W_global.flatten())
        
    # Add 4x4x4 tensors (A_44, B_44, C_44)
    for U_l, V_l, W_l in tensors_444:
        add_tensor(U_l, V_l, W_l, (0, 0), (0, 0), (0, 0))
        
    # Add 4x1x4 tensors (A_41, B_14, C_44)
    # A_41 is at (0, 4). B_14 is at (4, 0). C_44 is at (0, 0).
    for U_l, V_l, W_l in tensors_414:
        add_tensor(U_l, V_l, W_l, (0, 4), (4, 0), (0, 0))
        
    # Add 4x4x1 tensors (A_44, B_41, C_41)
    # A_44 is at (0, 0). B_41 is at (0, 4). C_41 is at (0, 4).
    for U_l, V_l, W_l in tensors_441:
        add_tensor(U_l, V_l, W_l, (0, 0), (0, 4), (0, 4))
        
    # Add 4x1x1 tensors (A_41, B_11, C_41)
    # A_41 is at (0, 4). B_11 is at (4, 4). C_41 is at (0, 4).
    for U_l, V_l, W_l in tensors_411:
        add_tensor(U_l, V_l, W_l, (0, 4), (4, 4), (0, 4))
        
    # Convert lists to matrices
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

