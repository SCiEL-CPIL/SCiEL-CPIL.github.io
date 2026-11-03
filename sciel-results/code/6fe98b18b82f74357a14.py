import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target rank 44
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Define index partitions
    # n=3 -> I1=[0], I2=[1,2]
    # m=4 -> J1=[0,1], J2=[2,3]
    # p=4 -> K1=[0,1], K2=[2,3]
    
    I_parts = [[0], [1, 2]]
    J_parts = [[0, 1], [2, 3]]
    K_parts = [[0, 1], [2, 3]]
    
    # Helper to add a block
    def add_block(I, J, K, type_block):
        nonlocal col
        n_b = len(I)
        m_b = len(J)
        p_b = len(K)
        
        if type_block == 'strassen':
            # 2x2x2
            r_b = 7
            # Local U indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
            # Local V indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
            # Local W indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
            
            # M1: (A00+A11)(B00+B11)
            # U: A00(1), A11(1) -> idx 0, 3
            # V: B00(1), B11(1) -> idx 0, 3
            # W: C00(1), C11(1) -> idx 0, 3
            
            # M2: (A10+A11)B00
            # U: A10(1), A11(1) -> idx 2, 3
            # V: B00(1) -> idx 0
            # W: C10(1) -> idx 2
            
            # M3: A00(B01-B11)
            # U: A00(1) -> idx 0
            # V: B01(1), B11(-1) -> idx 1, 3
            # W: C01(1) -> idx 1
            
            # M4: A11(B10-B00)
            # U: A11(1) -> idx 3
            # V: B10(1), B00(-1) -> idx 2, 0
            # W: C00(1), C10(1) -> idx 0, 2
            
            # M5: (A00+A01)B11
            # U: A00(1), A01(1) -> idx 0, 1
            # V: B11(1) -> idx 3
            # W: C00(-1), C01(1) -> idx 0, 1
            
            # M6: (A10-A00)(B00+B01)
            # U: A10(1), A00(-1) -> idx 2, 0
            # V: B00(1), B01(1) -> idx 0, 1
            # W: C11(1) -> idx 3
            
            # M7: (A01-A11)(B10+B11)
            # U: A01(1), A11(-1) -> idx 1, 3
            # V: B10(1), B11(1) -> idx 2, 3
            # W: C00(1), C11(1) -> idx 0, 3
            
            factors = [
                # U_coeffs, V_coeffs, W_coeffs
                ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1
                ([(2,1), (3,1)], [(0,1)], [(2,1)]), # M2
                ([(0,1)], [(1,1), (3,-1)], [(1,1)]), # M3
                ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]), # M4
                ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]), # M5
                ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]), # M6
                ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1), (3,1)]) # M7
            ]
            
            for u_coefs, v_coefs, w_coefs in factors:
                for local_idx, val in u_coefs:
                    global_i = I[local_idx // 2]
                    global_j = J[local_idx % 2]
                    global_u_idx = global_i * m + global_j
                    U[global_u_idx, col] += val
                for local_idx, val in v_coefs:
                    global_j = J[local_idx // 2]
                    global_k = K[local_idx % 2]
                    global_v_idx = global_j * p + global_k
                    V[global_v_idx, col] += val
                for local_idx, val in w_coefs:
                    global_k = K[local_idx // 2]
                    global_i = I[local_idx % 2]
                    global_w_idx = global_k * n + global_i
                    W[global_w_idx, col] += val
                col += 1

        elif type_block == 'trivial_122':
            # 1x2x2
            r_b = 4
            # U: A00(0), A01(1)
            # V: B00(0), B01(1), B10(2), B11(3)
            # W: C00(0), C01(1)
            
            factors = [
                # M1: A00 B00
                ([(0,1)], [(0,1)], [(0,1)]),
                # M2: A01 B10
                ([(1,1)], [(2,1)], [(0,1)]),
                # M3: A00 B01
                ([(0,1)], [(1,1)], [(1,1)]),
                # M4: A01 B11
                ([(1,1)], [(3,1)], [(1,1)])
            ]
            
            for u_coefs, v_coefs, w_coefs in factors:
                for local_idx, val in u_coefs:
                    global_i = I[0] # n_b=1
                    global_j = J[local_idx]
                    global_u_idx = global_i * m + global_j
                    U[global_u_idx, col] += val
                for local_idx, val in v_coefs:
                    global_j = J[local_idx // 2]
                    global_k = K[local_idx % 2]
                    global_v_idx = global_j * p + global_k
                    V[global_v_idx, col] += val
                for local_idx, val in w_coefs:
                    global_k = K[local_idx // 2]
                    global_i = I[0]
                    global_w_idx = global_k * n + global_i
                    W[global_w_idx, col] += val
                col += 1

    # Iterate through all blocks
    for K_idx, K in enumerate(K_parts):
        for I_idx, I in enumerate(I_parts):
            for J_idx, J in enumerate(J_parts):
                if len(I) == 2 and len(J) == 2:
                    add_block(I, J, K, 'strassen')
                else:
                    add_block(I, J, K, 'trivial_122')

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
