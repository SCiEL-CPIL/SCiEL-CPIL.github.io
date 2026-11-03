"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-114 tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Partition the 9 columns of B into groups: 4 + 4 + 1
    - Use Strassen's algorithm (rank 49) for the two 4x4x4 blocks
    - Use schoolbook (rank 16) for the 4x4x1 block
    - Total rank: 49 + 49 + 16 = 114
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, leave 20s for safety
    
    # Rank-7 decomposition for 2x2x2 (Bini)
    # U, V, W are 7x4 matrices
    bini_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    bini_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    bini_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [-1, 1, 0, 0]
    ])
    
    # Strassen coefficients for 4x4x4
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # 7 Strassen components
    strassen_U_blocks = np.array([
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ])
    strassen_V_blocks = np.array([
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ])
    
    # Strassen combination for C blocks
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M2
    strassen_W_blocks = np.array([
        [1, -1, 0, 0], # C00 coeff for M1..M7
        [0, 0, 1, 0],  # C01
        [0, 1, 0, 1],  # C10
        [1, 0, -1, 0]  # C11 (Wait, M1 - M3 + M6 - M2 => [1, -1, 1, 0] for M1,M2,M3,M6? No)
    ])
    # Let's be precise.
    # C00: M1(1), M4(-1), M5(1), M7(-1) -> [1, 0, 0, -1, 1, 0, -1]
    # C01: M3(1), M7(1) -> [0, 0, 1, 0, 0, 0, 1]
    # C10: M2(1), M4(1) -> [0, 1, 0, 1, 0, 0, 0]
    # C11: M1(1), M3(-1), M6(1), M2(-1) -> [1, -1, -1, 0, 0, 1, 0]
    
    strassen_comb = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ])
    
    # Build 4x4x4 decomposition
    def build_4x4x4():
        U_list = []
        V_list = []
        W_list = []
        
        # 7 Strassen components, each with 7 Bini components
        for s in range(7):
            # s is Strassen component index
            # Get linear forms for A and B
            # X = L_A(s) A, Y = L_B(s) B
            # X is 2x2, Y is 2x2
            # We decompose X Y using Bini
            for b in range(7):
                # b is Bini component index
                # U_b, V_b, W_b from Bini
                # X_{kl} = sum_{block} strassen_U_blocks[s, block] * A_{block, kl}
                # Y_{pq} = sum_{block} strassen_V_blocks[s, block] * B_{block, pq}
                
                # U for this component:
                # U_{total, A_{block, kl}} = bini_U[b, kl] * strassen_U_blocks[s, block]
                # A is 4x4. Flattened index: block * 4 + local_idx
                # block 00: 0..3, 01: 4..7, 10: 8..11, 11: 12..15
                U_comp = np.zeros(16)
                for block in range(4):
                    # block 0: 00, 1: 01, 2: 10, 3: 11
                    # local indices 0..3 correspond to 00, 01, 10, 11 of the 2x2 block
                    for local in range(4):
                        idx = block * 4 + local
                        U_comp[idx] = bini_U[b, local] * strassen_U_blocks[s, block]
                        
                V_comp = np.zeros(16)
                for block in range(4):
                    for local in range(4):
                        idx = block * 4 + local
                        V_comp[idx] = bini_V[b, local] * strassen_V_blocks[s, block]
                        
                # W for this component:
                # W_{total, C_{block, ij}} = strassen_comb[block, s] * bini_W[b, ij]
                # C is 4x4. Flattened index: block * 4 + local_idx
                W_comp = np.zeros(16)
                for block in range(4):
                    for local in range(4):
                        idx = block * 4 + local
                        W_comp[idx] = strassen_comb[block, s] * bini_W[b, local]
                        
                U_list.append(U_comp)
                V_list.append(V_comp)
                W_list.append(W_comp)
                
        return np.array(U_list).T, np.array(V_list).T, np.array(W_list).T

    U_444, V_444, W_444 = build_444()
    
    # Build 4x4x1 decomposition
    def build_4x4x1():
        U_list = []
        V_list = []
        W_list = []
        for i in range(4):
            for j in range(4):
                U_comp = np.zeros(16)
                V_comp = np.zeros(4)
                W_comp = np.zeros(4)
                
                U_comp[i * 4 + j] = 1
                V_comp[j] = 1
                W_comp[i] = 1
                
                U_list.append(U_comp)
                V_list.append(V_comp)
                W_list.append(W_comp)
        return np.array(U_list).T, np.array(V_list).T, np.array(W_list).T

    U_441, V_441, W_441 = build_441()
    
    # Combine
    # Block 1: cols 0-3 of B (4x4x4)
    # Block 2: cols 4-7 of B (4x4x4)
    # Block 3: col 8 of B (4x4x1)
    
    U_total = np.zeros((16, 49 + 49 + 16))
    V_total = np.zeros((36, 49 + 49 + 16)) # 4x9 = 36
    W_total = np.zeros((36, 49 + 49 + 16)) # 9x4 = 36
    
    # Block 1
    U_total[:, 0:49] = U_444
    V_total[0:16, 0:49] = V_444 # B cols 0-3
    W_total[0:16, 0:49] = W_444 # C cols 0-3
    
    # Block 2
    U_total[:, 49:98] = U_444
    V_total[16:32, 49:98] = V_444 # B cols 4-7
    W_total[16:32, 49:98] = W_444 # C cols 4-7
    
    # Block 3
    U_total[:, 98:114] = U_441
    V_total[32:36, 98:114] = V_441 # B col 8
    W_total[32:36, 98:114] = W_441 # C col 8
    
    # Convert to lists of lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

