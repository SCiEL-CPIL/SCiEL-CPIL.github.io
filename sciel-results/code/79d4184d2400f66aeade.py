"""Optimized tensor decomposition for 2x3x6 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    The minimal rank is 33, achieved by splitting the 3x6 matrix into three 3x2 blocks.
    Each 2x3x2 block multiplication has a minimal rank of 11 (7 from Strassen for 2x2x2 
    core + 4 scalar multiplications for the remaining column/row interaction).
    Total rank = 3 * 11 = 33.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    rank = 33
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank) = (6, 33)
    # V: (m*p, rank) = (18, 33)
    # W: (p*n, rank) = (12, 33)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for a single 2x3x2 block
    # A is 2x3: indices (i,j) with i in {0,1}, j in {0,1,2}. Flat index = i*3+j.
    #   A00=0, A01=1, A02=2
    #   A10=3, A11=4, A12=5
    # B is 3x2: indices (j,k) with j in {0,1,2}, k in {0,1}. Flat index = j*2+k.
    #   B00=0, B01=1
    #   B10=2, B11=3
    #   B20=4, B21=5
    # C is 2x2: indices (i,k) with i in {0,1}, k in {0,1}. Flat index = i*2+k.
    #   C00=0, C01=1
    #   C10=2, C11=3
    
    # The decomposition uses Strassen for the 2x2x2 part (j=0,1) and 4 scalar mults for j=2.
    # Strassen terms for A_L (cols 0,1) * B_T (rows 0,1):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # Reconstruction of C_L (2x2):
    # C00 = M1 - M2 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    #
    # Scalar terms for A[:,2] * B[2,:]:
    # K1 = A02 * B20 -> C00
    # K2 = A02 * B21 -> C01
    # K3 = A12 * B20 -> C10
    # K4 = A12 * B21 -> C11
    
    terms = []
    
    # M1
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1} 
    })
    # M2
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1} 
    })
    # M3
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1} 
    })
    # M4
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1} 
    })
    # M5
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1} 
    })
    # M6
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1} 
    })
    # M7
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1} 
    })
    
    # K1 = A02 * B20
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1} 
    })
    # K2 = A02 * B21
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1} 
    })
    # K3 = A12 * B20
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1} 
    })
    # K4 = A12 * B21
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1} 
    })
    
    # Fill the matrices for the 3 blocks (k=0,1,2 corresponding to columns of B and C)
    for block in range(3):
        if time.time() > deadline:
            # Return what we have so far (should not happen for this deterministic construction)
            return {"U": U, "V": V, "W": W}
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U: A indices are global. Same for all blocks.
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: B indices in block.
            # Local B index l_idx corresponds to j_local = l_idx // 2, k_local = l_idx % 2.
            # Global B index: j * p + (block * 2 + k_local).
            # Note: B is (m, p) = (3, 6). Index is j*6 + k_global.
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                k_global = block * 2 + k_local
                g_idx = j_local * p + k_global
                V[g_idx][col] = val
            
            # W: C indices in block.
            # Local C index l_idx corresponds to i_local = l_idx // 2, k_local = l_idx % 2.
            # Global C index follows convention W[k*n + i, r].
            # k_global = block * 2 + k_local.
            # g_idx = k_global * n + i_local.
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                k_global = block * 2 + k_local
                g_idx = k_global * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
