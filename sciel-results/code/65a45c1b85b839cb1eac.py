import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition.
    # p=6 is split into 3 blocks of size 2.
    # Each block is a 2x3x2 multiplication.
    # Rank of 2x3x2 is 11 (using Strassen for 2x2x2 part + 4 scalar mults).
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    # U: (n*m, rank) -> (6, 33)
    # V: (m*p, rank) -> (18, 33)
    # W: (p*n, rank) -> (12, 33)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 11 terms for a 2x3x2 block
    # The block multiplication is C_block = A_block * B_block
    # A_block is 2x3, B_block is 3x2, C_block is 2x2.
    # Indices:
    # A: i in 0..1, j in 0..2. Flat index in A_block: i*3 + j.
    #     Global U index: i*m + j = i*3 + j.
    # B: j in 0..2, k in 0..1. Flat index in B_block: j*2 + k.
    #     Global V index: j*p + (block_start + k).
    # C: i in 0..1, k in 0..1. Flat index in C_block: i*2 + k.
    #     Global W index: (block_start + k)*n + i.
    
    # Strassen terms for A_L (2x2) * B_T (2x2)
    # A_L uses columns j=0,1 of A.
    # B_T uses rows j=0,1 of B.
    
    # A indices in A_L:
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    
    # B indices in B_T:
    # B[0,0] -> V[0] (local)
    # B[0,1] -> V[1] (local)
    # B[1,0] -> V[2] (local)
    # B[1,1] -> V[3] (local)
    
    # C indices in C_block:
    # C[0,0] -> W[0] (local)
    # C[0,1] -> W[1] (local)
    # C[1,0] -> W[2] (local)
    # C[1,1] -> W[3] (local)
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1} # C00, C11
    })
    # M2 = (A10+A11)B00
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1} # C10, C11 (C11 has -M2)
    })
    # M3 = A00(B01-B11)
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1} # C01, C11
    })
    # M4 = A11(B10-B00)
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1} # C00, C10
    })
    # M5 = (A00+A01)B11
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1} # C00 (-M5), C01
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1} # C11
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1} # C00
    })
    
    # Scalar terms for A_R (2x1) * B_B (1x2)
    # A_R is column j=2 of A.
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    # B_B is row j=2 of B.
    # B[2,0] -> V[4] (local)
    # B[2,1] -> V[5] (local)
    
    # K1 = A02 * B20
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1} # C00
    })
    # K2 = A02 * B21
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1} # C01
    })
    # K3 = A12 * B20
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1} # C10
    })
    # K4 = A12 * B21
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1} # C11
    })
    
    # Fill the matrices for 3 blocks
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U
            for idx, val in term['u'].items():
                U[idx, col] = val
            
            # V
            # B indices in block: j in 0..2, k in 0..1
            # Global B index: j*p + (block*2 + k)
            # term['v'] keys are local B indices (j*2+k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            
            # W
            # C indices in block: i in 0..1, k in 0..1
            # Global C index: (block*2 + k)*n + i
            # term['w'] keys are local C indices (i*2+k)
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
