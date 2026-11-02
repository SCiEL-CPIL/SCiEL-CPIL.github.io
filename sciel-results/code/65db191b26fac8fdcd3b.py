import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition for 2x2x6
    # Using explicit hardcoded terms for the 3 Strassen blocks
    # This is a "new" program structure: direct data definition
    
    U = []
    V = []
    W = []
    
    # Strassen terms for one 2x2x2 block
    # A indices: 0,1,2,3 (00,01,10,11)
    # B indices: 0,1,2,3 (00,01,10,11)
    # C indices: 0,1,2,3 (00,01,10,11)
    
    # Term 0: P1
    U.append([1, 0, 0, 1])
    V.append([1, 0, 0, 1])
    W.append([1, 0, 0, 1]) # C00, C11
    
    # Term 1: P2
    U.append([0, 0, 1, 1])
    V.append([1, 0, 0, 0])
    W.append([0, 0, 1, -1]) # C10, C11
    
    # Term 2: P3
    U.append([1, 0, 0, 0])
    V.append([0, 1, 0, -1])
    W.append([0, 1, 0, 1]) # C01, C11
    
    # Term 3: P4
    U.append([0, 0, 0, 1])
    V.append([-1, 0, 1, 0])
    W.append([1, 0, 1, 0]) # C00, C10
    
    # Term 4: P5
    U.append([1, 1, 0, 0])
    V.append([0, 0, 0, 1])
    W.append([-1, 1, 0, 0]) # C00, C01
    
    # Term 5: P6
    U.append([-1, 0, 1, 0])
    V.append([1, 1, 0, 0])
    W.append([0, 0, 0, 1]) # C11
    
    # Term 6: P7
    U.append([0, 1, 0, -1])
    V.append([0, 0, 1, 1])
    W.append([1, 0, 0, 0]) # C00
    
    # The above 7 terms form one 2x2x2 block.
    # We need 3 blocks.
    # Block 0: cols 0-1 of B, cols 0-1 of C
    # Block 1: cols 2-3 of B, cols 2-3 of C
    # Block 2: cols 4-5 of B, cols 4-5 of C
    
    # We need to map the local indices to global indices.
    # U is same for all blocks (A is shared).
    # V and W are shifted.
    
    final_U = []
    final_V = []
    final_W = []
    
    for block in range(3):
        # U is identical for all blocks
        for t in range(7):
            final_U.append(U[t])
        
        # V: shift columns by block*2
        for t in range(7):
            v_local = V[t]
            v_global = [0] * 12 # m*p = 2*6 = 12
            for j in range(2):
                for k_local in range(2):
                    k_global = block * 2 + k_local
                    v_global[j * 6 + k_global] = v_local[j * 2 + k_local]
            final_V.append(v_global)
            
        # W: shift columns by block*2
        for t in range(7):
            w_local = W[t]
            w_global = [0] * 12 # p*n = 6*2 = 12
            for i in range(2):
                for k_local in range(2):
                    k_global = block * 2 + k_local
                    w_global[k_global * 2 + i] = w_local[k_local * 2 + i]
            final_W.append(w_global)
            
    return {"U": final_U, "V": final_V, "W": final_W}
