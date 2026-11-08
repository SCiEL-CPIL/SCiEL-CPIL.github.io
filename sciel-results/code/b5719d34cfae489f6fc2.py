import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Uses the rank-49 decomposition based on recursive Strassen's algorithm.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Define the rank-7 decomposition for 2x2 matrix multiplication (Strassen)
    # Indices for 2x2: 00, 01, 10, 11
    # U2, V2, W2 have shape (4, 7)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: P1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1  # C00 += P1
    W2[3, 0] = 1  # C11 += P1
    
    # Term 1: P2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1  # C10 += P2
    W2[3, 1] = -1 # C11 -= P2
    
    # Term 2: P3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1  # C01 += P3
    W2[3, 2] = 1  # C11 += P3
    
    # Term 3: P4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1  # C00 += P4
    W2[2, 3] = 1  # C10 += P4
    
    # Term 4: P5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00 -= P5
    W2[1, 4] = 1  # C01 += P5
    
    # Term 5: P6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1  # C11 += P6
    
    # Term 6: P7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1  # C00 += P7
    
    # Construct rank-49 decomposition for 4x4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for t1 in range(7):
        for t2 in range(7):
            t = t1 * 7 + t2
            for idx in range(16):
                i = idx // 4
                j = idx % 4
                
                br = i // 2
                sr = i % 2
                bc = j // 2
                sc = j % 2
                
                # Block index in 2x2 space: br*2 + bc
                # Scalar index in 2x2 space: sr*2 + sc
                b_idx = br * 2 + bc
                s_idx = sr * 2 + sc
                
                U[idx, t] = U2[b_idx, t1] * U2[s_idx, t2]
                V[idx, t] = V2[b_idx, t1] * V2[s_idx, t2]
                W[idx, t] = W2[b_idx, t1] * W2[s_idx, t2]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
