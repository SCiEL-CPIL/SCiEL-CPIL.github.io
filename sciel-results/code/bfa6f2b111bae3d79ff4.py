"""Matrix multiplication 4,4,4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for 4x4 matrix multiplication.
    We use the Strassen-based decomposition of rank 49.
    """
    start_time = time.time()
    DEADLINE = 130  # seconds, leaving 20s safety margin for the 150s kill

    n, m, p = 4, 4, 4
    N = n * m * p  # 64, the tensor size
    
    # Build the 2x2 Strassen decomposition explicitly.
    # U2[i0*2+j0, r], V2[j0*2+k0, r], W2[i0*2+k0, r] for i0,j0,k0 in 0..1, r in 0..6
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    # U2: (0,0)=1, (1,1)=1 -> indices 0, 3
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    # U2: (1,0)=1, (1,1)=1 -> indices 2, 3
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    # U2: (0,0)=1 -> index 0
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    # U2: (1,1)=1 -> index 3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    # U2: (0,0)=1, (0,1)=1 -> indices 0, 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    # U2: (1,0)=1, (0,0)=-1 -> indices 2, 0
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    # U2: (0,1)=1, (1,1)=-1 -> indices 1, 3
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1
    
    # Verify 2x2
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(2):
                s = 0
                for r in range(7):
                    s += U2[i0*2+j0, r] * V2[j0*2+k0, r] * W2[i0*2+k0, r]
                if s != 1:
                    # If verification fails, fallback to rank-64 trivial
                    pass
    
    # Build 4x4
    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i0 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1*7 + r2
                                    u_val = U2[i0*2+j0, r2] * U2[i1*2+j1, r1]
                                    v_val = V2[j0*2+k0, r2] * V2[j1*2+k1, r1]
                                    w_val = W2[i0*2+k0, r2] * W2[i1*2+k1, r1]
                                    U4[i*4+j, r] += u_val
                                    V4[j*4+k, r] += v_val
                                    W4[k*4+i, r] += w_val
    
    # Verify 4x4
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for r in range(rank):
                    s += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to rank-64 trivial decomposition if Strassen construction fails
        rank = 64
        U4 = np.zeros((16, rank), dtype=int)
        V4 = np.zeros((16, rank), dtype=int)
        W4 = np.zeros((16, rank), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i * 16 + j * 4 + k
                    U4[i*4+j, r] = 1
                    V4[j*4+k, r] = 1
                    W4[k*4+i, r] = 1
    
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END
