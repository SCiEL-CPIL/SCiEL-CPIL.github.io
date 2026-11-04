"""Matrix multiplication 2x4x4 with rank-16 decomposition."""
import time

DIMENSIONS = [2, 4, 4]


def construct():
    """
    Construct a rank-16 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Splits the computation into two independent 2x2 * 2x4 multiplications:
    1. A[:, 0:2] * B[0:2, :]
    2. A[:, 2:4] * B[2:4, :]
    
    Each 2x2 * 2x4 multiplication has rank 8 (schoolbook).
    Total rank: 8 + 8 = 16.
    
    Interface:
      U shape: (n*m, rank) = (8, 16)
      V shape: (m*p, rank) = (16, 16)
      W shape: (p*n, rank) = (8, 16)
    """
    t0 = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 16
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 16
    V = [[0] * rank for _ in range(m * p)]   # 16 x 16
    W = [[0] * rank for _ in range(p * n)]   # 8 x 16
    
    col = 0
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    # A indices: i in {0,1}, j in {0,1} -> U[i*4+j]
    # B indices: j in {0,1}, k in {0,1,2,3} -> V[j*4+k]
    # C indices: i in {0,1}, k in {0,1,2,3} -> W[k*2+i]
    
    for i in range(2):
        for j in range(2):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 2 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                if time.time() - t0 > deadline - 5:
                    break
            if time.time() - t0 > deadline - 5:
                break
        if time.time() - t0 > deadline - 5:
            break
    
    if time.time() - t0 > deadline - 5:
        return {"U": U, "V": V, "W": W}

    # Block 2: A[:, 2:4] * B[2:4, :]
    # A indices: i in {0,1}, j in {2,3} -> U[i*4+j]
    # B indices: j in {2,3}, k in {0,1,2,3} -> V[j*4+k]
    # C indices: i in {0,1}, k in {0,1,2,3} -> W[k*2+i]
    
    for i in range(2):
        for j in range(2, 4):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 2 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                if time.time() - t0 > deadline - 5:
                    break
            if time.time() - t0 > deadline - 5:
                break
        if time.time() - t0 > deadline - 5:
            break
            
    return {"U": U, "V": V, "W": W}
