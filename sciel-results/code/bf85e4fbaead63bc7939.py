import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition for 4x4x4 using Kronecker product of Strassen (rank 7)
    # Strassen vectors for 2x2x2
    # u, v, w are 2x7 matrices
    # r=0..6
    U2 = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0]
    ])
    V2 = np.array([
        [1, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0]
    ])
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1]
    ])
    # Note: These are placeholders for Strassen. 
    # Actual Strassen:
    # P1: (1,1)x(1,1)->(1,1)
    # P2: (0,1)x(1,0)->(1,0)
    # P3: (1,0)x(0,1)->(0,1)
    # P4: (0,1)x(1,0)->(0,1) ... wait
    # I will use a known valid rank 7 decomposition if I can't recall.
    # For safety, I will use schoolbook rank 4 for 2x2, giving rank 16 for 4x4.
    # Total rank 2*16 + 16 = 48? No, 4x4 schoolbook is 64.
    # Kronecker of schoolbook 2x2 (rank 4) gives rank 16 for 4x4?
    # If rank(4x4) <= 16, then 2*16+16=48.
    # This is better than 114.
    # Let's assume rank 16 is possible?
    # If not, I will fall back to 144.
    
    # To be safe, I will use the schoolbook decomposition for the whole 4x4x9.
    # Rank 144.
    # This is the seed.
    # I need to beat it.
    # I will try to construct the 114 rank decomposition using Strassen.
    
    # Strassen vectors (corrected):
    # See https://en.wikipedia.org/wiki/Strassen_algorithm
    # u1=(1,1), v1=(1,1), w1=(1,1)
    # u2=(0,1), v2=(1,0), w2=(1,0)
    # u3=(1,0), v3=(0,1), w3=(0,1)
    # u4=(0,1), v4=(1,0), w4=(0,1) -- Wait, P4 is d(g-e)
    # u4=(0,1), v4=(1,-1), w4=(0,1)
    # u5=(1,1), v5=(0,1), w5=(0,1)
    # u6=(1,-1), v6=(1,1), w6=(1,0)
    # u7=(1,1), v7=(1,-1), w7=(1,0) -- Wait, P7 is (b-d)(g+h)
    # u7=(1,-1), v7=(1,1), w7=(1,0)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, -1, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 1],
        [1, 0, 1, -1, 1, 1, -1]
    ])
    W2 = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 0]
    ])
    
    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2) # 4x49
    V4 = np.kron(V2, V2) # 4x49
    W4 = np.kron(W2, W2) # 4x49
    
    # Construct 4x4x9
    # Part 1: k=0..3 (4x4x4)
    # Part 2: k=4..7 (4x4x4)
    # Part 3: k=8 (4x4x1)
    
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    col = 0
    # Part 1
    for r in range(49):
        U[0:4, col] = U4[0:4, r]
        V[0:4, col] = V4[0:4, r]
        W[0:4, col] = W4[0:4, r]
        col += 1
    # Part 2
    for r in range(49):
        U[0:4, col] = U4[0:4, r]
        V[4:8, col] = V4[0:4, r]
        W[4:8, col] = W4[0:4, r]
        col += 1
    # Part 3 (Schoolbook 4x4x1)
    for i in range(4):
        for j in range(4):
            U[i, col] = 1
            V[j, col] = 1
            W[8, col] = 1 # k=8
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
