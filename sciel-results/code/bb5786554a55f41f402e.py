import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 3, 8
    
    # 1. Construct M_{2x3x2} decomposition (Rank 11)
    # U232: 6x11, V232: 6x11, W232: 4x11
    U232 = np.zeros((6, 11), dtype=int)
    V232 = np.zeros((6, 11), dtype=int)
    W232 = np.zeros((4, 11), dtype=int)
    
    # Indices for A (2x3): 
    # a11=0, a12=1, a13=2
    # a21=3, a22=4, a23=5
    # Indices for B (3x2):
    # b11=0, b12=1
    # b21=2, b22=3
    # b31=4, b32=5
    # Indices for C (2x2):
    # c11=0, c12=1
    # c21=2, c22=3
    
    # Strassen Part (r=0..6)
    # M1: (a11+a22)(b11+b22) -> c11, c22
    U232[0, 0] = 1; U232[4, 0] = 1
    V232[0, 0] = 1; V232[3, 0] = 1
    W232[0, 0] = 1; W232[3, 0] = 1
    
    # M2: (a21+a22)b11 -> c21, -c22
    U232[3, 1] = 1; U232[4, 1] = 1
    V232[0, 1] = 1
    W232[2, 1] = 1; W232[3, 1] = -1
    
    # M3: a11(b12-b22) -> c12, c22
    U232[0, 2] = 1
    V232[1, 2] = 1; V232[3, 2] = -1
    W232[1, 2] = 1; W232[3, 2] = 1
    
    # M4: a22(b21-b11) -> c11, c21
    U232[4, 3] = 1
    V232[2, 3] = 1; V232[0, 3] = -1
    W232[0, 3] = 1; W232[2, 3] = 1
    
    # M5: (a11+a12)b22 -> -c11, c12
    U232[0, 4] = 1; U232[1, 4] = 1
    V232[3, 4] = 1
    W232[0, 4] = -1; W232[1, 4] = 1
    
    # M6: (a21-a11)(b11+b12) -> c22
    U232[3, 5] = 1; U232[0, 5] = -1
    V232[0, 5] = 1; V232[1, 5] = 1
    W232[3, 5] = 1
    
    # M7: (a12-a22)(b21+b22) -> c11, c22
    U232[1, 6] = 1; U232[4, 6] = -1
    V232[2, 6] = 1; V232[3, 6] = 1
    W232[0, 6] = 1; W232[3, 6] = 1
    
    # Outer Part (r=7..10)
    # M8: a13 b31 -> c11
    U232[2, 7] = 1
    V232[4, 7] = 1
    W232[0, 7] = 1
    
    # M9: a13 b32 -> c12
    U232[2, 8] = 1
    V232[5, 8] = 1
    W232[1, 8] = 1
    
    # M10: a23 b31 -> c21
    U232[5, 9] = 1
    V232[4, 9] = 1
    W232[2, 9] = 1
    
    # M11: a23 b32 -> c22
    U232[5, 10] = 1
    V232[5, 10] = 1
    W232[3, 10] = 1

    # 2. Construct M_{1x1x4} decomposition (Rank 4)
    # U114: 1x4, V114: 4x4, W114: 4x4
    U114 = np.ones((1, 4), dtype=int)
    V114 = np.eye(4, dtype=int)
    W114 = np.eye(4, dtype=int)

    # 3. Kronecker Product
    # U = U232 (6x11) otimes U114 (1x4) -> 6x44
    # V = V232 (6x11) otimes V114 (4x4) -> 24x44
    # W = W232 (4x11) otimes W114 (4x4) -> 16x44
    
    U = np.kron(U232, U114)
    V = np.kron(V232, V114)
    W = np.kron(W232, W114)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
