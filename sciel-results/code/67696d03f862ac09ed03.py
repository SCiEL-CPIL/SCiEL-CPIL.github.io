import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2
    Us = np.zeros((4, 7))
    Vs = np.zeros((4, 7))
    Ws = np.zeros((4, 7))
    
    Us[0,0] = 1; Us[3,0] = 1
    Vs[0,0] = 1; Vs[3,0] = 1
    Ws[0,0,0] = 1; Ws[1,1,0] = 1
    
    Us[2,1] = 1; Us[3,1] = 1
    Vs[0,1] = 1
    Ws[0,1,1] = 1; Ws[1,1,1] = -1
    
    Us[0,2] = 1
    Vs[1,2] = 1; Vs[3,2] = -1
    Ws[1,0,2] = 1; Ws[1,1,2] = 1
    
    Us[3,3] = 1
    Vs[2,3] = 1; Vs[0,3] = -1
    Ws[0,0,3] = 1; Ws[0,1,3] = 1
    
    Us[0,4] = 1; Us[1,4] = 1
    Vs[3,4] = 1
    Ws[0,0,4] = -1; Ws[1,1,4] = 1
    
    Us[2,5] = 1; Us[0,5] = -1
    Vs[0,5] = 1; Vs[1,5] = 1
    Ws[1,1,5] = 1
    
    Us[1,6] = 1; Us[3,6] = -1
    Vs[2,6] = 1; Vs[3,6] = 1
    Ws[0,0,6] = 1; Ws[1,0,6] = 1

    # 2x2x1 matrices (Rank 4)
    U21 = np.zeros((4, 4))
    V21 = np.zeros((2, 4))
    W21 = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            c = i*2 + j
            U21[i*2+j, c] = 1
            V21[j, c] = 1
            W21[i, c] = 1

    # 4x4x4 (Rank 49)
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    for c1 in range(7):
        for c2 in range(7):
            c = c1*7 + c2
            U444[:, c] = np.kron(Us[:, c1], Us[:, c2])
            V444[:, c] = np.kron(Vs[:, c1], Vs[:, c2])
            W444[:, c] = np.kron(Ws[:, c1], Ws[:, c2])

    # 4x4x2 (Rank 28)
    U442 = np.zeros((16, 28))
    V442 = np.zeros((8, 28))
    W442 = np.zeros((8, 28))
    for c1 in range(7):
        for c2 in range(4):
            c = c1*4 + c2
            U442[:, c] = np.kron(Us[:, c1], U21[:, c2])
            V442[:, c] = np.kron(Vs[:, c1], V21[:, c2])
            W442[:, c] = np.kron(Ws[:, c1], W21[:, c2])

    # 4x4x1 (Rank 16)
    U441 = np.zeros((16, 16))
    V441 = np.zeros((4, 16))
    W441 = np.zeros((4, 16))
    for c1 in range(4):
        for c2 in range(4):
            c = c1*4 + c2
            U441[:, c] = np.kron(U21[:, c1], U21[:, c2])
            V441[:, c] = np.kron(V21[:, c1], V21[:, c2])
            W441[:, c] = np.kron(W21[:, c1], W21[:, c2])

    # Assemble U, V, W
    # U: 16x93
    U = np.hstack([U444, U442, U441])
    
    # V: 28x93
    # V444: 16x49 (j=0..3, k=0..3)
    # V442: 8x28 (j=0..3, k=0..1) -> shift k by 4
    # V441: 4x16 (j=0..3, k=0) -> shift k by 6
    
    V = np.zeros((28, 93))
    V[0:16, 0:49] = V444
    # V442 rows: j*2 + k? No, V is (m*p, rank) = (4*7, rank).
    # V[j*7+k, c].
    # V444: j=0..3, k=0..3. Rows 0..15.
    # V442: j=0..3, k=0..1. Rows 4*7+0 .. 4*7+1? No.
    # V[j*7+k, c].
    # For V442, k is 0..1. So rows are j*7+0, j*7+1.
    # We need to map V442[j*2+k, c] to V[j*7+k+4, c].
    for j in range(4):
        for k in range(2):
            V[j*7+k+4, 49:77] = V442[j*2+k, :]
            
    # V441: j=0..3, k=0. Rows j*7+0.
    # We need to map V441[j*1+0, c] to V[j*7+0+6, c].
    for j in range(4):
        V[j*7+6, 77:93] = V441[j, :]

    # W: 28x93
    # W444: 16x49 (k=0..3, i=0..3)
    # W442: 8x28 (k=0..1, i=0..3) -> shift k by 4
    # W441: 4x16 (k=0, i=0..3) -> shift k by 6
    
    W = np.zeros((28, 93))
    W[0:16, 0:49] = W444
    for k in range(2):
        for i in range(4):
            W[k*4+i+4*4, 49:77] = W442[k*4+i, :] # W442 is 8x28. k=0..1, i=0..3.
            # W442[k*4+i, c].
            # W[k*4+i+16, c].
            
    for i in range(4):
        W[0*4+i+6*4, 77:93] = W441[i, :] # W441 is 4x16. i=0..3.
        # W441[i, c].
        # W[i+24, c].

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
