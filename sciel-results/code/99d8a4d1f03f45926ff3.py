import numpy as np

def _strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    A_specs = [
        [(1,0),(1,3)],
        [(1,2),(1,3)],
        [(1,0)],
        [(1,3)],
        [(1,0),(1,1)],
        [(1,2),(-1,0)],
        [(1,1),(-1,3)]
    ]
    B_specs = [
        [(1,0),(1,3)],
        [(1,0)],
        [(1,1),(-1,3)],
        [(1,2),(-1,0)],
        [(1,3)],
        [(1,0),(1,1)],
        [(1,2),(1,3)]
    ]
    C_specs = [
        [(1,0),(1,3)],
        [(1,2),(-1,3)],
        [(1,1),(1,3)],
        [(1,0),(1,2)],
        [(-1,0),(1,1)],
        [(1,3)],
        [(1,0)]
    ]
    for k in range(7):
        for m in range(7):
            ci = k*7 + m
            for cf, blk in A_specs[k]:
                bi,bj = divmod(blk,2)
                for lr in range(2):
                    for lc in range(2):
                        U[bi*2+lr, ci] += cf * U2[lr*2+lc, m]
            for cf, blk in B_specs[k]:
                bi,bj = divmod(blk,2)
                for lr in range(2):
                    for lc in range(2):
                        V[bi*2+lc, ci] += cf * V2[lr*2+lc, m]
            for cf, blk in C_specs[k]:
                bi,bj = divmod(blk,2)
                for lr in range(2):
                    for lc in range(2):
                        W[lc*4+bi*2+lr, ci] += cf * W2[lr*2+lc, m]
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    U4, V4, W4 = _build_4x4x4()
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    for i in range(16):
        r, c = divmod(i, 4)
        V[r*8+c, 0:49] = V4[i, :]
        V[r*8+c+4, 49:98] = V4[i, :]
    W[0:16, 0:49] = W4
    W[16:32, 49:98] = W4
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
