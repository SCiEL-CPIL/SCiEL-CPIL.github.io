import time
import numpy as np

DIMENSIONS = [2, 3, 7]
_TIME_LIMIT = 120.0
_START_TIME = time.time()

def _remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _strassen_2_2_2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A00 + A11)(B00 + B11)
    # u: (0,0)=1, (1,1)=1
    # v: (0,0)=1, (1,1)=1
    # w: (0,0)=1, (1,1)=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # u: (1,0)=1, (1,1)=1
    # v: (0,0)=1
    # w: (1,0)=1, (1,1)=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    # u: (0,0)=1
    # v: (0,1)=1, (1,1)=-1
    # w: (0,1)=1, (1,1)=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    # u: (1,1)=1
    # v: (1,0)=1, (0,0)=-1
    # w: (0,0)=1, (1,0)=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    # u: (0,0)=1, (0,1)=1
    # v: (1,1)=1
    # w: (0,0)=-1, (0,1)=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # u: (1,0)=1, (0,0)=-1
    # v: (0,0)=1, (0,1)=1
    # w: (1,1)=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # u: (0,1)=1, (1,1)=-1
    # v: (1,0)=1, (1,1)=1
    # w: (0,0)=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _combine_p(D1, D2, n, m):
    # D1: M_{n, m, p1}, D2: M_{n, m, p2}
    U1, V1, W1 = D1
    U2, V2, W2 = D2
    p1 = V1.shape[1]
    p2 = V2.shape[1]
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    U = np.zeros((U1.shape[0], r1 + r2))
    V = np.zeros((m * (p1 + p2), r1 + r2))
    W = np.zeros((W1.shape[0], r1 + r2))
    
    U[:, :r1] = U1
    V[:, :p1] = V1
    W[:, :r1] = W1
    
    U[:, r1:] = U2
    V[:, p1:] = V2
    W[:, r1:] = W2
    
    return U, V, W

def _combine_m(D1, D2, n, p):
    # D1: M_{n, m1, p}, D2: M_{n, m2, p}
    U1, V1, W1 = D1
    U2, V2, W2 = D2
    m1 = U1.shape[1]
    m2 = U2.shape[1]
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    U = np.zeros((n * (m1 + m2), r1 + r2))
    V = np.zeros(((m1 + m2) * p, r1 + r2))
    W = np.zeros((W1.shape[0], r1 + r2))
    
    U[:, :m1] = U1
    V[:m1, :] = V1
    W[:, :r1] = W1
    
    U[:, m1:] = U2
    V[m1:, :] = V2
    W[:, r1:] = W2
    
    return U, V, W

def construct():
    # Build M_{2,3,7}
    # M_{2,2,1}
    D_2_2_1 = _schoolbook(2, 2, 1)
    # M_{2,2,2}
    D_2_2_2 = _strassen_2_2_2()
    # M_{2,2,3} = M_{2,2,2} + M_{2,2,1} (shifted p)
    D_2_2_3 = _combine_p(D_2_2_2, D_2_2_1, 2, 2)
    # M_{2,2,5} = M_{2,2,2} + M_{2,2,3} (shifted p)
    D_2_2_5 = _combine_p(D_2_2_2, D_2_2_3, 2, 2)
    # M_{2,2,7} = M_{2,2,2} + M_{2,2,5} (shifted p)
    D_2_2_7 = _combine_p(D_2_2_2, D_2_2_5, 2, 2)
    # M_{2,1,7}
    D_2_1_7 = _schoolbook(2, 1, 7)
    # M_{2,3,7} = M_{2,2,7} + M_{2,1,7} (shifted m)
    D_final = _combine_m(D_2_2_7, D_2_1_7, 2, 7)
    
    U, V, W = D_final
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
