import time
import numpy as np

DIMENSIONS = [2, 3, 7]
_TIME_LIMIT = 120.0
_START_TIME = time.time()

def _remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _additive_combine(U1, V1, W1, U2, V2, W2):
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    U = np.zeros((U1.shape[0], r1 + r2))
    V = np.zeros((V1.shape[0], r1 + r2))
    W = np.zeros((W1.shape[0], r1 + r2))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W

def _rectangular_2_1_7():
    n, m, p = 2, 1, 7
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

def _rectangular_2_2_1():
    n, m, p = 2, 2, 1
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

def _rectangular_2_2_2():
    # Known rank 7 decomposition for 2x2x2
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices: U is 4x7, V is 4x7, W is 4x7
    # Using the standard Strassen-like decomposition adapted for 2x2x2
    # The 7 terms correspond to:
    # 1: (1,1), (1,1), (1,1)  -> u0, v0, w0
    # 2: (1,2), (1,1), (1,2)  -> u1, v0, w1
    # 3: (2,1), (1,1), (2,1)  -> u2, v0, w2
    # 4: (2,2), (1,1), (2,2)  -> u3, v0, w3
    # 5: (1,1), (2,2), (1,1)  -> u0, v3, w0
    # 6: (1,2), (2,2), (1,2)  -> u1, v3, w1
    # 7: (2,1), (2,2), (2,1)  -> u2, v3, w2
    # Wait, this is not quite right. Let's use a known explicit decomposition.
    # Actually, the rank of 2x2x2 is 7. The decomposition can be constructed
    # from the 8 schoolbook terms by eliminating one.
    # Let's use a known set of coefficients.
    
    # Term 0: i=0,j=0; k=0 -> U[0,0]=1, V[0,0]=1, W[0,0]=1
    # Term 1: i=0,j=1; k=0 -> U[1,1]=1, V[0,1]=1, W[0,1]=1
    # Term 2: i=1,j=0; k=0 -> U[2,2]=1, V[0,2]=1, W[0,2]=1
    # Term 3: i=1,j=1; k=0 -> U[3,3]=1, V[0,3]=1, W[0,3]=1
    # Term 4: i=0,j=0; k=1 -> U[0,4]=1, V[2,4]=1, W[1,4]=1
    # Term 5: i=0,j=1; k=1 -> U[1,5]=1, V[2,5]=1, W[1,5]=1
    # Term 6: i=1,j=0; k=1 -> U[2,6]=1, V[2,6]=1, W[1,6]=1
    # This is rank 7, but we need to check if it's valid.
    # Actually, a simpler way: we can just use the schoolbook rank 8 for 2x2x2 if we don't have a clean rank 7.
    # But R(2,2,2)=7. Let's try to implement the known rank 7.
    
    # From literature, a rank 7 decomposition for 2x2x2:
    # w0 = u0 v0
    # w1 = u1 v1
    # w2 = u2 v2
    # w3 = u3 v3
    # w4 = (u0+u1+u2+u3) (v0+v1+v2+v3)
    # w5 = (u0+u1) (v2+v3)
    # w6 = (u2+u3) (v0+v1)
    # This doesn't seem to work directly as a bilinear form in the standard sense.
    
    # Let's stick to the schoolbook for 2x2x2 to ensure correctness, then add.
    # Rank 8 for 2x2x2.
    rank = 8
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

def _rectangular_2_2_3():
    # Use additive: R(2,2,3) <= R(2,2,2) + R(2,2,1)
    U1, V1, W1 = _rectangular_2_2_2()
    U2, V2, W2 = _rectangular_2_2_1()
    # But we need to align indices.
    # 2x2x2 uses k=0,1. 2x2x1 uses k=0.
    # For the additive construction of M_{2,2,3}, we can split the k-index.
    # M_{2,2,3} = M_{2,2,2} + M_{2,2,1} (shifted)
    # The second part uses k=2.
    # U for 2x2x1 has indices i*m+j. V has j*p+k. W has k*n+i.
    # For the shifted part, we add 2 to the k-index in V and W.
    U2_shifted = U2.copy()
    V2_shifted = V2.copy()
    W2_shifted = W2.copy()
    # V2_shifted indices: j*1 + 0 -> j*3 + 2
    # W2_shifted indices: 0*2 + i -> 2*2 + i
    V2_new = np.zeros((2 * 3, V2_shifted.shape[1]))
    W2_new = np.zeros((3 * 2, W2_shifted.shape[1]))
    for j in range(2):
        V2_new[j * 3 + 2, :] = V2_shifted[j * 1 + 0, :]
    for i in range(2):
        W2_new[2 * 2 + i, :] = W2_shifted[0 * 2 + i, :]
    
    U, V, W = _additive_combine(U1, V1, W1, U2_new, V2_new, W2_new)
    return U, V, W

def construct():
    # We will construct M_{2,3,7} by splitting the m-index (3) into 1 and 2.
    # M_{2,3,7} = M_{2,1,7} + M_{2,2,7}
    # M_{2,2,7} will be constructed via R(2,2,7) <= R(2,2,2) + R(2,2,5)
    # R(2,2,5) <= R(2,2,2) + R(2,2,3)
    # R(2,2,3) <= R(2,2,2) + R(2,2,1)
    # This is getting complex. Let's just use the schoolbook for the whole thing to be safe,
    # or a simpler split.
    # Actually, the simplest guaranteed improvement is to use the additive bound for m.
    # R(2,3,7) <= R(2,1,7) + R(2,2,7)
    # If we use schoolbook for both: 14 + 28 = 42. No improvement.
    # We need a better bound for R(2,2,7).
    # R(2,2,7) <= R(2,2,2) + R(2,2,5) <= 7 + 16 = 23? (Using known ranks)
    # But implementing the exact decompositions is hard.
    
    # Let's try a different split: split p=7 into p=2 and p=5.
    # R(2,3,7) <= R(2,3,2) + R(2,3,5)
    # R(2,3,2) <= 2*3*2 = 12 (schoolbook)
    # R(2,3,5) <= 2*3*5 = 30 (schoolbook)
    # Total 42.
    
    # It seems without specific low-rank decompositions, it's hard to beat 42.
    # However, the previous attempt was rejected for being "empty" or invalid.
    # I will return the schoolbook decomposition to ensure validity.
    n, m, p = DIMENSIONS
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
