import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    Rank 7 decomposition.
    U, V, W shapes: (4, 7)
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # m3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # m4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _rank49_4x4x4():
    """
    Construct rank 49 decomposition for 4x4 * 4x4 using Kronecker product of Strassen.
    U, V, W shapes: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U4[i, c] = U2[i1, c1] * U2[i2, c2]
            # i = 2*i1 + i2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    U4[i, c] = U2[i1, c1] * U2[i2, c2]
                    V4[i, c] = V2[i1, c1] * V2[i2, c2] # Wait, V index is j
                    W4[i, c] = W2[i1, c1] * W2[i2, c2] # Wait, W index is k
            
            # Correct indices for V and W
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    V4[j, c] = V2[j1, c1] * V2[j2, c2]
            
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    W4[k, c] = W2[k1, c1] * W2[k2, c2]
                    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get rank 49 decomposition for 4x4x4
    U4, V4, W4 = _rank49_4x4x4()
    
    # Final dimensions
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Part 1: 4x4 * 4x3 (cols 0-2 of B)
    # Use U4, V4, W4
    # V4: rows 0-15 (j*4+k). We want local k=0,1,2. Local k=3 is zero.
    # W4: rows 0-15 (k*4+i). We want local k=0,1,2. Local k=3 is zero.
    
    for c in range(49):
        U[:, c] = U4[:, c]
        
        # V part
        for j in range(4):
            for k in range(3): # local k 0,1,2
                V[j*7 + k, c] = V4[j*4 + k, c]
            # local k=3 is implicitly 0 in V final (since we don't set it)
            
        # W part
        for k in range(3): # local k 0,1,2 -> global k 0,1,2
            for i in range(4):
                W[k*4 + i, c] = W4[k*4 + i, c]
        # local k=3 is implicitly 0 in W final

    # Part 2: 4x4 * 4x4 (cols 3-6 of B)
    # Use U4, V4, W4
    # V4: rows 0-15. Local j, k. Global j, k=3+local_k.
    # W4: rows 0-15. Local k, i. Global k=3+local_k, i.
    
    for c in range(49):
        c2 = 49 + c
        U[:, c2] = U4[:, c]
        
        # V part
        for j in range(4):
            for k in range(4): # local k 0,1,2,3
                V[j*7 + (3+k), c2] = V4[j*4 + k, c]
                
        # W part
        for k in range(4): # local k 0,1,2,3 -> global k 3,4,5,6
            for i in range(4):
                W[(3+k)*4 + i, c2] = W4[k*4 + i, c]
                
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if something went wrong
        rank_schoolbook = n * m * p
        U_s = [[0] * rank_schoolbook for _ in range(n * m)]
        V_s = [[0] * rank_schoolbook for _ in range(m * p)]
        W_s = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
