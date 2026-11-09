import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    rank = 17
    U = [[0]*rank for _ in range(n*m)]  # 6x17
    V = [[0]*rank for _ in range(m*p)]  # 9x17
    W = [[0]*rank for _ in range(p*n)]  # 6x17
    
    # Part 1: Strassen 2x2x2 for A[:,0:2] * B[0:2,0:2]
    # U indices: A[i,j] -> i*m+j, i=0,1; j=0,1 -> 0,1,2,3
    # V indices: B[j,k] -> j*p+k, j=0,1; k=0,1 -> 0,1,3,4
    # W indices: C[k,i] -> k*n+i, k=0,1; i=0,1 -> 0,1,2,3
    
    # M1 = a00*(b00+b11)
    U[0][0] = 1
    V[0][0] = 1
    V[4][0] = 1
    W[0][0] = 1
    W[2][0] = 1
    
    # M2 = (a00+a01)*b11
    U[0][1] = 1
    U[1][1] = 1
    V[4][1] = 1
    W[1][1] = 1
    W[3][1] = -1
    
    # M3 = (a10+a11)*b00
    U[2][2] = 1
    U[3][2] = 1
    V[0][2] = 1
    W[2][2] = 1
    W[3][2] = 1
    
    # M4 = a11*(b01-b10)
    U[3][3] = 1
    V[1][3] = 1
    V[3][3] = -1
    W[0][3] = 1
    W[1][3] = 1
    
    # M5 = (a00+a10)*(b01+b10)
    U[0][4] = 1
    U[2][4] = 1
    V[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[3][4] = 1
    
    # M6 = (a01-a10)*(b00+b10)
    U[1][5] = 1
    U[2][5] = -1
    V[0][5] = 1
    V[3][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][6] = 1
    U[3][6] = -1
    V[3][6] = 1
    V[4][6] = 1
    W[0][6] = 1
    
    # Part 2: A[:,0:2] * B[0:2,2] (2x2x1, rank 4)
    for i in range(2):
        for j in range(2):
            col = 7 + i*2 + j
            U[i*m + j][col] = 1
            V[j*p + 2][col] = 1
            W[2*n + i][col] = 1
    
    # Part 3: A[:,2] * B[2,:] (2x1x3, rank 6)
    for i in range(2):
        for k in range(3):
            col = 11 + i*3 + k
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    expected = np.ones((n, m, p))
    print("Max error:", np.max(np.abs(T - expected)))
    print("Rank:", rank)
