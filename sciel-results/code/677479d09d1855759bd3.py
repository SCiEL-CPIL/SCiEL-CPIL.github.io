import time
import sys

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for k in {0, 1}
    # m1 = (a+d)(e+h)
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[10][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # m2 = (c+d)e
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1
    
    # m3 = a(f-h)
    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[10][r] = -1
    W[2][r] = 1
    
    # m4 = d(g-f)
    r = 3
    U[3][r] = 1
    V[9][r] = 1; V[1][r] = -1
    W[0][r] = 1; W[1][r] = 1
    
    # m5 = (a+b)h
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[10][r] = 1
    W[0][r] = -1; W[2][r] = 1
    
    # m6 = (c-a)(e+f)
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    
    # m7 = (b-d)(g+h)
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[9][r] = 1; V[10][r] = 1
    W[0][r] = 1
    
    # Schoolbook part for k in {2, ..., 8}
    r = 7
    for k in range(2, 9):
        # Term for j=0 (B[0, k])
        U[0][r] = 1
        U[2][r] = 1
        V[k][r] = 1  # index 0*p + k
        W[2*k][r] = 1
        W[2*k+1][r] = 1
        r += 1
        
        # Term for j=1 (B[1, k])
        U[1][r] = 1
        U[3][r] = 1
        V[9+k][r] = 1 # index 1*p + k
        W[2*k][r] = 1
        W[2*k+1][r] = 1
        r += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
