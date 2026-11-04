import numpy as np
import time

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in (0, 2):
        for k0 in (0, 2, 4, 6):
            j1 = j0 + 1
            k1 = k0 + 1
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            U[0*m + j0][col] = 1
            U[1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # Term 2: M2 = (A10+A11)B00
            U[1*m + j0][col] = 1
            U[1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 1][col] = 1
            W[k1*n + 1][col] = -1
            col += 1
            
            # Term 3: M3 = A00(B01-B11)
            U[0*m + j0][col] = 1
            V[j0*p + k1][col] = 1
            V[j1*p + k1][col] = -1
            W[k1*n + 0][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # Term 4: M4 = A11(B10-B00)
            U[1*m + j1][col] = 1
            V[j1*p + k0][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + 0][col] = 1
            W[k0*n + 1][col] = 1
            col += 1
            
            # Term 5: M5 = (A00+A01)B11
            U[0*m + j0][col] = 1
            U[0*m + j1][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + 0][col] = -1
            W[k1*n + 0][col] = 1
            col += 1
            
            # Term 6: M6 = (A10-A00)(B00+B01)
            U[1*m + j0][col] = 1
            U[0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # Term 7: M7 = (A01-A11)(B10+B11)
            U[0*m + j1][col] = 1
            U[1*m + j1][col] = -1
            V[j1*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
