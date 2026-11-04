import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 8
    
    # 8 independent 2x2x2 Strassen blocks: rank = 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    col = 0
    
    # For each (j0, k0) block, apply Strassen to the 2x2x2 sub-problem
    # A block: rows i={0,1}, cols j={j0, j0+1}
    # B block: rows j={j0, j0+1}, cols k={k0, k0+1}
    # C block: rows i={0,1}, cols k={k0, k0+1}
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if _time_left() < 1.0:
                break
            
            j1 = j0 + 1
            k1 = k0 + 1
            
            # Strassen 7 multiplications for 2x2*2x2:
            # M1 = (A00+A11)*(B00+B11)
            # M2 = (A10+A11)*B00
            # M3 = A00*(B01-B11)
            # M4 = A11*(B10-B00)
            # M5 = (A00+A01)*B11
            # M6 = (A10-A00)*(B00+B01)
            # M7 = (A01-A11)*(B10+B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Term 1: M1 = (A00+A11)*(B00+B11)
            # U: A[0,j0]=1, A[1,j1]=1
            # V: B[j0,k0]=1, B[j1,k1]=1
            # W: C[0,k0]=1, C[1,k1]=1
            U[0*m + j0][col] = 1
            U[1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # Term 2: M2 = (A10+A11)*B00
            # U: A[1,j0]=1, A[1,j1]=1
            # V: B[j0,k0]=1
            # W: C[1,k0]=1, C[1,k1]=-1
            U[1*m + j0][col] = 1
            U[1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 1][col] = 1
            W[k1*n + 1][col] = -1
            col += 1
            
            # Term 3: M3 = A00*(B01-B11)
            # U: A[0,j0]=1
            # V: B[j0,k1]=1, B[j1,k1]=-1
            # W: C[0,k1]=1, C[1,k1]=1
            U[0*m + j0][col] = 1
            V[j0*p + k1][col] = 1
            V[j1*p + k1][col] = -1
            W[k1*n + 0][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # Term 4: M4 = A11*(B10-B00)
            # U: A[1,j1]=1
            # V: B[j1,k0]=1, B[j0,k0]=-1
            # W: C[0,k0]=1, C[1,k0]=1
            U[1*m + j1][col] = 1
            V[j1*p + k0][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + 0][col] = 1
            W[k0*n + 1][col] = 1
            col += 1
            
            # Term 5: M5 = (A00+A01)*B11
            # U: A[0,j0]=1, A[0,j1]=1
            # V: B[j1,k1]=1
            # W: C[0,k0]=-1, C[0,k1]=1
            U[0*m + j0][col] = 1
            U[0*m + j1][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + 0][col] = -1
            W[k1*n + 0][col] = 1
            col += 1
            
            # Term 6: M6 = (A10-A00)*(B00+B01)
            # U: A[1,j0]=1, A[0,j0]=-1
            # V: B[j0,k0]=1, B[j0,k1]=1
            # W: C[1,k1]=1
            U[1*m + j0][col] = 1
            U[0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # Term 7: M7 = (A01-A11)*(B10+B11)
            # U: A[0,j1]=1, A[1,j1]=-1
            # V: B[j1,k0]=1, B[j1,k1]=1
            # W: C[0,k0]=1
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
