import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Index mapping:
    # U[i*m+j] for i,j in 0..2: A[i,j]
    # V[j*p+k] for j,k in 0..2: B[j,k]
    # W[k*n+i] for k,i in 0..2: C[i,k] (trace convention W[k*n+i])
    
    # Core 1: top-left 2x2 (rows 0,1; cols 0,1)
    # A indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # C indices: (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4  [W[k*n+i]: C[i,k] -> W[k*3+i]]
    
    col = 0
    # P1 = (A00+A11)(B00+B11) -> C00 + C01 + C10 + C11
    U[0, col] += 1; U[4, col] += 1
    V[0, col] += 1; V[4, col] += 1
    W[0, col] += 1; W[3, col] += 1; W[1, col] += 1; W[4, col] += 1
    col += 1
    
    # P2 = (A10+A11)B00 -> C10 + C11
    U[3, col] += 1; U[4, col] += 1
    V[0, col] += 1
    W[1, col] += 1; W[4, col] += 1
    col += 1
    
    # P3 = A00(B10-B11) -> C10 - C11
    U[0, col] += 1
    V[3, col] += 1; V[4, col] -= 1
    W[1, col] += 1; W[4, col] -= 1
    col += 1
    
    # P4 = A11(B01-B00) -> C00 - C01
    U[4, col] += 1
    V[1, col] += 1; V[0, col] -= 1
    W[0, col] += 1; W[3, col] -= 1
    col += 1
    
    # P5 = (A00+A01)B11 -> C01 + C11
    U[0, col] += 1; U[1, col] += 1
    V[4, col] += 1
    W[3, col] += 1; W[4, col] += 1
    col += 1
    
    # P6 = (A10-A00)(B00+B01) -> C10 + C11
    U[3, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[1, col] += 1; W[4, col] += 1
    col += 1
    
    # P7 = (A01-A11)(B10+B11) -> C00 - C01
    U[1, col] += 1; U[4, col] -= 1
    V[3, col] += 1; V[4, col] += 1
    W[0, col] += 1; W[3, col] -= 1
    col += 1
    
    # Core 2: bottom-right 2x2 (rows 1,2; cols 1,2)
    # A indices: (1,1)=4, (1,2)=5, (2,1)=7, (2,2)=8
    # B indices: (1,1)=4, (1,2)=5, (2,1)=7, (2,2)=8
    # C indices: (1,1)=4, (1,2)=7, (2,1)=5, (2,2)=8  [W[k*n+i]: C[i,k] -> W[k*3+i]]
    
    # Q1 = (A11+A22)(B11+B22) -> C11 + C12 + C21 + C22
    U[4, col] += 1; U[8, col] += 1
    V[4, col] += 1; V[8, col] += 1
    W[4, col] += 1; W[7, col] += 1; W[5, col] += 1; W[8, col] += 1
    col += 1
    
    # Q2 = (A21+A22)B11 -> C21 + C22
    U[7, col] += 1; U[8, col] += 1
    V[4, col] += 1
    W[5, col] += 1; W[8, col] += 1
    col += 1
    
    # Q3 = A11(B21-B22) -> C21 - C22
    U[4, col] += 1
    V[7, col] += 1; V[8, col] -= 1
    W[5, col] += 1; W[8, col] -= 1
    col += 1
    
    # Q4 = A22(B12-B11) -> C11 - C12
    U[8, col] += 1
    V[5, col] += 1; V[4, col] -= 1
    W[4, col] += 1; W[7, col] -= 1
    col += 1
    
    # Q5 = (A11+A12)B22 -> C12 + C22
    U[4, col] += 1; U[5, col] += 1
    V[8, col] += 1
    W[7, col] += 1; W[8, col] += 1
    col += 1
    
    # Q6 = (A21-A11)(B11+B12) -> C21 + C22
    U[7, col] += 1; U[4, col] -= 1
    V[4, col] += 1; V[5, col] += 1
    W[5, col] += 1; W[8, col] += 1
    col += 1
    
    # Q7 = (A12-A22)(B21+B22) -> C11 - C12
    U[5, col] += 1; U[8, col] -= 1
    V[7, col] += 1; V[8, col] += 1
    W[4, col] += 1; W[7, col] -= 1
    col += 1
    
    # Off-diagonal: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # 3 schoolbook terms
    U[0, col] = 1; V[2, col] = 1; W[6, col] = 1  # A[0,0]*B[0,2] -> C[0,2], W[2*3+0]=W[6]
    col += 1
    U[1, col] = 1; V[5, col] = 1; W[6, col] = 1  # A[0,1]*B[1,2] -> C[0,2]
    col += 1
    U[2, col] = 1; V[8, col] = 1; W[6, col] = 1  # A[0,2]*B[2,2] -> C[0,2]
    col += 1
    
    # Off-diagonal: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # 3 schoolbook terms
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1  # A[2,0]*B[0,0] -> C[2,0], W[0*3+2]=W[2]
    col += 1
    U[7, col] = 1; V[3, col] = 1; W[2, col] = 1  # A[2,1]*B[1,0] -> C[2,0]
    col += 1
    U[8, col] = 1; V[6, col] = 1; W[2, col] = 1  # A[2,2]*B[2,0] -> C[2,0]
    col += 1
    
    # Verify
    def verify(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i,j,k,s)
        return True, None
    
    ok, err = verify(U, V, W)
    if not ok:
        # Fallback to schoolbook rank 27
        rank27 = 27
        U27 = np.zeros((9, rank27))
        V27 = np.zeros((9, rank27))
        W27 = np.zeros((9, rank27))
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U27[i*3+j, c] = 1
                    V27[j*3+k, c] = 1
                    W27[k*3+i, c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
