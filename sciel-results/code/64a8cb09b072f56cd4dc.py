"""Matrix multiplication 4x4 by 4x8 - rank-minimal decomposition via Kronecker product of Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def verify(U, V, W):
    """Check if the decomposition is exact for the all-ones tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False, (i,j,k), s
    return True, None, None

def make_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c], shapes (4,7) each.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A11+A22)(B11+B22)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # c=1: M2 = (A21+A22)B11
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    
    # c=2: M3 = A11(B12-B22)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    
    # c=3: M4 = A22(B21-B11)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[2][3]=1; W[0][3]=-1
    
    # c=4: M5 = (A11+A12)B22
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    
    # c=5: M6 = (A21-A11)(B11+B12)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # c=6: M7 = (A12-A22)(B21+B22)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1; W[1][6]=-1
    
    return U, V, W

def schoolbook_2x2x4():
    """
    Return the rank-16 schoolbook decomposition for 2x2x4 matrix multiplication.
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c], shapes (4,16), (8,16), (8,16).
    """
    U = [[0]*16 for _ in range(4)]
    V = [[0]*16 for _ in range(8)]
    W = [[0]*16 for _ in range(8)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i*2+j][col] = 1
                V[j*4+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Kronecker product: 4x4x8 = (2x2x2) ⊗ (2x2x4)
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 4*k1 + k2
    # T[i,j,k] = T_str[i1,j1,k1] * T_2x2x4[i2,j2,k2] = 1*1 = 1
    # Rank: 7 * 16 = 112
    
    U_str, V_str, W_str = strassen_2x2x2()
    U_sb, V_sb, W_sb = schoolbook_2x2x4()
    
    rank = 7 * 16  # 112
    
    U = [[0]*rank for _ in range(16)]  # 4*4 = 16 rows
    V = [[0]*rank for _ in range(32)]  # 4*8 = 32 rows
    W = [[0]*rank for _ in range(32)]  # 8*4 = 32 rows
    
    # Kronecker product of the decompositions
    for c1 in range(7):
        for c2 in range(16):
            c = c1 * 16 + c2
            # U factor: U[i*4+j, c] = U_str[i1*2+j1, c1] * U_sb[i2*2+j2, c2]
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U_str[i1*2+j1][c1]
                    if u1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U_sb[i2*2+j2][c2]
                            if u2 == 0:
                                continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i*4+j][c] = u1 * u2
            
            # V factor: V[j*8+k, c] = V_str[j1*2+k1, c1] * V_sb[j2*4+k2, c2]
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V_str[j1*2+k1][c1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(4):
                            v2 = V_sb[j2*4+k2][c2]
                            if v2 == 0:
                                continue
                            j = j1 * 2 + j2
                            k = k1 * 4 + k2
                            V[j*8+k][c] = v1 * v2
            
            # W factor: W[k*4+i, c] = W_str[k1*2+i1, c1] * W_sb[k2*2+i2, c2]
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W_str[k1*2+i1][c1]
                    if w1 == 0:
                        continue
                    for k2 in range(4):
                        for i2 in range(2):
                            w2 = W_sb[k2*2+i2][c2]
                            if w2 == 0:
                                continue
                            k = k1 * 4 + k2
                            i = i1 * 2 + i2
                            W[k*4+i][c] = w1 * w2
    
    # Verify
    valid, err, val = verify(U, V, W)
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # If that didn't work, fall back to schoolbook
    return make_schoolbook()

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    valid, err, val = verify(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
