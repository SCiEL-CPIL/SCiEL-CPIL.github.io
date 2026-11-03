import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5

def _validate(U, V, W, tol=1e-9):
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_block_strassen3x3():
    """
    Returns rank-24 decomposition for 3x3 matrix multiplication using Strassen for the 2x2 core.
    U, V, W are 9x24.
    Indexing: U[i*3+j], V[j*3+k], W[k*3+i].
    """
    U = np.zeros((9, 24))
    V = np.zeros((9, 24))
    W = np.zeros((9, 24))
    r = 0

    # Block C11 = A11*B11 + A12*B21
    # A11 (2x2) at A[0,1,3,4], B11 (2x2) at B[0,1,3,4], C11 (2x2) part at C[0,1,3,4]
    # Strassen 2x2 for A11*B11
    
    # M1 = (A11[0,0]+A11[1,1]) * (B11[0,0]-B11[1,1])
    # U: A[0]=1, A[4]=1. V: B[0]=1, B[4]=-1.
    # W: C[0]=1, C[4]=-1, C[1]=-1, C[3]=1. (Standard Strassen reconstruction: M1 goes to C00, C11? 
    # Let's check standard Strassen:
    # M1 = (A11+A22)(B11-B22). C11 = M1+M2+M4-M5. C12 = M3+M5. C21 = M2-M4+M6. C22 = M1-M2+M3+M7.
    # My indices: A11=0, A22=4, B11=0, B22=4.
    # C11=0, C12=1, C21=3, C22=4.
    
    # M1
    U[0, r]=1; U[4, r]=1
    V[0, r]=1; V[4, r]=-1
    W[0, r]=1; W[1, r]=0; W[3, r]=0; W[4, r]=-1
    r+=1
    
    # M2 = A22*(B11+B22)
    U[4, r]=1
    V[0, r]=1; V[4, r]=1
    W[0, r]=-1; W[1, r]=0; W[3, r]=1; W[4, r]=-1
    r+=1
    
    # M3 = (A11-A22)*B22
    U[0, r]=1; U[4, r]=-1
    V[4, r]=1
    W[0, r]=0; W[1, r]=1; W[3, r]=0; W[4, r]=1
    r+=1
    
    # M4 = A11*B11
    U[0, r]=1
    V[0, r]=1
    W[0, r]=1; W[1, r]=0; W[3, r]=-1; W[4, r]=0
    r+=1
    
    # M5 = (A11+A22)*B22
    U[0, r]=1; U[4, r]=1
    V[4, r]=1
    W[0, r]=-1; W[1, r]=1; W[3, r]=0; W[4, r]=0
    r+=1
    
    # M6 = (A22-A11)*(B11+B22)
    U[0, r]=-1; U[4, r]=1
    V[0, r]=1; V[4, r]=1
    W[0, r]=0; W[1, r]=0; W[3, r]=1; W[4, r]=0
    r+=1
    
    # M7 = (A11-A22)*(B22-B11)
    U[0, r]=1; U[4, r]=-1
    V[0, r]=-1; V[4, r]=1
    W[0, r]=0; W[1, r]=0; W[3, r]=0; W[4, r]=1
    r+=1

    # Now A12*B21 part of C11.
    # A12 (2x1) at A[2,5]. B21 (1x2) at B[6,7]. C11 (2x2) at C[0,1,3,4].
    # C11[0,0] += A12[0]*B21[0] -> U[2]=1, V[6]=1, W[0]=1
    U[2, r]=1; V[6, r]=1; W[0, r]=1
    r+=1
    # C11[0,1] += A12[0]*B21[1] -> U[2]=1, V[7]=1, W[1]=1
    U[2, r]=1; V[7, r]=1; W[1, r]=1
    r+=1
    # C11[1,0] += A12[1]*B21[0] -> U[5]=1, V[6]=1, W[3]=1
    U[5, r]=1; V[6, r]=1; W[3, r]=1
    r+=1
    # C11[1,1] += A12[1]*B21[1] -> U[5]=1, V[7]=1, W[4]=1
    U[5, r]=1; V[7, r]=1; W[4, r]=1
    r+=1

    # Block C12 = A11*B12 + A12*B22
    # A11 (2x2) at A[0,1,3,4]. B12 (2x1) at B[2,5]. C12 (2x1) at C[2,5].
    # A11*B12:
    # C12[0] = A11[0,0]*B12[0] + A11[0,1]*B12[1]
    U[0, r]=1; V[2, r]=1; W[2, r]=1
    r+=1
    U[1, r]=1; V[5, r]=1; W[2, r]=1
    r+=1
    # C12[1] = A11[1,0]*B12[0] + A11[1,1]*B12[1]
    U[3, r]=1; V[2, r]=1; W[5, r]=1
    r+=1
    U[4, r]=1; V[5, r]=1; W[5, r]=1
    r+=1
    
    # A12*B22:
    # C12[0] += A12[0]*B22
    U[2, r]=1; V[8, r]=1; W[2, r]=1
    r+=1
    # C12[1] += A12[1]*B22
    U[5, r]=1; V[8, r]=1; W[5, r]=1
    r+=1

    # Block C21 = A21*B11 + A22*B21
    # A21 (1x2) at A[6,7]. B11 (2x2) at B[0,1,3,4]. C21 (1x2) at C[6,7].
    # A21*B11:
    # C21[0] = A21[0]*B11[0,0] + A21[1]*B11[1,0]
    U[6, r]=1; V[0, r]=1; W[6, r]=1
    r+=1
    U[7, r]=1; V[3, r]=1; W[6, r]=1
    r+=1
    # C21[1] = A21[0]*B11[0,1] + A21[1]*B11[1,1]
    U[6, r]=1; V[1, r]=1; W[7, r]=1
    r+=1
    U[7, r]=1; V[4, r]=1; W[7, r]=1
    r+=1

    # A22*B21:
    # C21[0] += A22*B21[0]
    U[8, r]=1; V[6, r]=1; W[6, r]=1
    r+=1
    # C21[1] += A22*B21[1]
    U[8, r]=1; V[7, r]=1; W[7, r]=1
    r+=1

    # Block C22 = A21*B12 + A22*B22
    # A21 (1x2) at A[6,7]. B12 (2x1) at B[2,5]. C22 (1x1) at C[8].
    # A21*B12:
    # C22[0] = A21[0]*B12[0] + A21[1]*B12[1]
    U[6, r]=1; V[2, r]=1; W[8, r]=1
    r+=1
    U[7, r]=1; V[5, r]=1; W[8, r]=1
    r+=1
    # A22*B22:
    # C22[0] += A22*B22
    U[8, r]=1; V[8, r]=1; W[8, r]=1
    r+=1
    
    return U, V, W

def construct():
    t0 = time.time()
    n, m, p = N, M, P
    U3, V3, W3 = _build_block_strassen3x3()
    
    # Total rank
    # Block 1: 3x3x3 (rank 24)
    # Block 2: 3x2x3 (schoolbook 18)
    # Block 3: 1x3x3 (schoolbook 9)
    # Block 4: 1x2x2 (schoolbook 4)
    total_rank = 24 + 18 + 9 + 4
    # Wait, 3x2x3: i in 3, j in 2, k in 3. 3*2*3 = 18.
    # 1x3x3: i in 1, j in 3, k in 3. 1*3*3 = 9.
    # 1x2x2: i in 1, j in 2, k in 2. 1*2*2 = 4.
    # Total 55.
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Map 3x3x3 block (i<3, j<3, k<3) to global indices
    # U[i*5+j], V[j*5+k], W[k*4+i]
    for r in range(24):
        for a in range(9):
            idx_u = a
            # a = i*3+j
            i, j = divmod(a, 3)
            U[i*5+j, r] = U3[a, r]
            
        for b in range(9):
            # b = j*3+k
            j, k = divmod(b, 3)
            V[j*5+k, r] = V3[b, r]
            
        for c in range(9):
            # c = k*3+i
            k, i = divmod(c, 3)
            W[k*4+i, r] = W3[c, r]
            
    # Block 2: i in [0,3), j in [3,5), k in [0,3)
    # Schoolbook 18 terms
    r_start = 24
    c = 0
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                U[i*5+j, r_start+c] = 1.0
                V[j*5+k, r_start+c] = 1.0
                W[k*4+i, r_start+c] = 1.0
                c += 1
    r_start += 18
    
    # Block 3: i in [3,4), j in [0,3), k in [0,3)
    # Schoolbook 9 terms
    c = 0
    for i in range(3, 4):
        for j in range(3):
            for k in range(3):
                U[i*5+j, r_start+c] = 1.0
                V[j*5+k, r_start+c] = 1.0
                W[k*4+i, r_start+c] = 1.0
                c += 1
    r_start += 9
    
    # Block 4: i in [3,4), j in [3,5), k in [3,5)
    # Schoolbook 4 terms
    c = 0
    for i in range(3, 4):
        for j in range(3, 5):
            for k in range(3, 5):
                U[i*5+j, r_start+c] = 1.0
                V[j*5+k, r_start+c] = 1.0
                W[k*4+i, r_start+c] = 1.0
                c += 1
    r_start += 4
    
    assert r_start == total_rank, f"Rank mismatch: {r_start} vs {total_rank}"
    
    if not _validate(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
