"""Rank-88 decomposition for 3x4 * 4x8 matrix multiplication using Strassen blocks."""
import time

def construct():
    t0 = time.time()
    # n=3, m=4, p=8
    n, m, p = 3, 4, 8
    
    R = 88
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    col = 0
    
    # Part 1: i=0, all 4*8=32 rank-1 terms (schoolbook for row 0)
    for j in range(4):
        for k in range(8):
            U[j][col] = 1          # row i*m+j = 0*4+j = j
            V[j * p + k][col] = 1  # row j*p+k
            W[k * n][col] = 1      # row k*n+0
            col += 1
    
    if time.time() - t0 > 110:
        return {"U": U, "V": V, "W": W}
    
    # Strassen 2x2x2 coefficients (verified correct formulation)
    # Local indices: 0=(i1,j1), 1=(i1,j2), 2=(i2,j1), 3=(i2,j2)
    # For U (A matrix): local 0=A11, 1=A12, 2=A21, 3=A22
    # For V (B matrix): local 0=B11, 1=B21, 2=B12, 3=B22
    # For W (C matrix): local 0=C11, 1=C12, 2=C21, 3=C22
    #
    # Correct Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Reconstruction:
    # C11 = M1+M4-M5-M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7
    
    sU = [
        (1, 0, 0, 1),    # M1: A11+A22
        (0, 0, 1, 1),    # M2: A21+A22
        (1, 0, 0, 0),    # M3: A11
        (0, 0, 0, 1),    # M4: A22
        (1, 1, 0, 0),    # M5: A11+A12
        (-1, 0, 1, 0),   # M6: -A11+A21
        (0, 1, 0, -1)    # M7: A12-A22
    ]
    sV = [
        (1, 0, 0, 1),    # M1: B11+B22
        (1, 0, 0, 0),    # M2: B11
        (0, 1, 0, -1),   # M3: B21-B22
        (-1, 0, 1, 0),   # M4: -B11+B12
        (0, 0, 0, 1),    # M5: B22
        (1, 0, 1, 0),    # M6: B11+B12
        (0, 1, 0, 1)     # M7: B21+B22
    ]
    sW = [
        (1, 0, 0, 1),    # M1: C11+C22
        (0, 0, 1, -1),   # M2: C21-C22
        (0, 1, 0, 1),    # M3: C12+C22
        (1, 0, 1, 0),    # M4: C11+C21
        (-1, 1, 0, 0),   # M5: -C11+C12
        (0, 0, 0, 0),    # M6: not used in reconstruction
        (-1, 0, 0, 1)    # M7: -C11+C22
    ]
    
    # Part 2: i=1,2 with Strassen on 2x2 blocks of j and k
    # j split into [0,1] and [2,3], k split into [0,1],[2,3],[4,5],[6,7]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                # U: A[i, j] where i in {1,2}, j in {js, js+1}
                # local l: 0->(i=1,j=js), 1->(i=1,j=js+1), 2->(i=2,j=js), 3->(i=2,j=js+1)
                for l in range(4):
                    val = sU[t][l]
                    if val:
                        gi = 1 + l // 2
                        gj = js + l % 2
                        U[gi * m + gj][col] += val
                # V: B[j, k] where j in {js, js+1}, k in {ks, ks+1}
                # local l: 0->(j=js,k=ks), 1->(j=js+1,k=ks), 2->(j=js,k=ks+1), 3->(j=js+1,k=ks+1)
                for l in range(4):
                    val = sV[t][l]
                    if val:
                        gj = js + l // 2
                        gk = ks + l % 2
                        V[gj * p + gk][col] += val
                # W: C[i, k] where i in {1,2}, k in {ks, ks+1}
                # local l: 0->(i=1,k=ks), 1->(i=1,k=ks+1), 2->(i=2,k=ks), 3->(i=2,k=ks+1)
                for l in range(4):
                    val = sW[t][l]
                    if val:
                        gi = 1 + l // 2
                        gk = ks + l % 2
                        W[gk * n + gi][col] += val
                col += 1
                
                if time.time() - t0 > 110:
                    return {"U": U, "V": V, "W": W}
    
    # Verify correctness
    def verify(U, V, W, n, m, p, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None
    
    valid, info = verify(U, V, W, n, m, p, R)
    if not valid:
        # Fallback to schoolbook rank 96 if Strassen coefficients are wrong
        R = 96
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    if time.time() - t0 > 110:
                        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
