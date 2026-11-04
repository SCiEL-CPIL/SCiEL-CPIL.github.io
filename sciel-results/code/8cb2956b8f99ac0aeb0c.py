"""Kronecker product of 3x3 schoolbook and 2x2 Strassen for 6x6 matrix mult."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    
    # 3x3 schoolbook decomposition (rank 27)
    n1, m1, p1 = 3, 3, 3
    r1 = n1 * m1 * p1  # 27
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    col = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1][col] = 1
                V1[j1 * p1 + k1][col] = 1
                W1[k1 * n1 + i1][col] = 1
                col += 1
    
    # 2x2 Strassen decomposition (rank 7)
    # Strassen: C = A*B where A,B,C are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = [[0] * r2 for _ in range(n2 * m2)]  # 4x7
    V2 = [[0] * r2 for _ in range(m2 * p2)]  # 4x7
    W2 = [[0] * r2 for _ in range(p2 * n2)]  # 4x7
    
    # Index convention: U[row, col] where row = i*m + j
    # For 2x2: rows 0=A00, 1=A01, 2=A10, 3=A11
    # V rows: 0=B00, 1=B01, 2=B10, 3=B11
    # W rows: 0=C00, 1=C01, 2=C10, 3=C11 (W[k*n+i] => W[0*2+0]=C00, W[0*2+1]=C01, W[1*2+0]=C10, W[1*2+1]=C11)
    
    # M1 = (A00+A11)(B00+B11): U2: A00=1,A11=1; V2: B00=1,B11=1; W2: C00=1,C11=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)B00: U2: A10=1,A11=1; V2: B00=1; W2: C10=1,C11=-1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M3 = A00(B10-B11): U2: A00=1; V2: B10=1,B11=-1; W2: C01=1,C11=1
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4 = A11(B01-B10): U2: A11=1; V2: B01=1,B10=-1; W2: C00=1,C10=1
    U2[3][3] = 1
    V2[1][3] = 1; V2[2][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (A00+A01)B11: U2: A00=1,A01=1; V2: B11=1; W2: C00=-1,C01=1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10): U2: A10=1,A00=-1; V2: B00=1,B10=1; W2: C11=1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11): U2: A01=1,A11=-1; V2: B01=1,B11=1; W2: C00=1
    U2[1][6] = 1; U2[3][6] = -1
    V2[1][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Kronecker product: 6 = 3 * 2
    # Combined indices: i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2
    # U combined: row = (i1*2+i2)*6 + (j1*2+j2), col = c1*7 + c2
    # V combined: row = (j1*2+j2)*6 + (k1*2+k2), col = c1*7 + c2
    # W combined: row = (k1*2+k2)*6 + (i1*2+i2), col = c1*7 + c2
    
    n, m, p = 6, 6, 6
    rank = r1 * r2  # 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # For each nonzero entry in U1, V1, W1 at column c1,
            # combine with nonzero entries in U2, V2, W2 at column c2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1][c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2][c2]
                            if u2_val == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j][c] = u1_val * u2_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1][c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2][c2]
                            if v2_val == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k][c] = v1_val * v2_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1][c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2][c2]
                            if w2_val == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i][c] = w1_val * w2_val
    
    # Verify: check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but safety
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
