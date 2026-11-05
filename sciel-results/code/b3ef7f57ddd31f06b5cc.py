import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A1 B11 (2x2 by 2x8)
    # A1 is A[0:2, 0:2]
    # B11 is B[0:2, 0:8]
    # Split B11 into 4 blocks of 2x2: B11_0, B11_1, B11_2, B11_3
    
    for block in range(4):
        # Strassen for block
        # A indices: 0, 1, 2, 3 (row-major 2x2)
        # B indices: 0, 1, 2, 3 (row-major 2x2) relative to block
        # B global row offset: 0, 1
        # B global col offset: block*2, block*2+1
        
        # M1 = (a+d)(e+h)
        # U: a+d -> rows 0, 3 of A1. A1 rows are 0, 1, 2, 3?
        # A is 2x3. A1 is cols 0,1.
        # A1[0,0] is U[0], A1[0,1] is U[1], A1[1,0] is U[2], A1[1,1] is U[3].
        
        # We need to add contributions to U, V, W.
        
        # M1
        U[0, col] += 1; U[3, col] += 1
        V[0, col] += 1; V[3, col] += 1 # B11[0,0], B11[1,1] relative to block
        # W: C11, C22
        W[0, col] += 1; W[7, col] += 1 # C[0,0], C[1,1] relative to block?
        # C is 2x8. C11 is C[0, block*2]. C22 is C[1, block*2+1].
        # W index: k*n + i.
        # C[0, block*2] -> k=block*2, i=0 -> W[block*2*2 + 0]
        # C[1, block*2+1] -> k=block*2+1, i=1 -> W[(block*2+1)*2 + 1]
        W[block*2*2 + 0, col] += 1
        W[(block*2+1)*2 + 1, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[2, col] += 1; U[3, col] += 1
        V[0, col] += 1 # B11[0,0]
        # W: C21
        W[block*2*2 + 1, col] += 1
        col += 1
        
        # M3 = a(f-h)
        U[0, col] += 1
        V[1, col] += 1; V[3, col] -= 1 # B11[0,1], B11[1,1]
        # W: C12, C22
        W[block*2*2 + 1, col] += 1
        W[(block*2+1)*2 + 1, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[3, col] += 1
        V[2, col] += 1; V[0, col] -= 1 # B11[1,0], B11[0,0]
        # W: C11, C21
        W[block*2*2 + 0, col] += 1
        W[block*2*2 + 1, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[0, col] += 1; U[1, col] += 1
        V[3, col] += 1 # B11[1,1]
        # W: C11, C12
        W[block*2*2 + 0, col] -= 1
        W[block*2*2 + 1, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[2, col] += 1; U[0, col] -= 1
        V[0, col] += 1; V[1, col] += 1
        # W: C22
        W[(block*2+1)*2 + 1, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[1, col] += 1; U[3, col] -= 1
        V[2, col] += 1; V[3, col] += 1
        # W: C11
        W[block*2*2 + 0, col] += 1
        col += 1
        
    # Part 2: A2 B12 (2x1 by 1x8)
    # A2 is A[0:2, 2]
    # B12 is B[2, 0:8]
    for k in range(8):
        # C[0, k] = A[0,2] B[2,k]
        U[2, col] += 1 # A[0,2] is index 2? A is 2x3. 0,1,2, 3,4,5.
        # A[0,0]=0, A[0,1]=1, A[0,2]=2.
        # A[1,0]=3, A[1,1]=4, A[1,2]=5.
        # So A[0,2] is U[2]. A[1,2] is U[5].
        V[2*8 + k, col] += 1 # B[2,k] is index 2*8+k.
        W[k*2 + 0, col] += 1 # C[0,k]
        col += 1
        
        # C[1, k] = A[1,2] B[2,k]
        U[5, col] += 1
        V[2*8 + k, col] += 1
        W[k*2 + 1, col] += 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
