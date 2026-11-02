import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of two 2x2 Strassen decompositions.
    """
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen factors
    # U2: (4, 7) -> indices i,j for A
    # V2: (4, 7) -> indices j,k for B
    # W2: (4, 7) -> indices k,i for C (note: task uses W[k*n+i], which is k,i)
    
    # Strassen 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indices for 2x2: i,j in {0,1}
    # i=0, j=0 -> A11
    # i=0, j=1 -> A12
    # i=1, j=0 -> A21
    # i=1, j=1 -> A22
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: A11, A22 ; B11, B22 ; C11, C22, C12, C21 contributions
    # U2[0,0]=1 (A11), U2[3,0]=1 (A22)
    # V2[0,0]=1 (B11), V2[3,0]=1 (B22)
    # W2[0,0]=1 (C11 from A11B11? No, W is for output C. 
    # The term is U_ij V_jk W_ki.
    # For M1: (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This contributes to C11 (A11B11+A22B22? No, C11 = A11B11+A12B21. 
    # Strassen formula:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's populate U2, V2, W2 based on standard Strassen.
    # Col 0 (M1):
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C11 and C22 get M1. Wait.
    # C11 coeff for M1 is 1. C22 coeff for M1 is 1.
    # C12 coeff for M1 is 0. C21 coeff for M1 is 0.
    # W2 index is k*n+i -> k,i.
    # C11 is i=0, k=0 -> index 0.
    # C22 is i=1, k=1 -> index 3.
    # So W2[0][0]=1, W2[3][0]=1 is correct for M1.
    
    # Col 1 (M2): (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # C21 = M2 + M4. C21 is i=1, k=0 -> index 2.
    W2[2][1] = 1
    
    # Col 2 (M3): A11(B12-B22)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    # C12 = M3 + M5. C12 is i=0, k=1 -> index 1.
    W2[1][2] = 1
    
    # Col 3 (M4): A22(B21-B11)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    # C11 = M1 - M4 + ... C11 is index 0.
    # C21 = M2 + M4. C21 is index 2.
    W2[0][3] = -1; W2[2][3] = 1
    
    # Col 4 (M5): (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # C11 = ... + M5 ... C11 index 0.
    # C12 = M3 + M5. C12 index 1.
    W2[0][4] = 1; W2[1][4] = 1
    
    # Col 5 (M6): (A21-A11)(B11+B12)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # C22 = M1 - M2 + M3 + M6. C22 index 3.
    W2[3][5] = 1
    
    # Col 6 (M7): (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    # C11 = M1 - M4 + M5 - M7. C11 index 0.
    W2[0][6] = -1
    
    # Now construct 4x4 factors via Kronecker product
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Indices:
    # i = 2*i1 + i2
    # j = 2*j1 + j2
    # k = 2*k1 + k2
    # r = 7*r1 + r2
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1 + j2
                    # U index is i*4+j
                    u_idx = i*4 + j
                    for r1 in range(7):
                        val1 = U2[i1*2+j1][r1]
                        if val1 == 0: continue
                        for r2 in range(7):
                            val2 = U2[i2*2+j2][r2]
                            if val2 == 0: continue
                            r = 7*r1 + r2
                            U[u_idx][r] += val1 * val2
    
    for j1 in range(2):
        for j2 in range(2):
            j = 2*j1 + j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1 + k2
                    # V index is j*4+k
                    v_idx = j*4 + k
                    for r1 in range(7):
                        val1 = V2[j1*2+k1][r1]
                        if val1 == 0: continue
                        for r2 in range(7):
                            val2 = V2[j2*2+k2][r2]
                            if val2 == 0: continue
                            r = 7*r1 + r2
                            V[v_idx][r] += val1 * val2
    
    for k1 in range(2):
        for k2 in range(2):
            k = 2*k1 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1 + i2
                    # W index is k*4+i
                    w_idx = k*4 + i
                    for r1 in range(7):
                        val1 = W2[k1*2+i1][r1]
                        if val1 == 0: continue
                        for r2 in range(7):
                            val2 = W2[k2*2+i2][r2]
                            if val2 == 0: continue
                            r = 7*r1 + r2
                            W[w_idx][r] += val1 * val2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
