import numpy as np

def construct():
    n, m, p = 2, 3, 3
    # Try rank 17
    rank = 17
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen part for A[0:2, 0:2] and B[0:2, 0:2]
    # A indices: 0,1, 3,4 (i*3+j)
    # B indices: 0,1, 3,4 (j*3+k)
    # W indices: 0,1, 2,3 (k*2+i) for k=0,1
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] += 1; U[4, 0] += 1
    V[0, 0] += 1; V[4, 0] += 1
    # C00 += M1, C01 += M1, C10 += -M1, C11 += M1
    W[0, 0] += 1 # k=0,i=0
    W[1, 0] += 1 # k=1,i=0
    W[2, 0] += -1 # k=0,i=1
    W[3, 0] += 1 # k=1,i=1
    
    # M2 = (A10+A11)B00
    U[3, 1] += 1; U[4, 1] += 1
    V[0, 1] += 1
    # C00 -= M2, C10 += M2
    W[0, 1] += -1
    W[2, 1] += 1
    
    # M3 = A00(B10-B11)
    U[0, 2] += 1
    V[3, 2] += 1; V[4, 2] += -1
    # C10 += M3, C11 -= M3
    W[2, 2] += 1
    W[3, 2] += -1
    
    # M4 = A11(B01-B00)
    U[4, 3] += 1
    V[1, 3] += 1; V[0, 3] += -1
    # C01 += M4, C11 += M4
    W[1, 3] += 1
    W[3, 3] += 1
    
    # M5 = (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[4, 4] += 1
    # C00 -= M5, C01 -= M5
    W[0, 4] += -1
    W[1, 4] += -1
    
    # M6 = (A10-A00)(B00+B01)
    U[3, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    # C00 += M6, C10 += M6
    W[0, 5] += 1
    W[2, 5] += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] += 1; U[4, 6] += -1
    V[3, 6] += 1; V[4, 6] += 1
    # C01 -= M7, C11 += M7
    W[1, 6] += -1
    W[3, 6] += 1
    
    # Schoolbook for A[:, 2] and B[2, :]
    # A indices: 2, 5 (i*3+2)
    # B indices: 6, 7, 8 (2*3+k)
    # W indices: 0..5 (k*2+i)
    
    # Term 7: A02 B20
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1 # k=0,i=0
    
    # Term 8: A02 B21
    U[2, 8] = 1
    V[7, 8] = 1
    W[1, 8] = 1 # k=1,i=0
    
    # Term 9: A02 B22
    U[2, 9] = 1
    V[8, 9] = 1
    W[4, 9] = 1 # k=2,i=0
    
    # Term 10: A12 B20
    U[5, 10] = 1
    V[6, 10] = 1
    W[2, 10] = 1 # k=0,i=1
    
    # Term 11: A12 B21
    U[5, 11] = 1
    V[7, 11] = 1
    W[3, 11] = 1 # k=1,i=1
    
    # Term 12: A12 B22
    U[5, 12] = 1
    V[8, 12] = 1
    W[5, 12] = 1 # k=2,i=1
    
    # Schoolbook for A[0:2, 0:2] and B[0:2, 2]
    # A indices: 0,1, 3,4
    # B indices: 2, 5 (j*3+2)
    # W indices: 4,5 (k=2, i=0,1)
    
    # Term 13: A00 B02
    U[0, 13] = 1
    V[2, 13] = 1
    W[4, 13] = 1 # k=2,i=0
    
    # Term 14: A01 B12
    U[1, 14] = 1
    V[5, 14] = 1
    W[4, 14] = 1 # k=2,i=0
    
    # Term 15: A10 B02
    U[3, 15] = 1
    V[2, 15] = 1
    W[5, 15] = 1 # k=2,i=1
    
    # Term 16: A11 B12
    U[4, 16] = 1
    V[5, 16] = 1
    W[5, 16] = 1 # k=2,i=1
    
    # Verify
    # Tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Should be 1 if i,j,k match multiplication tensor, else 0?
    # Multiplication tensor: T[i,j,k] = 1 if we compute C[i,k] += A[i,j]B[j,k].
    # So T[i,j,k] is 1 for all i,j,k.
    
    # Check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    # Fall back to schoolbook
                    return construct_schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_schoolbook():
    n, m, p = 2, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
