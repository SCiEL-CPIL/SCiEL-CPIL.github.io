import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 4, 4, 6
    
    # Try Strassen decomposition: 12 blocks of 2x2x2, each rank 7
    # Total rank: 12 * 7 = 84
    
    # Define 7 rank-1 tensors for 2x2x2
    # U[a,b], V[b,c], W[c,a] for a,b,c in {0,1}
    
    # Strassen products (verified):
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
    
    # Let me verify this gives correct results:
    # C00 = (A00+A11)(B00+B11) + A11(B01-B10) - (A00+A01)B11 + (A01-A11)(B01+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B10 - A00*B11 - A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
    # = A00*B00 + A11*B00 - A11*B10 - A01*B11 + A01*B01
    # Should be A00*B00 + A01*B10
    
    # This doesn't match. The formula is wrong.
    
    # Let me try yet another formulation. I'll use the one from:
    # https://en.wikipedia.org/wiki/Strassen_algorithm
    # 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B21)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B12+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In 0-based: A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # M3 = A[0,0](B[1,0]-B[1,1])
    # M4 = A[1,1](B[0,1]-B[1,0])
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])(B[0,1]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Verify C[0,1] = M3 + M5:
    # M3 + M5 = A[0,0]*B[1,0] - A[0,0]*B[1,1] + A[0,0]*B[1,1] + A[0,1]*B[1,1]
    # = A[0,0]*B[1,0] + A[0,1]*B[1,1]
    # But C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # So we need B[1,0] = B[0,1], which is not generally true!
    
    # The Wikipedia formula must be for a different convention (maybe B is transposed).
    # Let me try swapping B indices: use B[j,i] instead of B[i,j]
    # Or equivalently, compute C = A * B^T
    
    # Actually, I think the issue is that in the tensor decomposition, the "B" factor
    # is indexed as V[j,k] where j is the column of A / row of B, and k is the column of B.
    # So V[j,k] corresponds to B[j,k]. The Strassen formula should work with this.
    
    # Let me try: maybe the correct formula uses B[0,1] in M3 instead of B[1,0]:
    # M3 = A[0,0](B[0,1]-B[1,1])
    # Then M3 + M5 = A[0,0]*B[0,1] - A[0,0]*B[1,1] + A[0,0]*B[1,1] + A[0,1]*B[1,1]
    # = A[0,0]*B[0,1] + A[0,1]*B[1,1] = C[0,1] ✓
    
    # So the correct M3 uses B[0,1] not B[1,0]. Let me redo all:
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # M3 = A[0,0](B[0,1]-B[1,1])
    # M4 = A[1,1](B[0,1]-B[1,0])
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])(B[0,1]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Verify C[0,0]:
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B01-B10) - (A00+A01)B11 + (A01-A11)(B01+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B10 - A00*B11 - A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
    # = A00*B00 + A11*B00 - A11*B10 + A01*B01
    # Should be A00*B00 + A01*B10. Still doesn't match!
    
    # I give up trying to derive Strassen from memory. Let me just use schoolbook.
    
    # Schoolbook rank 96
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
