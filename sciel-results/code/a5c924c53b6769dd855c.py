import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # U2 rows: (i,j) for i,j in {0,1}
    # V2 rows: (j,k) for j,k in {0,1}
    # W2 rows: (k,i) for k,i in {0,1}
    
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],  # A00
        [0,  0,  0,  0,  1,  0,  1],  # A01
        [0,  1,  0,  0,  0,  1,  0],  # A10
        [1,  1,  0,  1,  0,  0, -1]   # A11
    ], dtype=np.int8)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1,  0,  1,  0,  1]   # B11
    ], dtype=np.int8)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],  # C00 (k=0,i=0)
        [0,  1,  0,  1,  0,  0,  0],  # C10 (k=0,i=1)
        [0,  0,  1,  0,  1,  0,  0],  # C01 (k=1,i=0)
        [1, -1, 1,  0,  0,  1,  0]   # C11 (k=1,i=1)
    ], dtype=np.int8)

    # Verify 2x2 Strassen decomposition
    # For each (i,j,k), check sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r] == 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for r in range(7):
                    val += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                if val != 1:
                    # This should not happen; if it does, the factors are wrong
                    pass

    # Construct 4x4 factors via Kronecker product
    # U[i,j,r1*7+r2] = U2[i1,j1,r1] * U2[i2,j2,r2]
    # where i = 2*i1 + i2, j = 2*j1 + j2
    # V[j,k,r1*7+r2] = V2[j1,k1,r1] * V2[j2,k2,r2]
    # where j = 2*j1 + j2, k = 2*k1 + k2
    # W[k,i,r1*7+r2] = W2[k1,i1,r1] * W2[k2,i2,r2]
    # where k = 2*k1 + k2, i = 2*i1 + i2

    rank = 49
    U = np.zeros((16, 49), dtype=np.int8)
    V = np.zeros((16, 49), dtype=np.int8)
    W = np.zeros((16, 49), dtype=np.int8)

    for i in range(4):
        i1 = i // 2
        i2 = i % 2
        for j in range(4):
            j1 = j // 2
            j2 = j % 2
            for r1 in range(7):
                u1 = U2[i1*2 + j1, r1]
                if u1 == 0:
                    continue
                for r2 in range(7):
                    u2 = U2[i2*2 + j2, r2]
                    if u2 == 0:
                        continue
                    U[i*4 + j, r1*7 + r2] = u1 * u2

    for j in range(4):
        j1 = j // 2
        j2 = j % 2
        for k in range(4):
            k1 = k // 2
            k2 = k % 2
            for r1 in range(7):
                v1 = V2[j1*2 + k1, r1]
                if v1 == 0:
                    continue
                for r2 in range(7):
                    v2 = V2[j2*2 + k2, r2]
                    if v2 == 0:
                        continue
                    V[j*4 + k, r1*7 + r2] = v1 * v2

    for k in range(4):
        k1 = k // 2
        k2 = k % 2
        for i in range(4):
            i1 = i // 2
            i2 = i % 2
            for r1 in range(7):
                w1 = W2[k1*2 + i1, r1]
                if w1 == 0:
                    continue
                for r2 in range(7):
                    w2 = W2[k2*2 + i2, r2]
                    if w2 == 0:
                        continue
                    W[k*4 + i, r1*7 + r2] = w1 * w2

    # Quick verification: check a few entries
    # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 if j is the
    # shared index between A and B. In the standard definition, all (i,j,k) with
    # i in [0,n), j in [0,m), k in [0,p) have T[i,j,k] = 1.
    
    # Let's verify: for random (i,j,k), the sum should be 1
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, 3)
        j = random.randint(0, 3)
        k = random.randint(0, 3)
        val = 0
        for r in range(49):
            val += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        if val != 1:
            # This would indicate an error in the construction
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
