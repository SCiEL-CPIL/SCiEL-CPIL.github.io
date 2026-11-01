import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    # Strassen decomposition for 2x2x2
    # 8 blocks (a in 0..1, b in 0..3), each rank 7 -> total rank 56
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    for a in range(2):  # A column block / B row block
        for b in range(4):  # B column block / C column block
            base = 7 * (4 * a + b)
            
            # A indices: A[i, 2a], A[i, 2a+1] for i=0,1
            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1
            
            # B indices: B[2a+j, 2b+k]
            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1
            
            # W indices: W[k*n + i] for C[i, k]
            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1
            
            # s=0: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            U[a00, base] += 1; U[a11, base] += 1
            V[b00, base] += 1; V[b11, base] += 1
            W[c00, base] += 1; W[c11, base] += 1
            
            # s=1: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
            U[a10, base+1] += 1; U[a11, base+1] += 1
            V[b00, base+1] += 1
            W[c10, base+1] += 1; W[c11, base+1] -= 1
            
            # s=2: M3 = A00(B01-B11) -> C01 += M3, C11 += M3
            U[a00, base+2] += 1
            V[b01, base+2] += 1; V[b11, base+2] -= 1
            W[c01, base+2] += 1; W[c11, base+2] += 1
            
            # s=3: M4 = A11(B10-B00) -> C00 += M4, C10 += M4
            U[a11, base+3] += 1
            V[b10, base+3] += 1; V[b00, base+3] -= 1
            W[c00, base+3] += 1; W[c10, base+3] += 1
            
            # s=4: M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
            U[a00, base+4] += 1; U[a01, base+4] += 1
            V[b11, base+4] += 1
            W[c00, base+4] -= 1; W[c01, base+4] += 1
            
            # s=5: M6 = (A10-A00)(B00+B01) -> C11 += M6
            U[a10, base+5] += 1; U[a00, base+5] -= 1
            V[b00, base+5] += 1; V[b01, base+5] += 1
            W[c11, base+5] += 1
            
            # s=6: M7 = (A01-A11)(B10+B11) -> C00 += M7
            U[a01, base+6] += 1; U[a11, base+6] -= 1
            V[b10, base+6] += 1; V[b11, base+6] += 1
            W[c00, base+6] += 1
    
    # Verify correctness
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: schoolbook rank 64
        U = np.zeros((nm, 64), dtype=float)
        V = np.zeros((mp, 64), dtype=float)
        W = np.zeros((pn, 64), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
