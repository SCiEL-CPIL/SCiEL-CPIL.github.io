import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Apply Strassen 2x2x2 to four non-overlapping blocks:
    # Block (di, dj, dk) for di in {0}, dj in {0,2}, dk in {0,2}
    # Each block covers i in {di, di+1}, j in {dj, dj+1}, k in {dk, dk+1}
    # 4 blocks * 7 terms = 28, plus schoolbook for i=2: 16 terms = 44 total
    
    R = 44
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    for di in [0]:
        for dj in [0, 2]:
            for dk in [0, 2]:
                # Strassen 2x2x2 for A[di:di+2, dj:dj+2] * B[dj:dj+2, dk:dk+2] = C[di:di+2, dk:dk+2]
                # M1 = (A00+A11)(B00+B11)
                U[di*m+dj, col] += 1
                U[(di+1)*m+(dj+1), col] += 1
                V[dj*p+dk, col] += 1
                V[(dj+1)*p+(dk+1), col] += 1
                W[dk*n+di, col] += 1
                W[(dk+1)*n+(di+1), col] += 1
                col += 1
                
                # M2 = (A10+A11)B00
                U[(di+1)*m+dj, col] += 1
                U[(di+1)*m+(dj+1), col] += 1
                V[dj*p+dk, col] += 1
                W[dk*n+(di+1), col] += 1
                col += 1
                
                # M3 = A00(B10-B11)
                U[di*m+dj, col] += 1
                V[(dj+1)*p+dk, col] += 1
                V[(dj+1)*p+(dk+1), col] += -1
                W[dk*n+di, col] += 1
                W[(dk+1)*n+di, col] += 1
                col += 1
                
                # M4 = A11(B01-B00)
                U[(di+1)*m+(dj+1), col] += 1
                V[dj*p+(dk+1), col] += 1
                V[dj*p+dk, col] += -1
                W[dk*n+di, col] += -1
                W[(dk+1)*n+di, col] += 1
                col += 1
                
                # M5 = (A00+A10)B11
                U[di*m+dj, col] += 1
                U[(di+1)*m+dj, col] += 1
                V[(dj+1)*p+(dk+1), col] += 1
                W[dk*n+di, col] += 1
                W[dk*n+(di+1), col] += 1
                col += 1
                
                # M6 = (A10-A00)(B00+B10)
                U[(di+1)*m+dj, col] += 1
                U[di*m+dj, col] += -1
                V[dj*p+dk, col] += 1
                V[(dj+1)*p+dk, col] += 1
                W[dk*n+(di+1), col] += 1
                W[(dk+1)*n+(di+1), col] += 1
                col += 1
                
                # M7 = (A11-A01)(B01+B11)
                U[(di+1)*m+(dj+1), col] += 1
                U[di*m+(dj+1), col] += -1
                V[dj*p+(dk+1), col] += 1
                V[(dj+1)*p+(dk+1), col] += 1
                W[dk*n+di, col] += -1
                W[(dk+1)*n+di, col] += 1
                W[(dk+1)*n+(di+1), col] += 1
                col += 1
    
    # Schoolbook for remaining: i=2, all j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    assert col == R, f"Rank mismatch: {col} vs {R}"
    
    # Verify
    def verify(U, V, W, tol=1e-8):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k2*n+i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
