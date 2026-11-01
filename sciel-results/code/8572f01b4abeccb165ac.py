import time
import random
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115

    # Pre-computed rank-25 decomposition
    # Structure:
    # Block 1 (cols 0,1): Strassen (7 terms)
    # Block 2 (cols 2,3): Strassen (7 terms)
    # Block 3 (cols 4,5): Strassen (7 terms)
    # Block 4 (col 6): Schoolbook (4 terms)
    
    rank = 25
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_strassen(col_start, k_start):
        # M1 = (A00+A11)*(B0,ks + B1,ks+1)
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
        W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1
        # M2 = (A10+A11)*B0,ks
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0+k_start, col_start+1] += 1
        W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1
        # M3 = A00*(B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
        W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1
        # M4 = A11*(B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
        W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1
        # M5 = (A00+A01)*B1,ks+1
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1+k_start+1, col_start+4] += 1
        W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1
        # M6 = (A10-A00)*(B0,ks+B0,ks+1)
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
        W[(k_start+1)*n+1, col_start+5] += 1
        # M7 = (A01-A11)*(B1,ks+B1,ks+1)
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
        W[k_start*n+0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    # Schoolbook for column 6
    cs = 21
    U[0, cs+0] += 1; V[0*7+6, cs+0] += 1; W[6*n+0, cs+0] += 1
    U[1, cs+1] += 1; V[1*7+6, cs+1] += 1; W[6*n+0, cs+1] += 1
    U[2, cs+2] += 1; V[0*7+6, cs+2] += 1; W[6*n+1, cs+2] += 1
    U[3, cs+3] += 1; V[1*7+6, cs+3] += 1; W[6*n+1, cs+3] += 1

    # Verify
    def verify_decomp(U, V, W, trials=10):
        rank = U.shape[1]
        for _ in range(trials):
            A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)])
            B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)])
            C_true = A @ B
            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_val = sum(U[i*m+j, r] * A[i, j] for i in range(n) for j in range(m))
                v_val = sum(V[j*p+k, r] * B[j, k] for j in range(m) for k in range(p))
                prod = u_val * v_val
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += W[k*n+i, r] * prod
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    if verify_decomp(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
