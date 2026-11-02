import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def _strassen_7():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) with keys being local indices.
    U local: 0=A00, 1=A01, 2=A10, 3=A11
    V local: 0=B00, 1=B01, 2=B10, 3=B11
    W local: 0=C00, 1=C01, 2=C10, 3=C11
    """
    return [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7
    ]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W uses trace convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    
    # Pick random integer values to test
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = np.array([U[i][r] for i in range(n*m)])
            v_col = np.array([V[j][r] for j in range(m*p)])
            w_col = np.array([W[k][r] for k in range(p*n)])

            uA = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    uA[i] += U[i*m + j][r] * A[i, j]
            
            vB = np.zeros(p)
            for j in range(m):
                for k in range(p):
                    vB[k] += V[j*p + k][r] * B[j, k]
            
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W[k*n + i][r] * uA[i] * vB[k]
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = 7 * num_blocks  # 49

    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    terms = _strassen_7()

    col = 0
    for b in range(num_blocks):
        # V global offsets for block b: B00->2b, B01->2b+1, B10->14+2b, B11->14+2b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        # W global offsets for block b: C00->4b, C01->4b+2, C10->4b+1, C11->4b+3
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]

        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1

    return {"U": U, "V": V, "W": W}
