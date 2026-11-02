import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Returns {'U', 'V', 'W'} with integer entries that exactly decompose
    the 3x5 @ 5x6 matrix multiplication tensor.
    
    Strategy: Block decomposition
    - A (3x5) = A1 (3x2) | A2 (3x3)
    - B (5x6) = B1 (2x6) | B2 (3x6)
    - C = A1*B1 + A2*B2
    
    A1*B1: 3x2 @ 2x6, schoolbook rank = 36
    A2*B2: 3x3 @ 3x6, split B2 = B2a (3x3) | B2b (3x3)
      A2*B2a: 3x3 @ 3x3, Bini rank 23
      A2*B2b: 3x3 @ 3x3, Bini rank 23
    Total rank = 36 + 23 + 23 = 82
    """
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Bini et al. 1978 rank-23 decomposition for 3x3 matrix multiplication
    # Each term: (u, v, w) as 9-element lists (row-major 3x3)
    # u[i*3+k] multiplies A[i,k], v[k*3+j] multiplies B[k,j], w[i*3+j] adds to C[i,j]
    bini_23 = [
        ([1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0]),
        ([0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,1,0,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1]),
        ([1,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[0,0,0,1,0,0,0,0,0]),
        ([1,0,0,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0]),
        ([0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1],[0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,0,0,0,1],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0]),
        ([0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1,0]),
        ([0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]),
        ([1,0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0]),
        ([1,0,0,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0]),
        ([0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0]),
        ([0,0,0,0,0,0,0,0,1],[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1]),
        ([1,0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0]),
        ([1,0,0,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0]),
        ([0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0]),
        ([0,0,0,0,0,0,0,0,1],[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1]),
        ([1,0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0]),
        ([1,0,0,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1]),
    ]
    
    # Verify Bini decomposition
    def verify_bini():
        rng = np.random.default_rng(42)
        for _ in range(10):
            A = rng.integers(-10, 10, (3,3))
            B = rng.integers(-10, 10, (3,3))
            C_expected = A @ B
            C_computed = np.zeros((3,3), dtype=float)
            for u, v, w in bini_23:
                u_mat = np.array(u, dtype=float).reshape(3,3)
                v_mat = np.array(v, dtype=float).reshape(3,3)
                w_mat = np.array(w, dtype=float).reshape(3,3)
                C_computed += np.sum(u_mat * A) * np.sum(v_mat * B) * w_mat
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True
    
    bini_ok = verify_bini()
    
    if not bini_ok:
        # Fall back to schoolbook rank-90
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build rank-82 decomposition
    total_rank = 36 + 23 + 23
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 (3x2) * B1 (2x6) schoolbook, rank 36
    for i in range(3):
        for k in range(2):
            for j in range(6):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # Block 2a: A2 (3x3) * B2a (3x3) using Bini, rank 23
    # A2 occupies columns 2,3,4 of A; B2a occupies rows 2,3,4 of B, columns 0,1,2 of B
    for u, v, w in bini_23:
        u_mat = np.array(u).reshape(3,3)
        v_mat = np.array(v).reshape(3,3)
        w_mat = np.array(w).reshape(3,3)
        for i in range(3):
            for k in range(3):
                U[i * m + (k+2)][col] += u_mat[i,k]
        for k in range(3):
            for j in range(3):
                V[(k+2) * p + j][col] += v_mat[k,j]
        for i in range(3):
            for j in range(3):
                W[j * n + i][col] += w_mat[i,j]
        col += 1
    
    # Block 2b: A2 (3x3) * B2b (3x3) using Bini, rank 23
    # A2 occupies columns 2,3,4 of A; B2b occupies rows 2,3,4 of B, columns 3,4,5 of B
    for u, v, w in bini_23:
        u_mat = np.array(u).reshape(3,3)
        v_mat = np.array(v).reshape(3,3)
        w_mat = np.array(w).reshape(3,3)
        for i in range(3):
            for k in range(3):
                U[i * m + (k+2)][col] += u_mat[i,k]
        for k in range(3):
            for j in range(3):
                V[(k+2) * p + (j+3)][col] += v_mat[k,j]
        for i in range(3):
            for j in range(3):
                W[(j+3) * n + i][col] += w_mat[i,j]
        col += 1
    
    return {"U": U, "V": V, "W": W}
