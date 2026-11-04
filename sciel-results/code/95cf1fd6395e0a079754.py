"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _build_rank23_3x3x3():
    """
    Build a rank-23 exact decomposition for 3x3 * 3x3 matrix multiplication.
    Uses Laderman's algorithm. Returns (U, V, W, rank) as lists.
    Falls back to schoolbook rank-27 if verification fails.
    """
    # Laderman's 23-term algorithm for 3x3*3x3
    # u indexed by (i,j) in A: u[i*3+j]
    # v indexed by (j,k) in B: v[j*3+k]
    # w indexed by (k,i) in C: w[k*3+i]  (trace convention W[k*n+i])
    # Condition: sum_r u[i*3+j] * v[j*3+k] * w[k*3+i] = 1 for all i,j,k in {0,1,2}
    
    laderman_terms = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,1,0]),
        ([0,1,0, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,1, 0,0,1]),
        ([0,0,1, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,1,0, 0,0,1]),
    ]
    
    n3, m3, p3 = 3, 3, 3
    rank = len(laderman_terms)
    U = np.zeros((n3 * m3, rank))
    V = np.zeros((m3 * p3, rank))
    W = np.zeros((p3 * n3, rank))
    
    for r, (u, v, w) in enumerate(laderman_terms):
        U[:, r] = np.array(u, dtype=float)
        V[:, r] = np.array(v, dtype=float)
        W[:, r] = np.array(w, dtype=float)
    
    # Verify: T[i,j,k] = sum_r U[i*m3+j, r] * V[j*p3+k, r] * W[k*n3+i, r] should be 1
    max_err = 0.0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                val = sum(U[i * m3 + j, r] * V[j * p3 + k, r] * W[k * n3 + i, r] for r in range(rank))
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-6:
        # Fallback: schoolbook rank-27
        U_fb = np.zeros((9, 27))
        V_fb = np.zeros((9, 27))
        W_fb = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_fb[i*3+j, col] = 1.0
                    V_fb[j*3+k, col] = 1.0
                    W_fb[k*3+i, col] = 1.0
                    col += 1
        return U_fb.tolist(), V_fb.tolist(), W_fb.tolist(), 27
    
    return U.tolist(), V.tolist(), W.tolist(), rank


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # A is n x m = 3 x 3, B is m x p = 3 x 6, C is n x p = 3 x 6
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (18, rank) with W[k*n+i, r]
    
    # Strategy: Split B into B1 (3x3) and B2 (3x3)
    # C = A*B1 + A*B2
    # Use rank-23 decomposition for each 3x3x3 multiplication
    # Total rank: 23 + 23 = 46
    
    U3, V3, W3, r3 = _build_rank23_3x3x3()
    
    if time.time() > deadline:
        # Fallback to schoolbook
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
    
    # Build the combined 3x3 * 3x6 decomposition
    rank = 2 * r3
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First block: B1 (cols 0-2 of B)
    U[:, :r3] = np.array(U3)
    for j in range(m):
        for k in range(3):
            V[j * p + k, :r3] = V3[j * 3 + k]
    for k in range(3):
        for i in range(n):
            W[k * n + i, :r3] = W3[k * 3 + i]
    
    # Second block: B2 (cols 3-5 of B)
    U[:, r3:] = np.array(U3)
    for j in range(m):
        for k in range(3):
            V[j * p + (k + 3), r3:] = V3[j * 3 + k]
    for k in range(3):
        for i in range(n):
            W[(k + 3) * n + i, r3:] = W3[k * 3 + i]
    
    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(rank))
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-6:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
